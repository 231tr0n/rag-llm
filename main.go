package main

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
	"mime"
	"net"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/net/netutil"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores/weaviate"
)

// Document addition request structs for unmarshaling request json
type document struct {
	Information string
}
type addDocumentsRequest struct {
	Documents []document
}

// Query request structs for unmarshaling query json
type queryRequest struct {
	Query string
}

// WrapperResponseWriter for logging functionality
type wrapperResponseWriter struct {
	http.ResponseWriter
	statusCode int
}

// WriteHeader method for writing storing the statuscode
func (w *wrapperResponseWriter) WriteHeader(statusCode int) {
	w.ResponseWriter.WriteHeader(statusCode)
	w.statusCode = statusCode
}

// Mutex for running a single request at a time
var mu *sync.Mutex

// VectorDatabase client
var vectorStore weaviate.Store

// Ollama client
var llmClient *ollama.LLM

// Context used by both ollama client and vector database
var ragCtx context.Context

// Rag model query template to be provided for the llm
const ragQueryTemplate = `
I will ask you a question and will provide some additional context information.
Assume this context information is factual and correct, as part of internal
documentation.
If the question relates to the context, answer it using the context.
If the question does not relate to the context, answer it as normal.

For example, let's say the context has nothing in it about tropical flowers;
then if I ask you about tropical flowers, just answer what you know about them
without referring to the context.

For example, if the context does mention minerology and I ask you about that,
provide information from the context along with general knowledge.

Question:
%s

Context:
%s
`

var ollamaServerURL = cmp.Or(os.Getenv("OLLAMA_SERVER_URL"), "http://localhost:11434")
var llmModelName = cmp.Or(os.Getenv("LLM_MODEL_NAME"), "llama3.2")
var weaviateServerURL = cmp.Or(os.Getenv("WEAVIATE_SERVER_URL"), "localhost:8080")
var webServerPort = cmp.Or(os.Getenv("WEB_SERVER_PORT"), ":8000")

func init() {
	// Remove prefix and default flags for standard library logger
	log.SetPrefix("")
	log.SetFlags(0)

	// Log env variables
	slog.Info("Environment parameters", "OLLAMA_SERVER_URL", ollamaServerURL, "LLM_MODEL_NAME", llmModelName, "WEAVIATE_SERVER_URL", weaviateServerURL, "WEB_SERVER_PORT", webServerPort)

	// Declare new ollama llm client
	var err error
	llmClient, err = ollama.New(ollama.WithServerURL(ollamaServerURL), ollama.WithModel(llmModelName))
	if err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}

	// Get embeddings of the llm client
	emb, err := embeddings.NewEmbedder(llmClient)
	if err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}

	// Create new vector store client with embeddings of llm
	vectorStore, err = weaviate.New(
		weaviate.WithEmbedder(emb),
		weaviate.WithScheme("http"),
		weaviate.WithHost(weaviateServerURL),
		weaviate.WithIndexName("Document"),
	)
	if err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}

	// Create new context which is used by both llm client and vector store client
	ragCtx = context.Background()

	// Initialize a mutex
	mu = &sync.Mutex{}
}

// addDocumentsHandler is used to store information provided to the vector store
func addDocumentsHandler(w http.ResponseWriter, req *http.Request) {
	addDocReq := &addDocumentsRequest{}

	// Check if content-type is correct in the http request
	contentType := req.Header.Get("Content-Type")
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil || mediaType != "application/json" {
		http.Error(w, "Wrong content-type", http.StatusBadRequest)
		slog.Error(err.Error())
		return
	}

	// Decode request body
	dec := json.NewDecoder(req.Body)
	dec.DisallowUnknownFields()
	err = dec.Decode(addDocReq)
	if err != nil {
		http.Error(w, "Wrong json format for document addition request", http.StatusBadRequest)
		slog.Error(err.Error())
		return
	}

	// Add request documents to vector store
	var wvDocs []schema.Document
	for _, doc := range addDocReq.Documents {
		wvDocs = append(wvDocs, schema.Document{PageContent: doc.Information})
	}
	_, err = vectorStore.AddDocuments(ragCtx, wvDocs)
	if err != nil {
		http.Error(w, "Error adding documents to vector store", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}
}

func queryHandler(w http.ResponseWriter, req *http.Request) {
	queryReq := &queryRequest{}

	// Check if content-type is correct in the http request
	contentType := req.Header.Get("Content-Type")
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil || mediaType != "application/json" {
		http.Error(w, "Wrong content-type", http.StatusBadRequest)
		slog.Error(err.Error())
		return
	}

	// Decode request body
	dec := json.NewDecoder(req.Body)
	dec.DisallowUnknownFields()
	err = dec.Decode(queryReq)
	if err != nil {
		http.Error(w, "Wrong json format for query request", http.StatusBadRequest)
		slog.Error(err.Error())
		return
	}

	// Search if query with similarity exists in vector store
	docs, err := vectorStore.SimilaritySearch(ragCtx, queryReq.Query, 3)
	if err != nil {
		http.Error(w, "Error searching for similarity in vector store", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}
	var docsContents []string
	for _, doc := range docs {
		docsContents = append(docsContents, doc.PageContent)
	}

	// Prompt the llm with context and query
	ragQuery := fmt.Sprintf(ragQueryTemplate, queryReq.Query, strings.Join(docsContents, "\n"))
	ragResponse, err := llms.GenerateFromSinglePrompt(ragCtx, llmClient, ragQuery)
	if err != nil {
		http.Error(w, "Error querying llm", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}

	// Marshal the llm response to json
	js, err := json.Marshal(ragResponse)
	if err != nil {
		http.Error(w, "Error marshaling response from llm", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}

	// Send marshalled json as response with 200 status
	w.Header().Set("Content-Type", "application/json")
	_, err = w.Write(js)
	if err != nil {
		http.Error(w, "Error writing response", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}
}

func main() {
	// Recover function incase any panic happens
	defer func() {
		if err := recover(); err != nil {
			err, ok := err.(error)
			if ok {
				slog.Error(err.Error())
				os.Exit(1)
			}
			slog.Error(err.Error())
			os.Exit(1)
		}
	}()

	// Handlers for the http server
	http.HandleFunc("POST /query", queryHandler)
	http.HandleFunc("POST /add", addDocumentsHandler)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	})

	// Creating server configuration with logging
	server := http.Server{
		// Addr:           port,
		MaxHeaderBytes: 50 << 20,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			startTime := time.Now()
			wr := &wrapperResponseWriter{
				ResponseWriter: w,
				statusCode:     http.StatusBadRequest,
			}
			time.Sleep(time.Second)
			http.DefaultServeMux.ServeHTTP(wr, r)
			if wr.statusCode > 399 {
				slog.Warn("REQUEST", "method", r.Method, "ip", r.RemoteAddr, "time", time.Since(startTime).Round(time.Second), "status", wr.statusCode, "path", r.URL.Path)
				mu.Unlock()
				return
			}
			slog.Info("REQUEST", "method", r.Method, "ip", r.RemoteAddr, "time", time.Since(startTime).Round(time.Second), "status", wr.statusCode, "path", r.URL.Path)
			mu.Unlock()
		}),
	}

	// Seperate goroutine for running the listener which is synchronous and blocking
	go func() {
		// Create a listener on port
		listener, err := net.Listen("tcp", webServerPort)
		if err != nil {
			panic(err)
		}
		defer listener.Close()

		// Create a limit listener to handle only 3 http connections at a time
		limitListener := netutil.LimitListener(listener, 3)

		// Serve http requests on the limit listener
		if err := server.Serve(limitListener); err != nil {
			if err != http.ErrServerClosed {
				panic(err)
			}
			slog.Warn(err.Error())
		}
	}()
	slog.Info("Started server on", "port", webServerPort)

	// Channel for interrupts
	interrupt := make(chan os.Signal, 1)
	defer close(interrupt)

	// Goroutine to listen for interrupts
	go (func() {
		signal.Notify(interrupt, os.Interrupt, syscall.SIGTERM, syscall.SIGINT)
	})()

	// Waiting for any interrupt
	val, ok := <-interrupt
	if !ok {
		log.Println()
		slog.Warn("Channel closed before receiving os signal")
	} else {
		log.Println()
		slog.Warn(val.String() + " received")
	}

	// Creating context with 5 second timeout to close the server
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Shutting down server within 5 seconds with context
	if err := server.Shutdown(ctx); err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}
	slog.Info("Shutdown server on", "port", webServerPort)
}
