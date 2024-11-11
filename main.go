package main

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
	"mime"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

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

func init() {
	// Remove prefix and default flags for standard library logger
	log.SetPrefix("")
	log.SetFlags(0)

	// Declare new ollama llm client
	var err error
	llmClient, err = ollama.New(ollama.WithServerURL(cmp.Or(os.Getenv("OLLAMA_SERVER_URL"), "http://localhost:11434")), ollama.WithModel(cmp.Or(os.Getenv("MODEL_NAME"), "llama3.2")))
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
		weaviate.WithHost("localhost"+cmp.Or(os.Getenv("WEAVIATE_SERVER_PORT"), ":8080")),
		weaviate.WithIndexName("Document"),
	)
	if err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}

	// Create new context which is used by both llm client and vector store client
	ragCtx = context.Background()
}

// addDocumentsHandler is used to store information provided to the vector store
func addDocumentsHandler(w http.ResponseWriter, req *http.Request) {
	addDocReq := &addDocumentsRequest{}

	contentType := req.Header.Get("Content-Type")
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil || mediaType != "application/json" {
		http.Error(w, "Wrong content-type", http.StatusBadRequest)
		slog.Error(err.Error())
		return
	}

	dec := json.NewDecoder(req.Body)
	dec.DisallowUnknownFields()
	err = dec.Decode(addDocReq)
	if err != nil {
		http.Error(w, "Wrong json format for document addition request", http.StatusBadRequest)
		slog.Error(err.Error())
		return
	}

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

	contentType := req.Header.Get("Content-Type")
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil || mediaType != "application/json" {
		http.Error(w, "Wrong content-type", http.StatusBadRequest)
		slog.Error(err.Error())
		return
	}

	dec := json.NewDecoder(req.Body)
	dec.DisallowUnknownFields()
	err = dec.Decode(queryReq)
	if err != nil {
		http.Error(w, "Wrong json format for query request", http.StatusBadRequest)
		slog.Error(err.Error())
		return
	}

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

	ragQuery := fmt.Sprintf(ragQueryTemplate, queryReq.Query, strings.Join(docsContents, "\n"))
	ragResponse, err := llms.GenerateFromSinglePrompt(ragCtx, llmClient, ragQuery)
	if err != nil {
		http.Error(w, "Error querying llm", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}

	js, err := json.Marshal(ragResponse)
	if err != nil {
		http.Error(w, "Error marshaling response from llm", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}
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

	// Getting port from defaults or env variable
	port := cmp.Or(os.Getenv("WEB_SERVER_PORT"), ":8000")

	// Creating server configuration with logging
	server := http.Server{
		Addr:           port,
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
		if err := server.ListenAndServe(); err != nil {
			if err != http.ErrServerClosed {
				panic(err)
			}
			slog.Warn(err.Error())
		}
	}()
	slog.Info("Started server on", "port", port)

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
	slog.Info("Shutdown server on", "port", port)
}
