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
	"strconv"
	"sync"
	"syscall"
	"time"

	"golang.org/x/net/netutil"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/localvec"
	"github.com/firebase/genkit/go/plugins/ollama"
)

// addDocumentRequest struct for adding document to vector store
type addDocumentRequest struct {
	Document string
}

// queryRequest structs for unmarshaling query json
type queryRequest struct {
	Query string
}

// queryResponse structs for unmarshaling query json
type queryResponse struct {
	Query    string
	Response string
	Context  string
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

var (
	mu              *sync.Mutex
	indexer         ai.Indexer
	retriever       ai.Retriever
	llmModel        ai.Model
	connections     int
	ollamaServerURL string
	llmModelName    string
	webServerPort   string
)

// Rag model query template to be provided for the llm
const ragQueryTemplate = `
You are a helpful agent that answers my questions with the help of the context if provided.

Question:
%s

Context:
%s
`

func init() {
	var err error

	// Remove prefix and default flags for standard library logger
	log.SetPrefix("")
	log.SetFlags(0)

	ollamaServerURL = cmp.Or(os.Getenv("OLLAMA_SERVER_URL"), "http://localhost:11434")
	llmModelName = cmp.Or(os.Getenv("LLM_MODEL_NAME"), "llama3.2:1b")
	webServerPort = cmp.Or(os.Getenv("WEB_SERVER_PORT"), ":8000")
	connections, err = strconv.Atoi(cmp.Or(os.Getenv("WEB_SERVER_CONNECTIONS"), "3"))
	if err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}

	slog.Info("Environment parameters", "OLLAMA_SERVER_URL", ollamaServerURL, "LLM_MODEL_NAME", llmModelName, "WEB_SERVER_PORT", webServerPort, "WEB_SERVER_CONNECTIONS", connections)

	// Initalize ollama package with server address
	err = ollama.Init(context.Background(), &ollama.Config{
		ServerAddress: ollamaServerURL,
	})
	if err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}

	// Declare new ollama llm client
	llmModel = ollama.DefineModel(ollama.ModelDefinition{
		Name: llmModelName,
		Type: "chat",
	},
		&ai.ModelCapabilities{
			Multiturn:  true,
			SystemRole: true,
			Tools:      false,
			Media:      false,
		},
	)

	// Initialize localvec package
	if err := localvec.Init(); err != nil {
		log.Fatal(err)
	}

	// Get embeddings of the llm client
	emb := ollama.DefineEmbedder(ollamaServerURL, llmModelName)
	if emb == nil {
		slog.Error("Could not retrieve embeddings")
		os.Exit(1)
	}

	// Create new local vector store with embeddings
	indexer, retriever, err = localvec.DefineIndexerAndRetriever("doc", localvec.Config{Dir: "temp", Embedder: emb})
	if err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}

	// Initialize the mutex which ensures only one request is handled at a time
	mu = &sync.Mutex{}
}

// addDocumentHandler is used to store information provided to the vector store
func addDocumentHandler(w http.ResponseWriter, req *http.Request) {
	addDocReq := &addDocumentRequest{}

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
	err = ai.Index(context.Background(), indexer, ai.WithIndexerDocs(ai.DocumentFromText(addDocReq.Document, nil)))
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
	queryReqDoc := ai.DocumentFromText(queryReq.Query, nil)
	contextDocs, err := ai.Retrieve(context.Background(), retriever, ai.WithRetrieverDoc(queryReqDoc))
	if err != nil {
		http.Error(w, "Error searching for context in vector store", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}

	// Append all the returned context docs with newline
	var contextStr string
	for _, doc := range contextDocs.Documents {
		contextStr += doc.Content[0].Text + "\n"
	}

	// Create the rag query to be passed to the llm
	ragQuery := fmt.Sprintf(ragQueryTemplate, queryReq.Query, contextStr)

	// Prompt the llm with context and query
	ragResponse, err := ai.GenerateText(context.Background(), llmModel, ai.WithTextPrompt(ragQuery))
	if err != nil {
		http.Error(w, "Error querying llm", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}

	// Marshal the llm response to json
	js, err := json.Marshal(&queryResponse{
		Query:    queryReq.Query,
		Context:  contextStr,
		Response: ragResponse,
	})
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
	http.HandleFunc("POST /add", addDocumentHandler)
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

		// Create a limit listener to handle only given number of connections at a time
		limitListener := netutil.LimitListener(listener, connections)

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
