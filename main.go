package main

import (
	"cmp"
	"context"
	_ "embed"
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

//go:embed index.html
var htmlWebPage string

// addDocumentRequest struct for adding document to vector store
type addDocumentRequest struct {
	Document string
}

// chatMessage struct for unmarshaling chat history json
type chatMessage struct {
	Role string
	Text string
}

// chatRequest struct for unmarshaling query json
type chatRequest struct {
	History []chatMessage
}

// WrapperResponseWriter struct for logging functionality
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
	connections     int
	indexer         ai.Indexer
	llmModel        ai.Model
	llmModelName    string
	mu              *sync.Mutex
	ollamaServerURL string
	retriever       ai.Retriever
	webServerPort   string
)

// systemRole defines the behaviour of the model
var systemRole = ai.NewTextMessage(
	ai.RoleSystem,
	"You are a chatbot that chats with users and answers their questions based on context provided by the user.",
)

// Rag model query template to be provided for the llm
const ragQueryTemplate = `Context:
%s

Question:
%s
`

func init() {
	var err error

	// Remove prefix and default flags for standard library logger
	log.SetPrefix("")
	log.SetFlags(0)

	ollamaServerURL = cmp.Or(os.Getenv("OLLAMA_SERVER_URL"), "http://localhost:11434")
	llmModelName = cmp.Or(os.Getenv("LLM_MODEL_NAME"), "llama3.2:latest")
	webServerPort = cmp.Or(os.Getenv("WEB_SERVER_PORT"), ":8000")
	connections, err = strconv.Atoi(cmp.Or(os.Getenv("WEB_SERVER_CONNECTIONS"), "3"))
	if err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}

	slog.Info(
		"Environment parameters",
		"OLLAMA_SERVER_URL",
		ollamaServerURL,
		"LLM_MODEL_NAME",
		llmModelName,
		"WEB_SERVER_PORT",
		webServerPort,
		"WEB_SERVER_CONNECTIONS",
		connections,
	)

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
	indexer, retriever, err = localvec.DefineIndexerAndRetriever(
		"doc",
		localvec.Config{Dir: "temp", Embedder: emb},
	)
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

	if addDocReq.Document == "" {
		http.Error(w, "Empty document", http.StatusBadRequest)
		slog.Error("Empty document")
		return
	}

	// Add request documents to vector store
	err = ai.Index(
		context.Background(),
		indexer,
		ai.WithIndexerDocs(ai.DocumentFromText(addDocReq.Document, nil)),
	)
	if err != nil {
		http.Error(w, "Error adding documents to vector store", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}
}

func chatHandler(w http.ResponseWriter, req *http.Request) {
	chatReq := &chatRequest{}

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
	err = dec.Decode(chatReq)
	if err != nil {
		http.Error(w, "Wrong json format for query request", http.StatusBadRequest)
		slog.Error(err.Error())
		return
	}

	if len(chatReq.History) == 0 {
		http.Error(w, "Empty chat history", http.StatusBadRequest)
		slog.Error("Empty chat history")
		return
	}

	if chatReq.History[len(chatReq.History)-1].Role != "user" {
		http.Error(w, "Last message is not sent by user", http.StatusBadRequest)
		slog.Error("Last message is not sent by user")
		return
	}

	historyMessages := make([]*ai.Message, len(chatReq.History))

	for index, historyMessage := range chatReq.History {
		if historyMessage.Role == "user" {
			historyTextDoc := ai.DocumentFromText(historyMessage.Text, nil)
			contextDocs, err := ai.Retrieve(
				context.Background(),
				retriever,
				ai.WithRetrieverDoc(historyTextDoc),
			)
			if err != nil {
				http.Error(
					w,
					"Error searching for context in vector store",
					http.StatusInternalServerError,
				)
				slog.Error(err.Error())
				return
			}

			ragQuery := ""

			if len(contextDocs.Documents) != 0 {
				// Append all the returned context docs with newline
				var contextStr string
				for _, doc := range contextDocs.Documents {
					contextStr += doc.Content[0].Text + "\n"
				}

				ragQuery = fmt.Sprintf(ragQueryTemplate, contextStr, historyMessage.Text)
			} else {
				ragQuery = historyMessage.Text
			}

			historyMessages[index] = ai.NewTextMessage(ai.RoleUser, ragQuery)
		} else if historyMessage.Role == "model" {
			historyMessages[index] = ai.NewTextMessage(ai.RoleModel, historyMessage.Text)
		} else {
			http.Error(
				w,
				"Wrong role sent",
				http.StatusInternalServerError,
			)
			slog.Error("Wrong role sent")
			return
		}
	}

	historyMessages = append([]*ai.Message{systemRole}, historyMessages...)

	out, err := json.Marshal(historyMessages)
	if err != nil {
		slog.Error(err.Error())
		return
	}

	// Prompt the llm with context and query
	ragResponse, err := ai.GenerateText(
		context.Background(),
		llmModel,
		ai.WithMessages(historyMessages...),
	)
	if err != nil {
		http.Error(w, "Error querying llm", http.StatusInternalServerError)
		slog.Error(err.Error())
		return
	}

	fmt.Println("----------------------------------")
	slog.Info(string(out))
	slog.Info(ragResponse)
	fmt.Println("----------------------------------")

	// Send marshalled json as response with 200 status
	w.Header().Set("Content-Type", "text/plain")
	_, err = w.Write([]byte(ragResponse))
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
	http.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, err := w.Write([]byte(htmlWebPage))
		if err != nil {
			slog.Error(err.Error())
		}
	})
	http.HandleFunc("POST /chat", chatHandler)
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
				slog.Warn(
					"REQUEST",
					"method",
					r.Method,
					"ip",
					r.RemoteAddr,
					"time",
					time.Since(startTime).Round(time.Second),
					"status",
					wr.statusCode,
					"path",
					r.URL.Path,
				)
				mu.Unlock()
				return
			}
			slog.Info(
				"REQUEST",
				"method",
				r.Method,
				"ip",
				r.RemoteAddr,
				"time",
				time.Since(startTime).Round(time.Second),
				"status",
				wr.statusCode,
				"path",
				r.URL.Path,
			)
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
