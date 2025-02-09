# rag-llm
A RAG model with ollama and weviate.

## Querying
```bash
curl -H 'Content-Type: application/json' \
  -d '{"history": [{"role": "user", "text": "Is bob an idiot?"}]}'
  -X POST \
  http://localhost:8000/query
```
## Adding documents
```bash
curl -H 'Content-Type: application/json' \
  -d '{"document": "Bob is an idiot."}' \
  -X POST \
  http://localhost:8000/add
```
