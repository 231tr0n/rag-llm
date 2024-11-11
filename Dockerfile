FROM alpine:latest

RUN apk add gcompat

WORKDIR /root/rag-llm

COPY rag-llm .

CMD ["./rag-llm"]
