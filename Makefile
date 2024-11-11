SHELL=/bin/bash

.PHONY: docker-build
docker-build:
	$(MAKE) build
	docker rmi -f rag-llm
	docker build -t rag-llm .

.PHONY: build
build:
	go mod tidy
	go build -v .

.PHONY: compose-up
compose-up:
	docker compose up -d

.PHONY: compose-down
compose-down:
	docker compose down
