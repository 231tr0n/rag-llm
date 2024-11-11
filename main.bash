#!/bin/bash

curl -H 'Content-Type: application/json' \
  -d '{"query": "Is bob an idiot?"}' \
  -X POST \
  https://example.com/posts
