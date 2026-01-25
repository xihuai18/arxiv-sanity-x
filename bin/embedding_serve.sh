#!/bin/bash
# Read port configuration from config.settings
EMBED_PORT=$(python3 -c "from config import settings; print(settings.embedding.port)")

# Ollama embedding service (CPU-only)
export OLLAMA_HOST="127.0.0.1:${EMBED_PORT}"
# export OLLAMA_LLM_LIBRARY="cpu"
# export CUDA_VISIBLE_DEVICES=""
# export HIP_VISIBLE_DEVICES=""
# export ROCR_VISIBLE_DEVICES=""

exec ollama serve
