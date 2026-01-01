#!/bin/bash
# 从 vars.py 读取端口配置
EMBED_PORT=$(python3 -c "from vars import EMBED_PORT; print(EMBED_PORT)")

# Ollama embedding service (CPU-only)
export OLLAMA_HOST="127.0.0.1:${EMBED_PORT}"
export OLLAMA_LLM_LIBRARY="cuda"
export CUDA_VISIBLE_DEVICES="0"

exec ollama serve
