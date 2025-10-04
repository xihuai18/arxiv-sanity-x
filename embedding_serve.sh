#!/bin/bash
# 从 vars.py 读取端口配置
EMBED_PORT=$(python3 -c "from vars import VLLM_EMBED_PORT; print(VLLM_EMBED_PORT)")
vllm serve ./qwen3-embed-0.6B --task embed --hf-overrides '{"is_matryoshka": true}' --gpu-memory-utilization 0.1 --port ${EMBED_PORT} --served-model-name qwen3-embed-0.6B --disable-log-requests --max-num-seqs 10 --max-model-len 4096
