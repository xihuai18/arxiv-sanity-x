#!/bin/bash
# 从 vars.py 读取端口配置
MINERU_PORT=$(python3 -c "from vars import MINERU_PORT; print(MINERU_PORT)")
set -x
mineru-vllm-server --port ${MINERU_PORT} --gpu-memory-utilization 0.3 --disable-log-requests --max-num-seqs 10
