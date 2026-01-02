#!/bin/bash
# 检查 MinerU 是否启用
MINERU_ENABLED=$(python3 -c "from vars import MINERU_ENABLED; print('true' if MINERU_ENABLED else 'false')")

if [ "$MINERU_ENABLED" != "true" ]; then
    echo "MinerU is disabled (ARXIV_SANITY_MINERU_ENABLED=false), exiting."
    exit 0
fi

# 从 vars.py 读取端口配置
MINERU_PORT=$(python3 -c "from vars import MINERU_PORT; print(MINERU_PORT)")
set -x
mineru-vllm-server --port ${MINERU_PORT} --gpu-memory-utilization 0.3 --disable-log-requests --max-num-seqs 10 --enforce-eager
