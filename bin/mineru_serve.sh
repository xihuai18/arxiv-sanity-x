#!/bin/bash
# Check if MinerU is enabled
MINERU_ENABLED=$(python3 -c "from config import settings; print('true' if settings.mineru.enabled else 'false')")

if [ "$MINERU_ENABLED" != "true" ]; then
    echo "MinerU is disabled (ARXIV_SANITY_MINERU_ENABLED=false), exiting."
    exit 0
fi

# Read port configuration from config.settings
MINERU_PORT=$(python3 -c "from config import settings; print(settings.mineru.port)")
set -x
mineru-vllm-server --port ${MINERU_PORT} --gpu-memory-utilization 0.3 --disable-log-requests --max-num-seqs 10 --enforce-eager
