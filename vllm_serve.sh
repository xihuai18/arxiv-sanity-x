vllm serve ./qwen3-embed-0.6B --task embed --hf-overrides '{"is_matryoshka": true}' --gpu-memory-utilization 0.2 --port 51000 --served-model-name qwen3-embed-0.6B --disable-log-requests
