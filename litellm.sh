LITELLM_PORT=$(python3 -c "from vars import LITELLM_PORT; print(LITELLM_PORT)")
litellm -c llm.yml --port $LITELLM_PORT
