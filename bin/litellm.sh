LITELLM_PORT=$(python3 -c "from config import settings; print(settings.litellm_port)")
LOG_DIR=$(python3 -c "from config import settings; print(settings.log_dir)")
LITELLM_VERBOSE=$(python3 -c "from config import settings; print('1' if settings.llm.litellm_verbose else '0')")

# Keep LiteLLM logs quiet by default, but print detailed context on error.
# Override behavior:
# - ARXIV_SANITY_LLM_LITELLM_VERBOSE=true : stream full logs to stdout
# - LITELLM_LOG_LEVEL / LITELLM_DETAILED_DEBUG : pass through to LiteLLM
# Allow env override for backward compatibility
ARXIV_SANITY_LITELLM_VERBOSE=${ARXIV_SANITY_LITELLM_VERBOSE:-$LITELLM_VERBOSE}

# Get the project root directory (parent of bin/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LLM_CONFIG="$PROJECT_ROOT/config/llm.yml"

if [ "$ARXIV_SANITY_LITELLM_VERBOSE" = "1" ]; then
	export LITELLM_LOG_LEVEL=${LITELLM_LOG_LEVEL:-DEBUG}
	export LITELLM_DETAILED_DEBUG=${LITELLM_DETAILED_DEBUG:-True}
	exec litellm -c "$LLM_CONFIG" --port $LITELLM_PORT
fi

export LITELLM_LOG_LEVEL=${LITELLM_LOG_LEVEL:-WARNING}
export LITELLM_DETAILED_DEBUG=${LITELLM_DETAILED_DEBUG:-False}

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/litellm.log"

# Start LiteLLM in foreground, capture logs quietly (no terminal output).
exec litellm -c "$LLM_CONFIG" --port $LITELLM_PORT >>"$LOG_FILE" 2>&1
