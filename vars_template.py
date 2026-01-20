import os

# database
DATA_DIR = "data"  # !put it on an ssd for speed
SUMMARY_DIR = os.path.join(DATA_DIR, "summary")

# email
from_email = "your_email@mail.com"
smtp_server = "smtp.mail.com"
smtp_port = 465  # 25 for public, 465 for ssl
email_username = "username"
email_passwd = os.environ.get("YOUR_EMAIL_PASSWD", "")
HOST = "Web Host"

# LLM API Configuration
# Recommended: Use OpenRouter with free models
# Example OpenRouter config:
#   LLM_BASE_URL = "https://openrouter.ai/api/v1"
#   LLM_API_KEY = "sk-or-v1-..."
#   LLM_NAME = "deepseek/deepseek-chat-v3.1:free"
LLM_SUMMARY_LANG = "zh"
LLM_BASE_URL = "your_llm_base_url"  # e.g., "https://openrouter.ai/api/v1"
LLM_API_KEY = os.environ.get("YOUR_LLM_API_KEY", "your_llm_api_key")
LLM_NAME = "llm_name_for_paper_summaries"  # e.g., "deepseek/deepseek-chat-v3.1:free"

# LiteLLM service port (optional, for multi-model gateway)
LITELLM_PORT = 53000  # LiteLLM service port

# Service port configuration
EMBED_PORT = 51000  # Ollama embedding service port
MINERU_PORT = 52000  # MinerU VLM service port (vLLM)

# Backward-compatible aliases (deprecated)
VLLM_EMBED_PORT = EMBED_PORT
VLLM_MINERU_PORT = MINERU_PORT

# Embedding API Configuration
# Set EMBED_USE_LLM_API=True to use LLM_BASE_URL for embeddings (OpenAI-compatible API)
# Set EMBED_USE_LLM_API=False to use local Ollama service at EMBED_PORT
EMBED_USE_LLM_API = os.environ.get("ARXIV_SANITY_EMBED_USE_LLM_API", "true").lower() in ("true", "1", "yes")
EMBED_MODEL_NAME = "qwen3-embedding:0.6b"
# Optional: Override embedding API base URL (defaults to LLM_BASE_URL when EMBED_USE_LLM_API=True)
EMBED_API_BASE = os.environ.get("ARXIV_SANITY_EMBED_API_BASE", "")  # Empty means use LLM_BASE_URL
EMBED_API_KEY = os.environ.get("ARXIV_SANITY_EMBED_API_KEY", "")  # Empty means use LLM_API_KEY

# Paper summary related configuration
SUMMARY_MIN_CHINESE_RATIO = 0.25  # Minimum Chinese character ratio threshold for summary cache
SUMMARY_DEFAULT_SEMANTIC_WEIGHT = 0.5  # Default weight for semantic search in hybrid search
SUMMARY_MARKDOWN_SOURCE = os.environ.get("ARXIV_SANITY_SUMMARY_SOURCE", "html")  # html (default) or mineru
SUMMARY_HTML_SOURCES = os.environ.get("ARXIV_SANITY_HTML_SOURCES", "ar5iv,arxiv")  # HTML source order

# MinerU configuration
MINERU_ENABLED = os.environ.get("ARXIV_SANITY_MINERU_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)  # Enable/disable MinerU PDF parsing
MINERU_BACKEND = os.environ.get("ARXIV_SANITY_MINERU_BACKEND", "api")  # pipeline, vlm-http-client, or api (default)
MINERU_DEVICE = os.environ.get("ARXIV_SANITY_MINERU_DEVICE", "cuda")  # cuda (default) or cpu (pipeline backend only)
MINERU_MAX_WORKERS = int(
    os.environ.get("ARXIV_SANITY_MINERU_MAX_WORKERS", "2")
)  # Maximum concurrent mineru processes (pipeline backend only)
MINERU_MAX_VRAM = int(
    os.environ.get("ARXIV_SANITY_MINERU_MAX_VRAM", "3")
)  # Max VRAM per mineru process in GB (pipeline+cuda only)
MINERU_API_KEY = os.environ.get("MINERU_API_KEY", "")  # MinerU API key (for api backend)
MINERU_API_POLL_INTERVAL = int(
    os.environ.get("MINERU_API_POLL_INTERVAL", "5")
)  # Polling interval in seconds for API backend
MINERU_API_TIMEOUT = int(os.environ.get("MINERU_API_TIMEOUT", "600"))  # API task timeout in seconds

# paper_summarizer.py related constants
MAIN_CONTENT_MIN_RATIO = 0.1  # Minimum content ratio after extracting main content

# serve.py related constants
SERVE_PORT = 55555  # Default port for serve.py application
SVM_C = 0.02  # C parameter for SVM classifier
SVM_MAX_ITER = 5000  # Maximum iterations for SVM
SVM_TOL = 1e-3  # SVM tolerance
SVM_NEG_WEIGHT = 5.0  # Weight for explicit negative feedback samples
