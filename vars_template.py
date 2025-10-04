import os

# database
DATA_DIR = "data"  # !put it on an ssd for speed
# email
from_email = "your_email@mail.com"
smtp_server = "smtp.mail.com"
smtp_port = 465  # 25 for public, 465 for ssl
email_username = "username"
email_passwd = os.environ.get("YOUR_EMAIL_PASSWD", "")
HOST = "Web Host"

LLM_SUMMARY_LANG = "zh"
LLM_BASE_URL = "your_llm_base_url"  # e.g., "https://api.openai.com/v1"
LLM_API_KEY = os.environ.get("YOUR_LLM_API_KEY", "your_llm_api_key")
LLM_NAME = "llm_name_for_paper_summaries"  # e.g., "gpt-3.5-turbo"

# vLLM service port configuration
VLLM_EMBED_PORT = 51000  # Qwen3 Embedding service port
VLLM_MINERU_PORT = 52000  # MinerU VLM service port

# Paper summary related configuration
SUMMARY_MIN_CHINESE_RATIO = 0.25  # Minimum Chinese character ratio threshold for summary cache
SUMMARY_DEFAULT_SEMANTIC_WEIGHT = 0.5  # Default weight for semantic search in hybrid search

# paper_summarizer.py related constants
MAIN_CONTENT_MIN_RATIO = 0.1  # Minimum content ratio after extracting main content

# serve.py related constants
SERVE_PORT = 55555  # Default port for serve.py application
SVM_C = 0.02  # C parameter for SVM classifier
SVM_MAX_ITER = 5000  # Maximum iterations for SVM
SVM_TOL = 1e-3  # SVM tolerance
