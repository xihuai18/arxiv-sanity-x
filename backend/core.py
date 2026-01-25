"""Core business logic module - refactored from legacy.py.

This module contains the main business logic for the arxiv-sanity application.
It has been refactored to use service modules for better organization.
"""

# Also export specific commonly used items
# Re-export everything from legacy for backward compatibility
from .legacy import *  # noqa: F401, F403
from .legacy import (  # Data access; User data; Search functions; Summary functions; Rendering; Request handling; Security
    add_security_headers,
    before_request,
    close_connection,
    default_context,
    enhanced_search_rank,
    generate_paper_summary,
    get_combined_tags,
    get_data_cached,
    get_features_cached,
    get_keys,
    get_metas,
    get_neg_tags,
    get_paper,
    get_papers,
    get_papers_bulk,
    get_pids,
    get_summary_status,
    get_tags,
    hybrid_search_rank,
    paper_exists,
    random_rank,
    render_pid,
    search_rank,
    semantic_search_rank,
    svm_rank,
    time_rank,
)
