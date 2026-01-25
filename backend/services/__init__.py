"""Services package initialization."""

from ..utils.validation import validate_keyword_name, validate_tag_name
from .api_helpers import api_error, api_success, normalize_name, parse_api_request
from .auth_service import get_user_email, login_user, logout_user, register_user_email
from .background import ensure_background_services_started
from .keyword_service import add_keyword, delete_keyword, rename_keyword
from .readinglist_service import (
    add_to_readinglist,
    get_user_readinglist,
    list_readinglist,
    remove_from_readinglist,
    trigger_summary_async,
    update_summary_status,
    update_summary_status_db,
)
from .render_service import (
    THUMB_CACHE,
    build_paper_text_fields,
    get_thumb_url,
    render_pid,
)
from .search_service import (
    MAX_RESULTS,
    QUERY_EMBED_CACHE,
    RET_NUM,
    SEARCH_RANK_CACHE,
    SVM_RANK_CACHE,
    apply_limit,
    extract_arxiv_ids,
    filter_by_time,
    normalize_text,
    normalize_text_loose,
    parse_search_query,
    random_rank,
    time_rank,
)
from .summary_service import (
    TLDR_CACHE,
    clear_model_summary,
    clear_paper_cache,
    extract_tldr_from_summary,
    get_summary_cache_stats,
    get_summary_status,
)
from .tag_service import (
    add_paper_to_tag,
    create_combined_tag,
    create_empty_tag,
    delete_combined_tag,
    delete_tag,
    get_tag_members,
    remove_paper_from_tag,
    rename_combined_tag,
    rename_tag,
    resolve_paper_titles,
    set_tag_feedback,
)
from .user_service import (
    before_request,
    build_user_combined_tag_list,
    build_user_key_list,
    build_user_tag_list,
    close_connection,
    get_combined_tags,
    get_keys,
    get_neg_tags,
    get_tags,
)

__all__ = [
    # API helpers
    "api_error",
    "api_success",
    "normalize_name",
    "parse_api_request",
    # Auth services
    "login_user",
    "logout_user",
    "register_user_email",
    "get_user_email",
    # Background services
    "ensure_background_services_started",
    # Keyword services
    "add_keyword",
    "delete_keyword",
    "rename_keyword",
    # Reading list services
    "get_user_readinglist",
    "add_to_readinglist",
    "remove_from_readinglist",
    "list_readinglist",
    "update_summary_status",
    "update_summary_status_db",
    "trigger_summary_async",
    # Tag services
    "create_empty_tag",
    "add_paper_to_tag",
    "remove_paper_from_tag",
    "delete_tag",
    "rename_tag",
    "create_combined_tag",
    "delete_combined_tag",
    "rename_combined_tag",
    "set_tag_feedback",
    "get_tag_members",
    "resolve_paper_titles",
    # User services
    "before_request",
    "build_user_combined_tag_list",
    "build_user_key_list",
    "build_user_tag_list",
    "close_connection",
    "get_combined_tags",
    "get_keys",
    "get_neg_tags",
    "get_tags",
    "validate_keyword_name",
    "validate_tag_name",
    # Search services
    "normalize_text",
    "normalize_text_loose",
    "extract_arxiv_ids",
    "parse_search_query",
    "apply_limit",
    "random_rank",
    "time_rank",
    "filter_by_time",
    "SVM_RANK_CACHE",
    "SEARCH_RANK_CACHE",
    "QUERY_EMBED_CACHE",
    "RET_NUM",
    "MAX_RESULTS",
    # Summary services
    "get_summary_status",
    "extract_tldr_from_summary",
    "get_summary_cache_stats",
    "clear_model_summary",
    "clear_paper_cache",
    "TLDR_CACHE",
    # Render services
    "get_thumb_url",
    "render_pid",
    "build_paper_text_fields",
    "THUMB_CACHE",
]
