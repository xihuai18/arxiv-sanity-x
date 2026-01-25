"""
Compose and send recommendation emails to arxiv-sanity-X users!

I run this script in a cron job to send out emails to the users with their
recommendations. There's a bit of copy paste code here but I expect that
the recommendations may become more complex in the future, so this is ok for now.

You'll notice that the file sendgrid_api_key.txt is not in the repo, you'd have
to manually register with sendgrid yourself, get an API key and put it in the file.
"""

import argparse
import html
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Dict, List, Set

import requests
from loguru import logger

# Ensure repository root is importable when executing this file directly.
# e.g. `python tools/send_emails.py` would otherwise miss sibling packages like `aslite/`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Repository layer for cleaner data access
from aslite.repositories import (
    CombinedTagRepository,
    KeywordRepository,
    PaperRepository,
    TagRepository,
    UserRepository,
)
from config import settings

HOST = settings.host
SERVE_PORT = settings.serve_port

# Thread configuration from centralized settings
MAX_NUM_THREADS = settings.reco.max_threads


def _resolve_num_threads(requested: int) -> int:
    """Return a safe thread count.

    requested <= 0 means "auto".
    """

    if requested <= 0:
        return min(cpu_count(), MAX_NUM_THREADS)
    return min(requested, MAX_NUM_THREADS)


def _configure_thread_env_vars(n: int) -> None:
    # Must run early (before heavy numeric libraries initialize thread pools).
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


@dataclass(frozen=True)
class RecoHyperparams:
    api_limit: int
    model_C: float


USE_INTEL_EXT = False

# API call configuration from centralized settings
API_BASE_URL = settings.reco.api_base_url
API_TIMEOUT = settings.reco.api_timeout

# Thread configuration from centralized settings
num_threads = _resolve_num_threads(settings.reco.num_threads)

# Recommendation hyperparams from centralized settings
RECO_HPARAMS = RecoHyperparams(
    api_limit=settings.reco.api_limit,
    model_C=settings.reco.model_c,
)

# Web name for email template
WEB = settings.reco.web_name

# -----------------------------------------------------------------------------
# the html template for the email


template = """
<!DOCTYPE HTML>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arxiv Sanity X Recommendations</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f7fa;
            margin: 0;
            padding: 20px;
        }

        .container {
            max-width: 680px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            overflow: hidden;
        }

        .header {
            background-color: #b91c1c;
            color: white;
            padding: 30px 20px;
            text-align: center;
        }

        .header h1 {
            margin: 0;
            font-size: 26px;
            font-weight: 700;
        }

        .header .subtitle {
            margin-top: 8px;
            font-size: 14px;
            opacity: 0.9;
        }

        .content {
            padding: 24px;
        }

        .greeting {
            font-size: 16px;
            margin-bottom: 24px;
            color: #1f2937;
        }

        .section {
            margin-bottom: 30px;
        }

        .section-title {
            font-size: 18px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e5e7eb;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .section-stats {
            background-color: #f9fafb;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #4b5563;
            border: 1px solid #e5e7eb;
        }

        .paper-item {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin-bottom: 24px;
            padding: 20px;
            background-color: #ffffff;
        }

        .paper-title {
            font-size: 18px;
            font-weight: 700;
            color: #111827;
            margin: 0 0 10px 0;
            line-height: 1.4;
        }

        .paper-title a {
            color: #111827;
            text-decoration: none;
        }

        .paper-title a:hover {
            color: #b91c1c;
            text-decoration: underline;
        }

        .paper-meta {
            margin-bottom: 12px;
        }

        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 8px;
            margin-bottom: 4px;
        }

        .badge-score {
            background-color: #fee2e2;
            color: #991b1b;
        }

        .badge-source {
            background-color: #e0f2fe;
            color: #075985;
        }

        .badge-keyword {
            background-color: #fef3c7;
            color: #92400e;
        }

        .badge-ctag {
            background-color: #d1fae5;
            color: #065f46;
        }

        .badge-date {
            background-color: #f3f4f6;
            color: #4b5563;
        }

        .paper-authors {
            font-size: 13px;
            color: #4b5563;
            margin-bottom: 16px;
            line-height: 1.5;
        }

        .paper-actions {
            margin-bottom: 16px;
        }

        .btn {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            text-decoration: none;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        .btn-sanity {
            background-color: #b91c1c;
            color: white;
        }
        .btn-sanity:hover {
            background-color: #991b1b;
        }

        .btn-summary {
            background-color: #d97706;
            color: white;
        }
        .btn-summary:hover {
            background-color: #b45309;
        }

        .btn-arxiv {
            background-color: #be123c;
            color: white;
        }
        .btn-arxiv:hover {
            background-color: #9f1239;
        }

        .btn-alphaxiv {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        .btn-alphaxiv:hover {
            background-color: #fecaca;
        }

        .btn-cool {
            background-color: #059669;
            color: white;
        }
        .btn-cool:hover {
            background-color: #047857;
        }

        .paper-tldr {
            background-color: #fff1f2;
            border-left: 3px solid #be123c;
            padding: 12px 16px;
            margin-bottom: 12px;
            border-radius: 4px;
        }

        .paper-tldr .tldr-label {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            color: #be123c;
            margin-bottom: 4px;
        }

        .paper-tldr .tldr-text {
            font-size: 13px;
            color: #881337;
            line-height: 1.5;
            margin: 0;
        }

        .paper-summary {
            font-size: 13px;
            color: #6b7280;
            line-height: 1.6;
        }

        .footer {
            background-color: #f9fafb;
            padding: 30px;
            text-align: center;
            border-top: 1px solid #e5e7eb;
        }

        .footer p {
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #6b7280;
        }

        .footer a {
            color: #b91c1c;
            text-decoration: none;
            font-weight: 500;
        }

        .footer .brand {
            font-weight: 700;
            color: #1f2937;
            font-size: 18px;
            margin-top: 15px;
        }

        @media (max-width: 600px) {
            body { padding: 10px; }
            .header, .content, .footer { padding: 20px; }
            .paper-item { padding: 16px; }
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Arxiv Sanity X</h1>
            <div class="subtitle">Your Daily Research Companion</div>
        </div>

        <div class="content">
            <div class="greeting">
                👋 Hi <strong>__USER__</strong>! Here are your personalized <a href="__HOST__" style="color: #667eea; text-decoration: none; font-weight: 600;">__WEB__</a> recommendations.
            </div>

            __SECTION_TAG__

            __SECTION_CTAG__

            __SECTION_KEYWORD__
        </div>

        <div class="footer">
            <p>To stop these emails, remove your email in your <a href="__HOST__/profile">account settings</a>.</p>
            <p>Your account: <strong>__ACCOUNT__</strong></p>
            <div class="brand">🎓 __WEB__</div>
        </div>
    </div>
</body>
</html>
"""


def _h(value) -> str:
    return html.escape(str(value), quote=True)


def _crop_summary(text: str, limit: int = 500) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _abbr_author_middle(name: str) -> str:
    """Abbreviate an author name by keeping first and last names, abbreviating middle names.

    Examples:
      - "Huilin Deng" -> "Huilin Deng"
      - "John Ronald Reuel Tolkien" -> "John R. R. Tolkien"
      - "Jean-Pierre Vincent" -> "Jean-Pierre Vincent" (kept)
    """

    if not name:
        return ""

    parts = [p for p in str(name).strip().split() if p]
    if len(parts) <= 2:
        return " ".join(parts)

    # Keep first and last name, abbreviate middle names
    first = parts[0]
    last = parts[-1]
    middle_abbr = " ".join(f"{m[0]}." for m in parts[1:-1] if m)

    return f"{first} {middle_abbr} {last}"


from backend.utils.summary_utils import markdown_to_email_html
from backend.utils.summary_utils import (
    read_tldr_from_summary_file as _extract_tldr_from_summary,
)


def _render_paper_html(paper: dict, pid: str, score: float, source_label: str, source_class: str = "badge-source"):
    title = _h(paper.get("title", ""))

    # Truncate authors - keep first 15 and last 5, omit middle ones if > 20
    author_list = paper.get("authors", [])
    if len(author_list) > 20:
        first_authors = ", ".join(_abbr_author_middle(a.get("name", "")) for a in author_list[:15])
        last_authors = ", ".join(_abbr_author_middle(a.get("name", "")) for a in author_list[-5:])
        omitted_count = len(author_list) - 20
        authors_str = f"{first_authors}, ... ({omitted_count} authors omitted) ..., {last_authors}"
    else:
        authors_str = ", ".join(_abbr_author_middle(a.get("name", "")) for a in author_list)
    authors = _h(authors_str)

    full_summary = _h(paper.get("summary", ""))
    time_str = _h(paper.get("_time_str", ""))
    tldr_raw = _extract_tldr_from_summary(pid)
    tldr = markdown_to_email_html(tldr_raw) if tldr_raw else ""

    url = _h(f"{HOST}/?rank=pid&pid={pid}")
    summary_url = _h(f"{HOST}/summary?pid={pid}")
    arxiv_url = _h(f"https://arxiv.org/abs/{pid}")
    alphaxiv_url = _h(f"https://www.alphaxiv.org/overview/{pid}")
    cool_url = _h(f"https://papers.cool/arxiv/{pid}")

    # Render TL;DR section if available (tldr is already HTML from markdown_to_email_html)
    content_html = ""
    if tldr:
        content_html = f"""
        <div class="paper-tldr">
            <div class="tldr-label">💡 TL;DR</div>
            <div class="tldr-text">{tldr}</div>
        </div>
        """
    else:
        content_html = f'<div class="paper-summary">{full_summary}</div>'

    return f"""
    <div class="paper-item">
        <div class="paper-title"><a href="{url}">{title}</a></div>

        <div class="paper-meta">
            <span class="badge badge-score">⭐ {float(score):.2f}</span>
            <span class="badge {source_class}">🏷️ {_h(source_label)}</span>
            <span class="badge badge-date">📅 {time_str}</span>
        </div>

        <div class="paper-authors">{authors}</div>

        <div class="paper-actions">
            <a href="{url}" class="btn btn-sanity">🔍 Sanity</a>
            <a href="{summary_url}" class="btn btn-summary">📝 Summary</a>
            <a href="{arxiv_url}" class="btn btn-arxiv">📚 arXiv</a>
            <a href="{alphaxiv_url}" class="btn btn-alphaxiv">αXiv</a>
            <a href="{cool_url}" class="btn btn-cool">Cool</a>
        </div>

        {content_html}
    </div>
    """


def _api_worker_count(n_tasks: int) -> int:
    # Prefer CLI, fallback to settings.
    try:
        cli_args = globals().get("args")
        max_workers = int(getattr(cli_args, "api_workers", None) or settings.email.api_workers)
    except Exception:
        max_workers = 8
    return max(1, min(max_workers, n_tasks))


def _post_recommendation(url: str, payload: dict, label: str, key: str):
    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
    except requests.RequestException as e:
        logger.error(f"API request exception for {label} {key}: {e}")
        return None

    if response.status_code != 200:
        logger.error(f"API request failed for {label} {key}: {response.status_code}")
        return None

    try:
        result = response.json()
    except Exception as e:
        logger.error(f"API response parse failed for {label} {key}: {e}")
        return None

    if not result.get("success"):
        logger.error(f"API error for {label} {key}: {result.get('error')}")
        return None

    return key, result.get("pids", []), result.get("scores", [])


def calculate_ctag_recommendation(
    ctags: List[str],
    tags: dict,
    user: str,  # Add user parameter
    time_delta: int = 3,
):
    """Use API to call joint tag recommendations"""
    all_pids, all_scores = {}, {}

    tasks = []
    for ctag in ctags:
        _tags = list(map(str.strip, ctag.split(",")))
        valid_tags = [tag for tag in _tags if tag in tags and len(tags[tag]) > 0]
        if not valid_tags:
            continue
        payload = {
            "tags": valid_tags,
            "user": user,
            "logic": "and",
            "time_delta": time_delta,
            "limit": RECO_HPARAMS.api_limit,
            "C": RECO_HPARAMS.model_C,
        }
        tasks.append((ctag, payload))

    if not tasks:
        return all_pids, all_scores

    max_workers = _api_worker_count(len(tasks))
    if max_workers <= 1:
        for ctag, payload in tasks:
            result = _post_recommendation(f"{API_BASE_URL}/api/tags_search", payload, "ctag", ctag)
            if result:
                key, pids, scores = result
                all_pids[key] = pids
                all_scores[key] = scores
        return all_pids, all_scores

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_post_recommendation, f"{API_BASE_URL}/api/tags_search", payload, "ctag", ctag): ctag
            for ctag, payload in tasks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"API request exception for ctag {futures[future]}: {e}")
                continue
            if result:
                key, pids, scores = result
                all_pids[key] = pids
                all_scores[key] = scores

    return all_pids, all_scores


def calculate_recommendation(
    tags: dict,
    user: str,  # Add user parameter
    time_delta: int = 3,  # how recent papers are we recommending? in days
):
    """Use API to call single tag recommendations"""
    all_pids, all_scores = {}, {}

    tasks = []
    for tag, tpids in tags.items():
        if len(tpids) == 0:
            continue
        payload = {
            "tag_name": tag,
            "user": user,
            "time_delta": time_delta,
            "limit": RECO_HPARAMS.api_limit,
            "C": RECO_HPARAMS.model_C,
        }
        tasks.append((tag, payload))

    if not tasks:
        return all_pids, all_scores

    max_workers = _api_worker_count(len(tasks))
    if max_workers <= 1:
        for tag, payload in tasks:
            result = _post_recommendation(f"{API_BASE_URL}/api/tag_search", payload, "tag", tag)
            if result:
                key, pids, scores = result
                all_pids[key] = pids
                all_scores[key] = scores
        return all_pids, all_scores

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_post_recommendation, f"{API_BASE_URL}/api/tag_search", payload, "tag", tag): tag
            for tag, payload in tasks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"API request exception for tag {futures[future]}: {e}")
                continue
            if result:
                key, pids, scores = result
                all_pids[key] = pids
                all_scores[key] = scores

    return all_pids, all_scores


# -----------------------------------------------------------------------------
# Keywords recommendation
# -----------------------------------------------------------------------------
def search_keywords_recommendations(user: str, keywords: Dict[str, Set[str]], time_delta: int = 3):
    """Use API to call keyword search"""
    all_pids, all_scores = {}, {}

    tasks = []
    for keyword, pids in keywords.items():
        payload = {"keyword": keyword, "time_delta": time_delta, "limit": RECO_HPARAMS.api_limit}
        tasks.append((keyword, payload, pids))

    if not tasks:
        return all_pids, all_scores

    def _post_keyword(keyword, payload, existing_pids):
        result = _post_recommendation(f"{API_BASE_URL}/api/keyword_search", payload, "keyword", keyword)
        if not result:
            return None
        key, rec_pids, rec_scores = result
        keep = [i for i, pid in enumerate(rec_pids) if pid not in existing_pids]
        rec_pids = [rec_pids[i] for i in keep]
        rec_scores = [rec_scores[i] for i in keep]
        return key, rec_pids, rec_scores

    max_workers = _api_worker_count(len(tasks))
    if max_workers <= 1:
        for keyword, payload, existing_pids in tasks:
            result = _post_keyword(keyword, payload, existing_pids)
            if result:
                key, rec_pids, rec_scores = result
                all_pids[key] = rec_pids
                all_scores[key] = rec_scores
        return all_pids, all_scores

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_post_keyword, keyword, payload, existing_pids): keyword
            for keyword, payload, existing_pids in tasks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"API request exception for keyword {futures[future]}: {e}")
                continue
            if result:
                key, rec_pids, rec_scores = result
                all_pids[key] = rec_pids
                all_scores[key] = rec_scores

    return all_pids, all_scores


# -----------------------------------------------------------------------------


# TODO: Link to summary interface
def render_recommendations(
    user,
    tags,
    tag_pids,
    tag_scores,
    ctags,
    ctag_pids,
    ctag_scores,
    keywords,
    kpids,
    kscores,
):
    out = template
    safe_user = _h(user)
    safe_host = _h(HOST)
    safe_web = _h(WEB)
    out = out.replace("__USER__", safe_user)
    out = out.replace("__HOST__", safe_host)
    out = out.replace("__WEB__", safe_web)

    # Collect all recommended papers, deduplicate by priority: tag > ctag > keyword
    tag_papers = set()  # Papers recommended by tag
    ctag_papers = set()  # Papers recommended by ctag
    keyword_papers = set()  # Papers recommended by keyword

    # First collect tag recommended papers
    if sum(len(tag_pids[tag]) for tag in tag_pids) > 0:
        for tag in tag_pids:
            tag_papers.update(tag_pids[tag])

    # Collect ctag recommended papers, exclude those already in tag
    if sum(len(ctag_pids[ctag]) for ctag in ctag_pids) > 0:
        for ctag in ctag_pids:
            ctag_papers.update(ctag_pids[ctag])
        ctag_papers -= tag_papers  # Deduplicate: exclude those already in tag

    # Collect keyword recommended papers, exclude those already in tag and ctag
    if sum(len(kpids[keyword]) for keyword in kpids) > 0:
        for keyword in kpids:
            keyword_papers.update(kpids[keyword])
        keyword_papers -= tag_papers  # Deduplicate: exclude those already in tag
        keyword_papers -= ctag_papers  # Deduplicate: exclude those already in ctag

    # Apply deduplicated results, modify original recommendation data
    # Process tag recommendations
    filtered_tag_pids = {}
    filtered_tag_scores = {}
    for tag in tag_pids:
        filtered_pids = []
        filtered_scores = []
        for pid, score in zip(tag_pids[tag], tag_scores[tag]):
            if pid in tag_papers:  # Only keep papers in tag_papers
                filtered_pids.append(pid)
                filtered_scores.append(score)
        filtered_tag_pids[tag] = filtered_pids
        filtered_tag_scores[tag] = filtered_scores

    # Process ctag recommendations
    filtered_ctag_pids = {}
    filtered_ctag_scores = {}
    for ctag in ctag_pids:
        filtered_pids = []
        filtered_scores = []
        for pid, score in zip(ctag_pids[ctag], ctag_scores[ctag]):
            if pid in ctag_papers:  # Only keep papers in ctag_papers
                filtered_pids.append(pid)
                filtered_scores.append(score)
        filtered_ctag_pids[ctag] = filtered_pids
        filtered_ctag_scores[ctag] = filtered_scores

    # Process keyword recommendations
    filtered_kpids = {}
    filtered_kscores = {}
    for keyword in kpids:
        filtered_pids = []
        filtered_scores = []
        for pid, score in zip(kpids[keyword], kscores[keyword]):
            if pid in keyword_papers:  # Only keep papers in keyword_papers
                filtered_pids.append(pid)
                filtered_scores.append(score)
        filtered_kpids[keyword] = filtered_pids
        filtered_kscores[keyword] = filtered_scores

    # render the paper recommendations into the html template
    if sum(len(filtered_tag_pids[tag]) for tag in filtered_tag_pids) > 0:
        # first we are going to merge all of the papers / scores together using a MAX
        max_score = {}
        max_source_tag = {}
        for tag in filtered_tag_pids:
            for pid, score in zip(filtered_tag_pids[tag], filtered_tag_scores[tag]):
                cur = max_score.get(pid, -99999)
                # Prefer higher score; on ties pick lexicographically smaller tag for determinism
                if score > cur or (score == cur and tag < max_source_tag.get(pid, "~")):
                    max_score[pid] = score
                    max_source_tag[pid] = tag

        # now we have a dict of pid -> max score. sort by score
        max_score_list = sorted(max_score.items(), key=lambda x: x[1], reverse=True)
        if max_score_list:
            pids, scores = zip(*max_score_list)
        else:
            pids, scores = [], []

        # Batch fetch papers using Repository layer (optimization: avoid N+1 queries)
        pids_to_fetch = list(pids[: args.num_recommendations])
        papers_cache = PaperRepository.get_by_ids(pids_to_fetch)

        # now render the html for each individual recommendation
        parts = []
        for score, pid in zip(scores, pids):
            if len(parts) >= args.num_recommendations:
                break
            p = papers_cache.get(pid)
            if p is None:
                logger.warning(f"Missing paper in db for tag recommendation: {pid}")
                continue
            # Show only the max-score tag (truncate long tag for email layout)
            t = max_source_tag.get(pid, "")
            source_label = (t[:20] + "…") if len(t) > 20 else t
            parts.append(_render_paper_html(p, pid, score, source_label))

        # render the recommendations
        final = "".join(parts)

        # render the stats
        num_papers_tagged = len(set().union(*tags.values()))
        tags_str = ", ".join([f'"{_h(t)}" ({len(tags[t])})' for t in tags.keys()])
        n = len(parts)
        stats = f"📊 Analyzed <strong>{num_papers_tagged}</strong> papers across your <strong>{len(tags)}</strong> tags ({tags_str}). \
                Found <strong>{len(pids)}</strong> new papers from the last <strong>{args.time_delta}</strong> days. \
                Showing top <strong>{n}</strong> recommendations."

        # render the complete section
        section_html = f"""
            <div class="section">
                <div class="section-title" style="border-color: #1e40af;">🏷️ Tag-based Recommendations</div>
                <div class="section-stats">{stats}</div>
                <div>{final}</div>
            </div>"""
        out = out.replace("__SECTION_TAG__", section_html)
    else:
        out = out.replace("__SECTION_TAG__", "")

    if sum(len(filtered_ctag_pids[ctag]) for ctag in filtered_ctag_pids) > 0:
        # first we are going to merge all of the papers / scores together using a MAX
        max_score = {}
        max_source_ctag = {}
        for ctag in filtered_ctag_pids:
            for pid, score in zip(filtered_ctag_pids[ctag], filtered_ctag_scores[ctag]):
                max_score[pid] = max(max_score.get(pid, -99999), score)  # lol
                if max_score[pid] == score:
                    max_source_ctag[pid] = ctag

        # now we have a dict of pid -> max score. sort by score
        max_score_list = sorted(max_score.items(), key=lambda x: x[1], reverse=True)
        if max_score_list:
            pids, scores = zip(*max_score_list)
        else:
            pids, scores = [], []

        # Batch fetch papers using Repository layer (optimization: avoid N+1 queries)
        pids_to_fetch = list(pids[: args.num_recommendations])
        papers_cache = PaperRepository.get_by_ids(pids_to_fetch)

        # now render the html for each individual recommendation
        parts = []
        for score, pid in zip(scores, pids):
            if len(parts) >= args.num_recommendations:
                break
            p = papers_cache.get(pid)
            if p is None:
                logger.warning(f"Missing paper in db for ctag recommendation: {pid}")
                continue
            parts.append(_render_paper_html(p, pid, score, max_source_ctag.get(pid, ""), source_class="badge-ctag"))

        # render the recommendations
        final = "".join(parts)

        # render the stats
        # Calculate total number of papers involved in joint tags
        ctag_papers = set()
        for ctag in ctags:
            _tags = list(map(str.strip, ctag.split(",")))
            for tag in _tags:
                if tag in tags:
                    ctag_papers.update(tags[tag])
        num_papers_ctagged = len(ctag_papers)

        ctags_str = ", ".join([f'"{_h(t)}"' for t in ctags])
        n = len(parts)
        stats = f"🔗 Analyzed <strong>{num_papers_ctagged}</strong> papers across your <strong>{len(ctags)}</strong> combined tags ({ctags_str}). \
                Found <strong>{len(pids)}</strong> new papers from the last <strong>{args.time_delta}</strong> days. \
                Showing top <strong>{n}</strong> recommendations."

        # render the complete section
        section_html = f"""
            <div class="section">
                <div class="section-title" style="border-color: #10b981;">🔗 Combined Tag Recommendations</div>
                <div class="section-stats">{stats}</div>
                <div>{final}</div>
            </div>"""
        out = out.replace("__SECTION_CTAG__", section_html)
    else:
        out = out.replace("__SECTION_CTAG__", "")

    if sum(len(filtered_kpids[keyword]) for keyword in filtered_kpids) > 0:
        # first we are going to merge all of the papers / scores together using a MAX
        max_score = {}
        max_source_keyword = {}
        for keyword in filtered_kpids:
            for pid, score in zip(filtered_kpids[keyword], filtered_kscores[keyword]):
                max_score[pid] = max(max_score.get(pid, -99999), score)  # lol
                if max_score[pid] == score:
                    max_source_keyword[pid] = keyword

        # now we have a dict of pid -> max score. sort by score
        max_score_list = sorted(max_score.items(), key=lambda x: x[1], reverse=True)
        if max_score_list:
            pids, scores = zip(*max_score_list)
        else:
            pids, scores = [], []

        # Batch fetch papers using Repository layer (optimization: avoid N+1 queries)
        pids_to_fetch = list(pids[: args.num_recommendations])
        papers_cache = PaperRepository.get_by_ids(pids_to_fetch)

        # now render the html for each individual recommendation
        parts = []
        for score, pid in zip(scores, pids):
            if len(parts) >= args.num_recommendations:
                break
            p = papers_cache.get(pid)
            if p is None:
                logger.warning(f"Missing paper in db for keyword recommendation: {pid}")
                continue
            parts.append(
                _render_paper_html(p, pid, score, max_source_keyword.get(pid, ""), source_class="badge-keyword")
            )

        # render the recommendations
        final = "".join(parts)

        # render the stats
        keywords_str = ", ".join([f'"{_h(k)}"' for k, pids in keywords.items()])
        n = len(parts)
        stats = f"🔍 Searched your <strong>{len(keywords)}</strong> keywords ({keywords_str}). \
                Found <strong>{len(pids)}</strong> matching papers from the last <strong>{args.time_delta}</strong> days. \
                Showing top <strong>{n}</strong> recommendations."

        # render the complete section
        section_html = f"""
            <div class="section">
                <div class="section-title" style="border-color: #d97706;">🔑 Keyword-based Recommendations</div>
                <div class="section-stats">{stats}</div>
                <div>{final}</div>
            </div>"""
        out = out.replace("__SECTION_KEYWORD__", section_html)
    else:
        out = out.replace("__SECTION_KEYWORD__", "")

    # render the account
    out = out.replace("__ACCOUNT__", safe_user)

    return out


# -----------------------------------------------------------------------------


def send_email(to_email, html):
    if args.dry_run:
        logger.debug(html)
    else:
        import smtplib
        from email.header import Header
        from email.mime.text import MIMEText

        from config import settings

        # construct the email
        message = MIMEText(html, "html", "utf-8")
        subject = tnow_str + " Arxiv Sanity X recommendations"
        message["Subject"] = Header(subject, "utf-8")

        try:
            mail = smtplib.SMTP_SSL(settings.email.smtp_server, settings.email.smtp_port)
            mail.login(settings.email.username, settings.email.password)
            mail.sendmail(settings.email.from_email, [to_email], message.as_string())
        except smtplib.SMTPException as e:
            logger.error(e)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sends emails with recommendations")
    parser.add_argument(
        "-n",
        "--num-recommendations",
        type=int,
        default=20,
        help="number of recommendations to send per person",
    )
    parser.add_argument(
        "-t",
        "--time-delta",
        type=float,
        default=2,
        help="how recent papers to recommended, in days",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="dry run mode (no emails sent)",
    )
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        default="",
        help="restrict recommendations only to a single given user (used for debugging)",
    )
    parser.add_argument(
        "-m",
        "--min-papers",
        type=int,
        default=1,
        help="user must have at least this many papers for us to send recommendations",
    )

    # --- Hyperparameter/config knobs (CLI + env) ---
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=API_BASE_URL,
        help="Recommendation API base url (env: ARXIV_SANITY_RECO_API_BASE_URL)",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=API_TIMEOUT,
        help="Recommendation API timeout seconds (env: ARXIV_SANITY_RECO_API_TIMEOUT)",
    )
    parser.add_argument(
        "--api-workers",
        type=int,
        default=settings.email.api_workers,
        help="Max concurrent API requests per user (env: ARXIV_SANITY_EMAIL_API_WORKERS)",
    )
    parser.add_argument(
        "--api-limit",
        type=int,
        default=RECO_HPARAMS.api_limit,
        help="Per-query candidate limit requested from API (env: ARXIV_SANITY_RECO_API_LIMIT)",
    )
    parser.add_argument(
        "--reco-c",
        type=float,
        default=RECO_HPARAMS.model_C,
        help="Model hyperparameter C passed to tag recommenders (env: ARXIV_SANITY_RECO_MODEL_C)",
    )
    parser.add_argument(
        "--reco-C",
        dest="reco_c",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=settings.reco.num_threads,
        help=f"Thread count for BLAS/NumExpr (0=auto, capped at {MAX_NUM_THREADS}; env: ARXIV_SANITY_RECO_NUM_THREADS)",
    )
    parser.add_argument(
        "--web-name",
        type=str,
        default=WEB,
        help="Brand name shown in email template (env: ARXIV_SANITY_RECO_WEB_NAME)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    global API_BASE_URL, API_TIMEOUT, WEB, num_threads, RECO_HPARAMS, USE_INTEL_EXT, args, tnow_str

    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    # Apply CLI/env configuration to globals used by helper functions.
    API_BASE_URL = args.api_base_url
    API_TIMEOUT = float(args.api_timeout)
    WEB = args.web_name
    num_threads = _resolve_num_threads(int(args.num_threads))
    RECO_HPARAMS = RecoHyperparams(api_limit=int(args.api_limit), model_C=float(args.reco_c))
    _configure_thread_env_vars(num_threads)

    logger.remove()
    log_level = settings.log_level.upper()
    if args.dry_run:
        log_level = "DEBUG"
    logger.add(sys.stdout, level=log_level)

    # Try to use Intel extensions (if available)
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        logger.debug(f"Intel scikit-learn extension enabled with {num_threads} threads")
        USE_INTEL_EXT = True
    except ImportError:
        logger.debug(f"Using standard sklearn with {num_threads} threads")
        USE_INTEL_EXT = False

    logger.debug(args)

    tnow = time.time()
    tnow_str = time.strftime("%b %d", time.localtime(tnow))  # e.g. "Nov 27"

    # read entire db simply into RAM
    tags = TagRepository.get_all_tags()
    ctags = CombinedTagRepository.get_all_combined_tags()
    keywords = KeywordRepository.get_all_keywords()
    emails = UserRepository.get_all_emails()

    # iterate all users, create recommendations, send emails
    num_sent = 0
    if args.user:
        if args.user not in tags:
            logger.error(f"user {args.user} not found in tags db")
            return 1
        tags = {args.user: tags[args.user]}

    for user, utags in tags.items():
        # verify that we have an email for this user
        email = emails.get(user, None)
        logger.debug(f"processing user {user} email {email}")
        if not email:
            logger.debug(f"skipping user {user}, no email")
            continue
        if args.user and user != args.user:
            logger.debug(f"skipping user {user}, not {args.user}")
            continue

        # verify that we have at least one positive example...
        try:
            num_papers_tagged = len(set().union(*utags.values())) if utags else 0
        except TypeError:
            # Defensive: if utags.values() is empty.
            num_papers_tagged = 0
        if num_papers_tagged < args.min_papers:
            logger.debug("skipping user %s, only has %d papers tagged" % (user, num_papers_tagged))
            continue

        # calculate the recommendations
        try:
            pids, scores = calculate_recommendation(utags, user=user, time_delta=args.time_delta)
            pids_set = set().union(*pids.values()) if pids.values() else set()
            logger.debug(f"From tags, found {len(pids_set)} papers for {user} within {args.time_delta} days")

            u_ctag = ctags.get(user, set())
            cpids, cscores = calculate_ctag_recommendation(u_ctag, utags, user=user, time_delta=args.time_delta)

            cpids_set = set().union(*cpids.values()) if cpids.values() else set()
            logger.debug(f"From ctags, found {len(cpids_set)} papers for {user} within {args.time_delta} days")

            ukeywords = keywords.get(user, {})

            kpids, kscores = search_keywords_recommendations(user, ukeywords, args.time_delta)
            kpids_set = set().union(*kpids.values()) if kpids.values() else set()
            logger.debug(f"From keywords, found {len(kpids_set)} papers for {user} within {args.time_delta} days")

            # Check if there are any recommendation results
            has_tag_recs = any(len(lst) > 0 for _tag, lst in pids.items())
            has_ctag_recs = any(len(lst) > 0 for _ctag, lst in cpids.items())
            has_keyword_recs = any(len(lst) > 0 for _key, lst in kpids.items())

            if not (has_tag_recs or has_ctag_recs or has_keyword_recs):
                logger.debug(f"skipping user {user}, no recommendations were produced")
                continue

            # render the html
            logger.debug(
                "rendering top max %d recommendations into a report for %s..." % (args.num_recommendations, user)
            )
            email_html = render_recommendations(
                user, utags, pids, scores, u_ctag, cpids, cscores, ukeywords, kpids, kscores
            )

            # temporarily for debugging write recommendations to disk for manual inspection
            if os.path.isdir("recco"):
                with open(f"recco/{user}.html", "w", encoding="utf-8") as f:
                    f.write(email_html)

            # actually send the email
            logger.debug("sending email...")
            n_send_try = 0
            while n_send_try < 3:
                try:
                    send_email(email, email_html)
                    break
                except Exception as e:
                    logger.warning(f"Failed to send email attempt {n_send_try + 1}: {e}")
                    n_send_try += 1
            if not args.dry_run:
                num_sent += 1
        except Exception as e:
            logger.error(f"meeting errors {str(e)} in processing user {user}")

    logger.success("done.")
    logger.success("sent %d emails" % (num_sent,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
