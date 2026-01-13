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
from multiprocessing import cpu_count
from typing import Dict, List, Set

import requests
from loguru import logger

from aslite.db import (
    get_combined_tags_db,
    get_email_db,
    get_keywords_db,
    get_papers_db,
    get_tags_db,
)
from vars import HOST, SERVE_PORT

# Set multi-threading environment variables
num_threads = min(cpu_count(), 192)  # Reasonable thread limit
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

# Try to use Intel extensions (if available)
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    logger.info(f"Intel scikit-learn extension enabled with {num_threads} threads")
    USE_INTEL_EXT = True
except ImportError:
    logger.info(f"Using standard sklearn with {num_threads} threads")
    USE_INTEL_EXT = False

# API call configuration
API_BASE_URL = f"http://localhost:{SERVE_PORT}"  # serve.py default port
API_TIMEOUT = 120  # API request timeout (seconds)

# -----------------------------------------------------------------------------
# the html template for the email


WEB = "Arxiv Sanity X"

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
            max-width: 700px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #b31b1b 0%, #8b1515 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }

        .header h1 {
            margin: 0;
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 1px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.15);
        }

        .header .subtitle {
            margin-top: 8px;
            font-size: 14px;
            opacity: 0.9;
        }

        .content {
            padding: 20px;
        }

        .greeting {
            font-size: 16px;
            margin-bottom: 20px;
            color: #2c3e50;
        }

        .section {
            margin-bottom: 25px;
        }

        .section-title {
            font-size: 18px;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 3px solid #b31b1b;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .section-stats {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 12px 18px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #6c757d;
            border-left: 4px solid #b31b1b;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .paper-item {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            margin-bottom: 18px;
            padding: 18px;
            background-color: #ffffff;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .paper-item::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            width: 4px;
            background: linear-gradient(180deg, #b31b1b 0%, #8b1515 100%);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .paper-item:hover {
            box-shadow: 0 4px 12px rgba(179, 27, 27, 0.15);
            transform: translateY(-2px);
        }

        .paper-item:hover::before {
            opacity: 1;
        }

        .paper-header {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
            gap: 8px;
        }

        .score {
            background: linear-gradient(135deg, #b31b1b 0%, #8b1515 100%);
            color: #ffffff;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 11px;
            display: inline-block;
            margin-right: 6px;
            box-shadow: 0 2px 4px rgba(179, 27, 27, 0.3);
        }

        .paper-source {
            display: inline-block;
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            color: #0369a1;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 6px;
            border: none;
        }

        .paper-source.keyword-source {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            color: #b45309;
            border: none;
        }

        .paper-source.ctag-source {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            color: #047857;
            border: none;
        }

        .paper-title {
            font-size: 16px;
            font-weight: 600;
            color: #1e293b;
            margin: 10px 0;
            line-height: 1.4;
        }

        .paper-links {
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .paper-links a {
            display: inline-block;
            background: linear-gradient(135deg, #b31b1b 0%, #8b1515 100%);
            color: white;
            text-decoration: none;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 11px;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(179, 27, 27, 0.3);
        }

        .paper-links a:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(179, 27, 27, 0.4);
        }

        .paper-links a.arxiv-link {
            background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
            box-shadow: 0 2px 4px rgba(30, 64, 175, 0.3);
        }

        .paper-links a.arxiv-link:hover {
            box-shadow: 0 4px 8px rgba(30, 64, 175, 0.4);
        }

        .paper-links a.alphaxiv-link,
        .paper-links a.cool-link {
            background: linear-gradient(135deg, #64748b 0%, #475569 100%);
            color: white;
            padding: 6px 12px;
            box-shadow: 0 2px 4px rgba(100, 116, 139, 0.3);
        }

        .paper-links a.alphaxiv-link:hover,
        .paper-links a.cool-link:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(100, 116, 139, 0.4);
        }

        .paper-links .right-links {
            margin-left: auto;
            display: flex;
            gap: 12px;
        }

        .paper-authors {
            font-size: 13px;
            color: #64748b;
            margin-bottom: 8px;
            font-style: italic;
            padding-left: 2px;
        }

        .paper-authors::before {
            content: "👤 ";
        }

        .paper-date {
            font-size: 12px;
            color: #059669;
            font-weight: 600;
            margin-bottom: 10px;
            padding-left: 2px;
        }

        .paper-tldr {
            background-color: #fef2f2;
            background-image: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border-left: 3px solid #b31b1b;
            padding: 10px 12px;
            margin: 12px 0 0 0;
            border-radius: 8px;
            font-size: 13px;
            line-height: 1.5;
            color: #374151;
        }

        .paper-tldr .tldr-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #b31b1b;
            margin-bottom: 4px;
        }

        .paper-tldr .tldr-text {
            margin: 0;
        }

        .paper-summary {
            font-size: 13px;
            color: #475569;
            line-height: 1.6;
            text-align: justify;
            background: #f8fafc;
            padding: 12px;
            border-radius: 8px;
            border-left: 3px solid #e2e8f0;
        }

        .footer {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 30px;
            text-align: center;
            border-top: 1px solid #e2e8f0;
        }

        .footer p {
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #64748b;
        }

        .footer a {
            color: #b31b1b;
            text-decoration: none;
            font-weight: 500;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        .footer .brand {
            font-weight: 700;
            color: #1e293b;
            font-size: 18px;
            margin-top: 15px;
            background: linear-gradient(135deg, #b31b1b 0%, #8b1515 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        @media (max-width: 600px) {
            body {
                padding: 10px;
            }

            .container {
                border-radius: 12px;
            }

            .header, .content, .footer {
                padding: 20px;
            }

            .header h1 {
                font-size: 22px;
            }

            .paper-item {
                padding: 14px;
                margin-bottom: 14px;
            }

            .paper-links {
                gap: 6px;
            }

            .paper-links a {
                padding: 5px 10px;
                font-size: 10px;
            }
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


from summary_utils import markdown_to_email_html
from summary_utils import read_tldr_from_summary_file as _extract_tldr_from_summary


def _render_paper_html(paper: dict, pid: str, score: float, source_label: str, source_class: str = "paper-source"):
    title = _h(paper.get("title", ""))
    authors = _h(", ".join(a.get("name", "") for a in paper.get("authors", [])))
    summary = _h(_crop_summary(paper.get("summary", "")))
    time_str = _h(paper.get("_time_str", ""))
    tldr_raw = _extract_tldr_from_summary(pid)
    tldr = markdown_to_email_html(tldr_raw) if tldr_raw else ""

    url = _h(f"{HOST}/?rank=pid&pid={pid}")
    summary_url = _h(f"{HOST}/summary?pid={pid}")
    arxiv_url = _h(f"https://arxiv.org/abs/{pid}")
    alphaxiv_url = _h(f"https://www.alphaxiv.org/overview/{pid}")
    cool_url = _h(f"https://papers.cool/arxiv/{pid}")

    # Render TL;DR section if available (tldr is already HTML from markdown_to_email_html)
    tldr_html = ""
    if tldr:
        tldr_html = f"""
        <div class="paper-tldr">
            <div class="tldr-label">💡 TL;DR</div>
            <div class="tldr-text">{tldr}</div>
        </div>
        """

    return f"""
    <div class="paper-item">
        <div class="paper-header">
            <div class="score">⭐ {float(score):.2f}</div>
            <div class="{source_class}">🏷️ {_h(source_label)}</div>
        </div>
        <div class="paper-title">📄 {title}</div>
        <div class="paper-links">
            <a href="{url}">🔍 Sanity</a>
            <a href="{summary_url}">📝 Summary</a>
            <a href="{arxiv_url}" class="arxiv-link">📚 arXiv</a>
            <div class="right-links">
                <a href="{alphaxiv_url}" class="alphaxiv-link">αXiv</a>
                <a href="{cool_url}" class="cool-link">Cool</a>
            </div>
        </div>
        <div class="paper-authors">{authors}</div>
        <div class="paper-date">📅 {time_str}</div>{tldr_html}
        <div class="paper-summary">{summary}</div>
    </div>
    """


def _api_worker_count(n_tasks: int) -> int:
    try:
        max_workers = int(os.environ.get("ARXIV_SANITY_EMAIL_API_WORKERS", "8"))
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
            "limit": 1000,
            "C": 0.1,
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
            "limit": 1000,
            "C": 0.1,
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
        payload = {"keyword": keyword, "time_delta": time_delta, "limit": 1000}
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
                max_score[pid] = max(max_score.get(pid, -99999), score)  # lol
                if max_score[pid] == score:
                    max_source_tag[pid] = tag

        # now we have a dict of pid -> max score. sort by score
        max_score_list = sorted(max_score.items(), key=lambda x: x[1], reverse=True)
        if max_score_list:
            pids, scores = zip(*max_score_list)
        else:
            pids, scores = [], []

        # now render the html for each individual recommendation
        parts = []
        for score, pid in zip(scores, pids):
            if len(parts) >= args.num_recommendations:
                break
            p = pdb.get(pid)
            if p is None:
                logger.warning(f"Missing paper in db for tag recommendation: {pid}")
                continue
            parts.append(_render_paper_html(p, pid, score, max_source_tag.get(pid, "")))

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

        # now render the html for each individual recommendation
        parts = []
        for score, pid in zip(scores, pids):
            if len(parts) >= args.num_recommendations:
                break
            p = pdb.get(pid)
            if p is None:
                logger.warning(f"Missing paper in db for ctag recommendation: {pid}")
                continue
            parts.append(
                _render_paper_html(p, pid, score, max_source_ctag.get(pid, ""), source_class="paper-source ctag-source")
            )

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

        # now render the html for each individual recommendation
        parts = []
        for score, pid in zip(scores, pids):
            if len(parts) >= args.num_recommendations:
                break
            p = pdb.get(pid)
            if p is None:
                logger.warning(f"Missing paper in db for keyword recommendation: {pid}")
                continue
            parts.append(
                _render_paper_html(
                    p, pid, score, max_source_keyword.get(pid, ""), source_class="paper-source keyword-source"
                )
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

        from vars import (
            email_passwd,
            email_username,
            from_email,
            smtp_port,
            smtp_server,
        )

        from_email = from_email
        # construct the email
        message = MIMEText(html, "html", "utf-8")
        subject = tnow_str + " Arxiv Sanity X recommendations"
        message["Subject"] = Header(subject, "utf-8")

        try:
            mail = smtplib.SMTP_SSL(smtp_server, smtp_port)
            mail.login(email_username, email_passwd)
            mail.sendmail(from_email, [to_email], message.as_string())
        except smtplib.SMTPException as e:
            logger.error(e)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
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
        type=int,
        default=0,
        help="if set to 1 do not actually send the emails",
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
    args = parser.parse_args()
    logger.remove()
    if args.dry_run:
        logger.add(sys.stdout, level="DEBUG")
    else:
        # logger.add(sys.stdout, level="DEBUG")
        logger.add(sys.stdout, level="INFO")
    logger.info(args)

    tnow = time.time()
    tnow_str = time.strftime("%b %d", time.localtime(tnow))  # e.g. "Nov 27"

    # read entire db simply into RAM
    with get_tags_db() as tags_db:
        tags = {k: v for k, v in tags_db.items()}

    with get_combined_tags_db() as ctags_db:
        ctags = {k: v for k, v in ctags_db.items()}

        # read entire db simply into RAM
    with get_keywords_db() as keywords_db:
        keywords = {k: v for k, v in keywords_db.items()}

    # read entire db simply into RAM
    with get_email_db() as edb:
        emails = {k: v for k, v in edb.items()}

    # keep the papers as only a handle, since this can be larger
    pdb = get_papers_db()

    # iterate all users, create recommendations, send emails
    num_sent = 0
    if args.user:
        if args.user not in tags:
            logger.error(f"user {args.user} not found in tags db")
            sys.exit(1)
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
        num_papers_tagged = len(set().union(*utags.values()))
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
            has_tag_recs = any(len(lst) > 0 for tag, lst in pids.items())
            has_ctag_recs = any(len(lst) > 0 for ctag, lst in cpids.items())
            has_keyword_recs = any(len(lst) > 0 for key, lst in kpids.items())

            if not (has_tag_recs or has_ctag_recs or has_keyword_recs):
                logger.info(f"skipping user {user}, no recommendations were produced")
                continue
            # render the html
            logger.info(
                "rendering top max %d recommendations into a report for %s..." % (args.num_recommendations, user)
            )
            email_html = render_recommendations(
                user, utags, pids, scores, u_ctag, cpids, cscores, ukeywords, kpids, kscores
            )
            # temporarily for debugging write recommendations to disk for manual inspection
            if os.path.isdir("recco"):
                with open(f"recco/{user}.html", "w") as f:
                    f.write(email_html)
            # actually send the email
            logger.info("sending email...")
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
