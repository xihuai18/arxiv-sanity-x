"""
Compose and send recommendation emails to arxiv-sanity-X users!

I run this script in a cron job to send out emails to the users with their
recommendations. There's a bit of copy paste code here but I expect that
the recommendations may become more complex in the future, so this is ok for now.

You'll notice that the file sendgrid_api_key.txt is not in the repo, you'd have
to manually register with sendgrid yourself, get an API key and put it in the file.
"""

import argparse
import os
import sys
import time
from multiprocessing import cpu_count
from typing import Dict, List, Set

import requests
from loguru import logger

from aslite.db import (
    get_combined_tags_db,
    get_email_db,
    get_keywords_db,
    get_metas_db,
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
            background: #dc3545;
            color: white;
            padding: 20px;
            text-align: center;
        }

        .header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 600;
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
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 2px solid #e74c3c;
        }

        .section-stats {
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 13px;
            color: #6c757d;
            border-left: 3px solid #007bff;
        }

        .paper-item {
            border: 1px solid #e9ecef;
            border-radius: 5px;
            margin-bottom: 15px;
            padding: 15px;
            background-color: #ffffff;
        }

        .paper-item:hover {
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .paper-header {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            flex-wrap: wrap;
            gap: 8px;
        }

        .score {
            background: transparent;
            color: #000000;
            padding: 3px 6px;
            border-radius: 10px;
            font-weight: 500;
            font-size: 11px;
            display: inline-block;
            margin-right: 6px;
        }

        .paper-source {
            display: inline-block;
            background-color: #b3d9ff;
            color: #0056b3;
            padding: 3px 6px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 500;
            margin-right: 6px;
            border: 1px solid #87ceeb;
        }

        .paper-source.keyword-source {
            background-color: #ffd280;
            color: #b8860b;
            border: 1px solid #daa520;
        }

        .paper-title {
            font-size: 15px;
            font-weight: 500;
            color: #2c3e50;
            margin: 8px 0;
            line-height: 1.3;
        }

        .paper-links {
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .paper-links a {
            display: inline-block;
            background-color: #dc3545;
            color: white;
            text-decoration: none;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
            margin-right: 4px;
            transition: background-color 0.2s ease;
        }

        .paper-links a:hover {
            background-color: #c82333;
        }

        .paper-links a.arxiv-link {
            background-color: #b31b1b;
        }

        .paper-links a.arxiv-link:hover {
            background-color: #9a1717;
        }

        .paper-links a.alphaxiv-link {
            background-color: transparent;
            color: #dc3545;
            font-weight: bold;
            padding: 0;
            border-radius: 0;
            font-size: 11px;
            margin-right: 8px;
        }

        .paper-links a.alphaxiv-link:hover {
            background-color: transparent;
            color: #c82333;
            text-decoration: underline;
        }

        .paper-links a.cool-link {
            background-color: transparent;
            color: #28a745;
            font-weight: bold;
            padding: 0;
            border-radius: 0;
            font-size: 11px;
        }

        .paper-links a.cool-link:hover {
            background-color: transparent;
            color: #218838;
            text-decoration: underline;
        }

        .paper-links .right-links {
            margin-left: auto;
            display: flex;
            gap: 6px;
        }

        .paper-authors {
            font-size: 13px;
            color: #6c757d;
            margin-bottom: 6px;
            font-style: italic;
        }

        .paper-date {
            font-size: 12px;
            color: #28a745;
            font-weight: 500;
            margin-bottom: 8px;
        }

        .paper-summary {
            font-size: 13px;
            color: #495057;
            line-height: 1.4;
            text-align: justify;
        }

        .footer {
            background-color: #f8f9fa;
            padding: 25px 30px;
            text-align: center;
            border-top: 1px solid #e9ecef;
        }

        .footer p {
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #6c757d;
        }

        .footer .brand {
            font-weight: 600;
            color: #2c3e50;
            font-size: 16px;
        }

        @media (max-width: 600px) {
            body {
                padding: 10px;
            }

            .container {
                border-radius: 8px;
            }

            .header, .content, .footer {
                padding: 20px;
            }

            .header h1 {
                font-size: 24px;
            }

            .score-cell {
                width: 60px;
                padding: 15px 10px;
            }

            .paper-cell {
                padding: 15px 10px;
            }
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="header">
            <h1>Arxiv Sanity X</h1>
        </div>

        <div class="content">
            <div class="greeting">
                Hi <strong>__USER__</strong>! Here are your personalized <a href="__HOST__" style="color: #007bff; text-decoration: none;">__WEB__</a> recommendations.
            </div>

            __SECTION_TAG__

            __SECTION_CTAG__

            __SECTION_KEYWORD__
        </div>

        <div class="footer">
            <p>To stop these emails, remove your email in your <a href="__HOST__/profile" style="color: #667eea; text-decoration: none;">account settings</a>.</p>
            <p>Your account: <strong>__ACCOUNT__</strong></p>
            <div class="brand">__WEB__</div>
        </div>
    </div>
</body>
</html>
"""


def calculate_ctag_recommendation(
    ctags: List[str],
    tags: dict,
    user: str,  # Add user parameter
    time_delta: int = 3,
):
    """Use API to call joint tag recommendations"""
    all_pids, all_scores = {}, {}

    for ctag in ctags:
        _tags = list(map(str.strip, ctag.split(",")))
        # Filter out valid tags (with papers)
        valid_tags = [tag for tag in _tags if tag in tags and len(tags[tag]) > 0]

        if not valid_tags:
            continue

        try:
            # Call API, pass tag name list and user information
            response = requests.post(
                f"{API_BASE_URL}/api/tags_search",
                json={
                    "tags": valid_tags,  # tag name list
                    "user": user,  # username
                    "logic": "and",
                    "time_delta": time_delta,
                    "limit": 1000,
                    "C": 0.1,
                },
                timeout=API_TIMEOUT,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    all_pids[ctag] = result["pids"]
                    all_scores[ctag] = result["scores"]
                else:
                    logger.error(f"API error for ctag {ctag}: {result.get('error')}")
            else:
                logger.error(f"API request failed for ctag {ctag}: {response.status_code}")

        except requests.RequestException as e:
            logger.error(f"API request exception for ctag {ctag}: {e}")

    return all_pids, all_scores


def calculate_recommendation(
    tags: dict,
    user: str,  # Add user parameter
    time_delta: int = 3,  # how recent papers are we recommending? in days
):
    """Use API to call single tag recommendations"""
    all_pids, all_scores = {}, {}

    for tag, tpids in tags.items():
        if len(tpids) == 0:
            continue

        try:
            # Call API, pass tag name and user information
            response = requests.post(
                f"{API_BASE_URL}/api/tag_search",
                json={
                    "tag_name": tag,  # tag name
                    "user": user,  # username
                    "time_delta": time_delta,
                    "limit": 1000,
                    "C": 0.1,
                },
                timeout=API_TIMEOUT,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    all_pids[tag] = result["pids"]
                    all_scores[tag] = result["scores"]
                else:
                    logger.error(f"API error for tag {tag}: {result.get('error')}")
            else:
                logger.error(f"API request failed for tag {tag}: {response.status_code}")

        except requests.RequestException as e:
            logger.error(f"API request exception for tag {tag}: {e}")

    return all_pids, all_scores


# -----------------------------------------------------------------------------
# Keywords recommendation
# -----------------------------------------------------------------------------
def search_keywords_recommendations(user: str, keywords: Dict[str, Set[str]], time_delta: int = 3):
    """Use API to call keyword search"""
    all_pids, all_scores = {}, {}

    for keyword, pids in keywords.items():
        try:
            # Call API
            response = requests.post(
                f"{API_BASE_URL}/api/keyword_search",
                json={"keyword": keyword, "time_delta": time_delta, "limit": 1000},
                timeout=API_TIMEOUT,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    # Exclude existing papers
                    rec_pids = result["pids"]
                    rec_scores = result["scores"]
                    keep = [i for i, pid in enumerate(rec_pids) if pid not in pids]
                    rec_pids = [rec_pids[i] for i in keep]
                    rec_scores = [rec_scores[i] for i in keep]

                    all_pids[keyword] = rec_pids
                    all_scores[keyword] = rec_scores
                else:
                    logger.error(f"API error for keyword {keyword}: {result.get('error')}")
            else:
                logger.error(f"API request failed for keyword {keyword}: {response.status_code}")

        except requests.RequestException as e:
            logger.error(f"API request exception for keyword {keyword}: {e}")

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
    out = out.replace("__USER__", user)
    out = out.replace("__HOST__", HOST)
    out = out.replace("__WEB__", WEB)

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
    if sum(len(kpids[keyword]) for keyword in keywords) > 0:
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
        # Each tag recommendation type recommends at most num_recommendations papers
        n = min(len(scores), args.num_recommendations)
        for score, pid in zip(scores[:n], pids[:n]):
            p = pdb[pid]
            authors = ", ".join(a["name"] for a in p["authors"])
            # crop the abstract
            summary = p["summary"]
            summary = summary[: min(500, len(summary))]
            if len(summary) == 500:
                summary += "..."
            # create the url that will feature this paper on top and also show the most similar papers
            url = f"{HOST}/?rank=pid&pid=" + pid
            arxiv_url = f"https://arxiv.org/abs/{pid}"

            parts.append(
                """
    <div class="paper-item">
        <div class="paper-header">
            <div class="paper-source">%s</div>
            <div class="score">%.2f</div>
        </div>
        <div class="paper-title">%s</div>
        <div class="paper-links">
            <a href="%s">Sanity</a>
            <a href="%s/summary?pid=%s">📝 Sanity Summary</a>
            <a href="%s" class="arxiv-link">Arxiv</a>
            <div class="right-links">
                <a href="https://www.alphaxiv.org/overview/%s" class="alphaxiv-link">alphaXiv</a>
                <a href="https://papers.cool/arxiv/%s" class="cool-link">Cool</a>
            </div>
        </div>
        <div class="paper-authors">%s</div>
        <div class="paper-date">📅 %s</div>
        <div class="paper-summary">%s</div>
    </div>
    """
                % (
                    max_source_tag[pid],
                    score,
                    p["title"],
                    url,
                    HOST,
                    pid,
                    arxiv_url,
                    pid,
                    pid,
                    authors,
                    p["_time_str"],
                    summary,
                )
            )

        # render the recommendations
        final = "".join(parts)

        # render the stats
        num_papers_tagged = len(set().union(*tags.values()))
        tags_str = ", ".join(['"%s" (%d)' % (t, len(tags[t])) for t in tags.keys()])
        stats = f"We took the {num_papers_tagged} papers across your {len(tags)} tags ({tags_str}) and \
                ranked {len(pids)} papers that showed up on arxiv over the last \
                {args.time_delta} days using tfidf SVMs over paper abstracts. Below are the \
                top {n} papers. Remember that the more you tag, \
                the better this gets:"

        # render the complete section
        section_html = f"""
            <div class="section">
                <div class="section-title">Tag-based Recommendations</div>
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
        # Each ctag recommendation type recommends at most num_recommendations papers
        n = min(len(scores), args.num_recommendations)
        for score, pid in zip(scores[:n], pids[:n]):
            p = pdb[pid]
            authors = ", ".join(a["name"] for a in p["authors"])
            # crop the abstract
            summary = p["summary"]
            summary = summary[: min(500, len(summary))]
            if len(summary) == 500:
                summary += "..."
            # create the url that will feature this paper on top and also show the most similar papers
            url = f"{HOST}/?rank=pid&pid=" + pid
            arxiv_url = f"https://arxiv.org/abs/{pid}"
            parts.append(
                """
    <div class="paper-item">
        <div class="paper-header">
            <div class="paper-source">%s</div>
            <div class="score">%.2f</div>
        </div>
        <div class="paper-title">%s</div>
        <div class="paper-links">
            <a href="%s">Sanity</a>
            <a href="%s/summary?pid=%s">📝 Sanity Summary</a>
            <a href="%s" class="arxiv-link">Arxiv</a>
            <div class="right-links">
                <a href="https://www.alphaxiv.org/overview/%s" class="alphaxiv-link">alphaXiv</a>
                <a href="https://papers.cool/arxiv/%s" class="cool-link">Cool</a>
            </div>
        </div>
        <div class="paper-authors">%s</div>
        <div class="paper-date">📅 %s</div>
        <div class="paper-summary">%s</div>
    </div>
    """
                % (
                    max_source_ctag[pid],
                    score,
                    p["title"],
                    url,
                    HOST,
                    pid,
                    arxiv_url,
                    pid,
                    pid,
                    authors,
                    p["_time_str"],
                    summary,
                )
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

        ctags_str = ", ".join(['"%s"' % (t) for t in ctags])
        stats = f"We took the {num_papers_ctagged} papers across your {len(ctags)} registered combined tags ({ctags_str}) and \
                ranked {len(pids)} papers that showed up on arxiv over the last \
                {args.time_delta} days using tfidf SVMs over paper abstracts. Below are the \
                top {n} papers. Remember that the more you tag, \
                the better this gets:"

        # render the complete section
        section_html = f"""
            <div class="section">
                <div class="section-title">Combined Tag Recommendations</div>
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
        # Each keyword recommendation type recommends at most num_recommendations papers
        n = min(len(scores), args.num_recommendations)
        for score, pid in zip(scores[:n], pids[:n]):
            p = pdb[pid]
            authors = ", ".join(a["name"] for a in p["authors"])
            # crop the abstract
            summary = p["summary"]
            summary = summary[: min(500, len(summary))]
            if len(summary) == 500:
                summary += "..."
            # create the url that will feature this paper on top and also show the most similar papers
            url = f"{HOST}/?rank=pid&pid=" + pid
            arxiv_url = f"https://arxiv.org/abs/{pid}"
            parts.append(
                """
    <div class="paper-item">
        <div class="paper-header">
            <div class="paper-source keyword-source">%s</div>
            <div class="score">%.2f</div>
        </div>
        <div class="paper-title">%s</div>
        <div class="paper-links">
            <a href="%s">Sanity</a>
            <a href="%s/summary?pid=%s">📝 Sanity Summary</a>
            <a href="%s" class="arxiv-link">Arxiv</a>
            <div class="right-links">
                <a href="https://www.alphaxiv.org/overview/%s" class="alphaxiv-link">alphaXiv</a>
                <a href="https://papers.cool/arxiv/%s" class="cool-link">Cool</a>
            </div>
        </div>
        <div class="paper-authors">%s</div>
        <div class="paper-date">📅 %s</div>
        <div class="paper-summary">%s</div>
    </div>
    """
                % (
                    max_source_keyword[pid],
                    score,
                    p["title"],
                    url,
                    HOST,
                    pid,
                    arxiv_url,
                    pid,
                    pid,
                    authors,
                    p["_time_str"],
                    summary,
                )
            )

        # render the recommendations
        final = "".join(parts)

        # render the stats
        keywords_str = ", ".join(['"%s"' % k for k, pids in keywords.items()])
        stats = f"We search your {len(keywords)} keywords ({keywords_str}) and \
                ranked {len(pids)} papers that showed up on arxiv over the last \
                {args.time_delta} days using tfidf SVMs over paper abstracts. Below are the \
                top {n} papers. Remember that the more keywords, \
                the better this gets:"

        # render the complete section
        section_html = f"""
            <div class="section">
                <div class="section-title">Keyword-based Recommendations</div>
                <div class="section-stats">{stats}</div>
                <div>{final}</div>
            </div>"""
        out = out.replace("__SECTION_KEYWORD__", section_html)
    else:
        out = out.replace("__SECTION_KEYWORD__", "")

    # render the account
    out = out.replace("__ACCOUNT__", user)

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
    with get_metas_db() as mdb:
        metas = {k: v for k, v in mdb.items()}

    # read entire db simply into RAM
    with get_email_db() as edb:
        emails = {k: v for k, v in edb.items()}

    # keep the papers as only a handle, since this can be larger
    pdb = get_papers_db()

    # iterate all users, create recommendations, send emails
    num_sent = 0
    if args.user:
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
            html = render_recommendations(user, utags, pids, scores, u_ctag, cpids, cscores, ukeywords, kpids, kscores)
            # temporarily for debugging write recommendations to disk for manual inspection
            if os.path.isdir("recco"):
                with open(f"recco/{user}.html", "w") as f:
                    f.write(html)
            # actually send the email
            logger.info("sending email...")
            n_send_try = 0
            while n_send_try < 3:
                try:
                    send_email(email, html)
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
