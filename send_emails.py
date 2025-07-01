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
from vars import HOST

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

# API调用配置
API_BASE_URL = "http://localhost:55555"  # serve.py默认端口
API_TIMEOUT = 30  # API请求超时时间（秒）

# -----------------------------------------------------------------------------
# the html template for the email


WEB = "Arxiv Sanity X"

template = """
<!DOCTYPE HTML>
<html>

<head>
<style>
body {
    font-family: Arial, sans-serif;
}
.s {
    font-weight: bold;
    margin-right: 10px;
}
.a {
    color: #333;
}
.u {
    font-size: 12px;
    color: #333;
    margin-bottom: 10px;
}
.f {
    color: #933;
    display: inline-block;
}
</style>
</head>

<body>

<br><br>
<div>Hi! __USER__. Here are your <a href="__HOST__">__WEB__</a> recommendations.</div>
<br>

<div>
    <b>__STATS_TAG__</b>
</div>

<div>
    __CONTENT_TAG__
</div>
<br>

<div>
    <b>__STATS_CTAG__</b>
</div>

<div>
    __CONTENT_CTAG__
</div>
<br>

<div>
    <b>__STATS_KEYWORD__</b>
</div>

<div>
    __CONTENT_KEYWORD__
</div>


<br><br>
<div>
To stop these emails remove your email in your <a href="__HOST__/profile">account</a> settings. (your account is __ACCOUNT__).
</div>
<div> __WEB__ </div>

</body>
</html>
"""


def calculate_ctag_recommendation(
    ctags: List[str],
    tags: dict,
    user: str,  # 添加user参数
    time_delta: int = 3,
):
    """使用API调用联合tag推荐"""
    all_pids, all_scores = {}, {}

    for ctag in ctags:
        _tags = list(map(str.strip, ctag.split(",")))
        # 过滤出有效的tags（有论文的）
        valid_tags = [tag for tag in _tags if tag in tags and len(tags[tag]) > 0]

        if not valid_tags:
            continue

        try:
            # 调用API，传递tag名称列表和用户信息
            response = requests.post(
                f"{API_BASE_URL}/api/tags_search",
                json={
                    "tags": valid_tags,  # tag名称列表
                    "user": user,  # 用户名
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
    user: str,  # 添加user参数
    time_delta: int = 3,  # how recent papers are we recommending? in days
):
    """使用API调用单个tag推荐"""
    all_pids, all_scores = {}, {}

    for tag, tpids in tags.items():
        if len(tpids) == 0:
            continue

        try:
            # 调用API，传递tag名称和用户信息
            response = requests.post(
                f"{API_BASE_URL}/api/tag_search",
                json={
                    "tag_name": tag,  # tag名称
                    "user": user,  # 用户名
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
    """使用API调用关键词搜索"""
    all_pids, all_scores = {}, {}

    for keyword, pids in keywords.items():
        try:
            # 调用API
            response = requests.post(
                f"{API_BASE_URL}/api/keyword_search",
                json={"keyword": keyword, "time_delta": time_delta, "limit": 1000},
                timeout=API_TIMEOUT,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    # 排除已有的论文
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
    # render the paper recommendations into the html template
    if sum(len(tag_pids[tag]) for tag in tag_pids) > 0:
        # first we are going to merge all of the papers / scores together using a MAX
        max_score = {}
        max_source_tag = {}
        for tag in tag_pids:
            for pid, score in zip(tag_pids[tag], tag_scores[tag]):
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
        # 每个tag推荐类型各自最多推荐num_recommendations个论文
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
    <tr>
    <td valign="top"><div class="s">%.2f</div></td>
    <td>
    <div class="f">(%s)</div> %s <a href="%s">Sanity Link</a> <a href="%s">Arxiv Link</a>
    <div class="a">%s</div>
    <div class="u">%s</div>
    </td>
    </tr>
    """
                % (score, max_source_tag[pid], p["title"], url, arxiv_url, authors, summary)
            )

        # render the recommendations
        final = "<table>" + "".join(parts) + "</table>"
        out = out.replace("__CONTENT_TAG__", final)

        # render the stats
        num_papers_tagged = len(set().union(*tags.values()))
        tags_str = ", ".join(['"%s" (%d)' % (t, len(tags[t])) for t in tags.keys()])
        stats = f"We took the {num_papers_tagged} papers across your {len(tags)} tags ({tags_str}) and \
                ranked {len(pids)} papers that showed up on arxiv over the last \
                {args.time_delta} days using tfidf SVMs over paper abstracts. Below are the \
                top {n} papers. Remember that the more you tag, \
                the better this gets:"
        out = out.replace("__STATS_TAG__", stats)
    else:
        out = out.replace("__CONTENT_TAG__", "")
        out = out.replace("__STATS_TAG__", "")

    if sum(len(ctag_pids[ctag]) for ctag in ctag_pids) > 0:
        # first we are going to merge all of the papers / scores together using a MAX
        max_score = {}
        max_source_ctag = {}
        for ctag in ctag_pids:
            for pid, score in zip(ctag_pids[ctag], ctag_scores[ctag]):
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
        # 每个ctag推荐类型各自最多推荐num_recommendations个论文
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
    <tr>
    <td valign="top"><div class="s">%.2f</div></td>
    <td>
    <div class="f">(%s)</div> %s <a href="%s">Sanity Link</a> <a href="%s">Arxiv Link</a>
    <div class="a">%s</div>
    <div class="u">%s</div>
    </td>
    </tr>
    """
                % (score, max_source_ctag[pid], p["title"], url, arxiv_url, authors, summary)
            )

        # render the recommendations
        final = "<table>" + "".join(parts) + "</table>"
        out = out.replace("__CONTENT_CTAG__", final)

        # render the stats
        # 计算联合tags涉及的所有论文数量
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
        out = out.replace("__STATS_CTAG__", stats)
    else:
        out = out.replace("__CONTENT_CTAG__", "")
        out = out.replace("__STATS_CTAG__", "")

    if sum(len(kpids[keyword]) for keyword in keywords) > 0:
        # first we are going to merge all of the papers / scores together using a MAX
        max_score = {}
        max_source_keyword = {}
        for keyword in kpids:
            for pid, score in zip(kpids[keyword], kscores[keyword]):
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
        # 每个keyword推荐类型各自最多推荐num_recommendations个论文
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
    <tr>
    <td valign="top"><div class="s">%.2f</div></td>
    <td>
    <div class="f">(%s)</div> %s <a href="%s">Sanity Link</a> <a href="%s">Arxiv Link</a>
    <div class="a">%s</div>
    <div class="u">%s</div>
    </td>
    </tr>
    """
                % (score, max_source_keyword[pid], p["title"], url, arxiv_url, authors, summary)
            )

        # render the recommendations
        final = "<table>" + "".join(parts) + "</table>"
        out = out.replace("__CONTENT_KEYWORD__", final)

        # render the stats
        keywords_str = ", ".join(['"%s"' % k for k, pids in keywords.items()])
        stats = f"We search your {len(keywords)} keywords ({keywords_str}) and \
                ranked {len(pids)} papers that showed up on arxiv over the last \
                {args.time_delta} days using tfidf SVMs over paper abstracts. Below are the \
                top {n} papers. Remember that the more keywords, \
                the better this gets:"
        out = out.replace("__STATS_KEYWORD__", stats)
    else:
        out = out.replace("__CONTENT_KEYWORD__", "")
        out = out.replace("__STATS_KEYWORD__", "")

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

            # 检查是否有任何推荐结果
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
