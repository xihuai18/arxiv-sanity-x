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
from typing import Dict, List, Set

import numpy as np
from loguru import logger
from sklearn import svm

from aslite.db import (
    get_combined_tags_db,
    get_email_db,
    get_keywords_db,
    get_metas_db,
    get_papers_db,
    get_tags_db,
    load_features,
)
from vars import HOST

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
    time_delta: int = 3,
):
    db_x, db_pids = features["x"], features["pids"]
    all_pids, all_scores = {}, {}
    deltat = time_delta * 60 * 60 * 24  # allowed time delta in seconds
    for ctag in ctags:
        _tags = list(map(str.strip, ctag.split(",")))
        ctag_tpids = []
        for t_i, tag in enumerate(_tags):
            tpids = tags[tag]
            if len(tpids) == 0:
                continue

            ctag_tpids += tpids

        keep = [i for i, pid in enumerate(db_pids) if (tnow - metas[pid]["_time"]) < deltat or pid in ctag_tpids]
        logger.debug(f"Keep {len(keep)} pids for ctag {ctag}")
        if len(keep) == len(ctag_tpids):
            continue

        pids = [db_pids[i] for i in keep]

        x = db_x[keep]
        n = x.shape[0]
        # construct the positive set for this tag
        ptoi, itop = {}, {}
        for i, p in enumerate(pids):
            ptoi[p] = i
            itop[i] = p
        y = np.zeros(n, dtype=np.float32)
        for t_i, tag in enumerate(_tags):
            tpids = tags[tag]
            for pid in tpids:
                y[ptoi[pid]] = 1.0 + t_i
        # classify
        clf = svm.LinearSVC(class_weight="balanced", verbose=False, max_iter=10000, tol=1e-5, C=0.1, dual="auto")
        clf.fit(x, y)
        s = clf.decision_function(x)
        if len(s.shape) > 1:
            s = s[:, 1:].mean(axis=-1)
        sortix = np.argsort(-s)
        sortix = [ix for ix in sortix if s[ix] >= -0.25]
        pids = [itop[ix] for ix in sortix]
        scores = [100 * float(s[ix]) for ix in sortix]

        # finally exclude the papers we already have tagged
        have = set().union(*tags.values())
        keep = [i for i, pid in enumerate(pids) if pid not in have]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

        # store results
        all_pids[ctag] = pids
        all_scores[ctag] = scores

    return all_pids, all_scores


def calculate_recommendation(
    tags: dict,
    time_delta: int = 3,  # how recent papers are we recommending? in days
):
    # tags: Dict[str, set()]
    # a bit of preprocessing
    db_x, db_pids = features["x"], features["pids"]

    # loop over all the tags
    all_pids, all_scores = {}, {}
    deltat = time_delta * 60 * 60 * 24  # allowed time delta in seconds
    for tag, tpids in tags.items():
        if len(tpids) == 0:
            continue

        keep = [i for i, pid in enumerate(db_pids) if (tnow - metas[pid]["_time"]) < deltat or pid in tpids]
        if len(keep) == len(tpids):
            continue
        logger.debug(f"keep {len(keep)} papers according to time for tag {tag}")
        pids = [db_pids[i] for i in keep]
        x = db_x[keep]
        n = x.shape[0]
        # construct the positive set for this tag
        ptoi, itop = {}, {}
        for i, p in enumerate(pids):
            ptoi[p] = i
            itop[i] = p
        y = np.zeros(n, dtype=np.float32)
        for pid in tpids:
            y[ptoi[pid]] = 1.0

        # classify
        clf = svm.LinearSVC(class_weight="balanced", verbose=False, max_iter=10000, tol=1e-5, C=0.1, dual="auto")
        clf.fit(x, y)
        s = clf.decision_function(x)
        sortix = np.argsort(-s)
        sortix = [ix for ix in sortix if s[ix] >= -0.25]
        pids = [itop[ix] for ix in sortix]
        scores = [100 * float(s[ix]) for ix in sortix]

        # finally exclude the papers we already have tagged
        have = set().union(*tags.values())
        keep = [i for i, pid in enumerate(pids) if pid not in have]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

        # store results
        all_pids[tag] = pids
        all_scores[tag] = scores

    return all_pids, all_scores


# -----------------------------------------------------------------------------
# Keywords recommendation
# -----------------------------------------------------------------------------
def search_rank(q: str = "", time_pids=[]):
    if not q:
        return [], []  # no query? no results
    q = q.lower().strip()
    qs = q.split()  # split query by spaces and lowercase

    match = lambda s: sum(min(3, s.lower().count(qp)) for qp in qs) / len(qs)
    matchu = lambda s: sum(int(s.lower().count(qp) > 0) for qp in qs) / len(qs)
    matcht = lambda s: int(q in s.lower())
    pairs = []
    for pid in time_pids:
        p = pdb[pid]
        score = 0.0
        score += 10.0 * matchu(" ".join([a["name"] for a in p["authors"]]))
        score += 20.0 * matchu(p["title"])
        score += 10.0 * matcht(p["title"])
        score += 5.0 * match(p["summary"])
        if score >= 10:
            pairs.append((score, pid))

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
    return pids, scores


def search_keywords_recommendations(user: str, keywords: Dict[str, Set[str]], time_delta: int = 3):
    all_pids, all_scores = {}, {}
    deltat = time_delta * 60 * 60 * 24
    db_pids = pdb.keys()
    for keyword, pids in keywords.items():
        time_pids = [pid for pid in db_pids if (tnow - metas[pid]["_time"]) < deltat]
        s_pids, s_scores = search_rank(keyword, time_pids)
        keep = [i for i, s_pid in enumerate(s_pids) if s_pid not in pids]
        s_pids, s_scores = [s_pids[i] for i in keep], [s_scores[i] for i in keep]

        all_pids[keyword] = s_pids
        all_scores[keyword] = s_scores
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
    if sum(len(tag_pids[tag]) for tag in tags) > 0:
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
        pids, scores = zip(*max_score_list)

        # now render the html for each individual recommendation
        parts = []
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
        tags_str = ", ".join(['"%s" (%d)' % (t, len(pids)) for t, pids in tags.items()])
        stats = f"We took the {num_papers_tagged} papers across your {len(tags)} tags ({tags_str}) and \
                ranked {len(pids)} papers that showed up on arxiv over the last \
                {args.time_delta} days using tfidf SVMs over paper abstracts. Below are the \
                top {n} papers. Remember that the more you tag, \
                the better this gets:"
        out = out.replace("__STATS_TAG__", stats)
    else:
        out = out.replace("__CONTENT_TAG__", "")
        out = out.replace("__STATS_TAG__", "")

    if sum(len(ctag_pids[ctag]) for ctag in ctags) > 0:
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
        pids, scores = zip(*max_score_list)

        # now render the html for each individual recommendation
        parts = []
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
        num_papers_ctagged = len(set().union(*tags.values()))
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
        pids, scores = zip(*max_score_list)

        # now render the html for each individual recommendation
        parts = []
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

    # read tfidf features into RAM
    features = load_features()

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
            pids, scores = calculate_recommendation(utags, time_delta=args.time_delta)
            pids_set = set().union(*pids.values())
            logger.debug(f"From tags, found {len(pids_set)} papers for {user} within {args.time_delta} days")

            u_ctag = ctags.get(user, set())
            cpids, cscores = calculate_ctag_recommendation(u_ctag, utags, time_delta=args.time_delta)

            cpids_set = set().union(*cpids.values())
            logger.debug(f"From ctags, found {len(cpids_set)} papers for {user} within {args.time_delta} days")

            ukeywords = keywords.get(user, {})

            kpids, kscores = search_keywords_recommendations(user, ukeywords, args.time_delta)
            pids_set = set().union(*kpids.values())
            logger.debug(f"From keywords, found {len(pids_set)} papers for {user} within {args.time_delta} days")

            if all(len(lst) == 0 for tag, lst in pids.items()) and all(len(lst) == 0 for key, lst in kpids.items()):
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
                except:
                    n_send_try += 1
            if not args.dry_run:
                num_sent += 1
        except Exception as e:
            logger.error(f"meeting errors {str(e)} in processing user {user}")

    logger.success("done.")
    logger.success("sent %d emails" % (num_sent,))
