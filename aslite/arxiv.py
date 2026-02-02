"""
Utils for dealing with arxiv API and related processing
"""

import logging
import random
import time
import urllib.error
import urllib.request
from collections import OrderedDict

import feedparser

from config import settings

logger = logging.getLogger(__name__)

_ARXIV_RETRY_MAX_TRIES = 3


def _sleep_backoff(attempt: int, base_s: float = 1.0, cap_s: float = 10.0) -> None:
    """Sleep with exponential backoff and jitter."""

    sleep_s = min(cap_s, base_s * (2**attempt))
    # Add small jitter to reduce thundering herd.
    jitter_s = sleep_s * 0.15 * random.random()
    time.sleep(sleep_s + jitter_s)


def get_response(search_query, start_index=0, max_r=100):
    """pings arxiv.org API to fetch a batch of 100 papers"""
    # fetch raw response
    base_url = "https://export.arxiv.org/api/query?"
    add_url = "search_query=%s&sortBy=lastUpdatedDate&start=%d&max_results=%d" % (
        search_query,
        start_index,
        max_r,
    )
    # add_url = 'search_query=%s&sortBy=submittedDate&start=%d&max_results=100' % (search_query, start_index)
    search_query = base_url + add_url
    logger.debug(f"arxiv url {search_query}")
    logger.debug(f"Searching arxiv for {search_query}")
    req = urllib.request.Request(
        search_query,
        headers={
            "User-Agent": "arxiv-sanity-x (+https://github.com/karpathy/arxiv-sanity-lite)",
        },
    )
    last_exc: Exception | None = None
    for attempt in range(_ARXIV_RETRY_MAX_TRIES):
        try:
            with urllib.request.urlopen(req, timeout=settings.arxiv.api_timeout) as url:
                response = url.read()
                if getattr(url, "status", 200) != 200:
                    logger.error("arxiv did not return status 200 response")
                return response
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_exc = e
            if attempt >= _ARXIV_RETRY_MAX_TRIES - 1:
                break
            logger.warning(f"arxiv API request failed (attempt {attempt+1}/{_ARXIV_RETRY_MAX_TRIES}): {e}")
            _sleep_backoff(attempt)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("arxiv API request failed")

    # Unreachable: kept to satisfy type checkers.
    return b""


def encode_feedparser_dict(d):
    """helper function to strip feedparser objects using a deep copy"""
    if isinstance(d, feedparser.FeedParserDict) or isinstance(d, dict):
        return {k: encode_feedparser_dict(d[k]) for k in d.keys()}
    elif isinstance(d, list):
        return [encode_feedparser_dict(k) for k in d]
    else:
        return d


def parse_arxiv_url(url):
    """
    examples is http://arxiv.org/abs/1512.08756v2
    we want to extract the raw id (1512.08756) and the version (2)
    """
    ix = url.rfind("/")
    if ix < 0:
        raise ValueError(f"bad url: {url}")
    idv = url[ix + 1 :]  # extract just the id (and the version)
    if not idv:
        raise ValueError(f"bad url (empty id): {url}")

    rawid = idv
    version = 1
    try:
        left, right = idv.rsplit("v", 1)
        if left and right.isdigit():
            rawid = left
            version = int(right)
    except ValueError:
        # No 'v' in idv - treat as v1.
        pass

    idv_norm = f"{rawid}v{version}"
    return idv_norm, rawid, version


def parse_response(response):
    out = []
    parse = feedparser.parse(response)
    # for e in tqdm.tqdm(parse.entries, desc="Parsing papers"):
    for e in parse.entries:
        j = encode_feedparser_dict(e)
        # extract / parse id information
        try:
            idv, rawid, version = parse_arxiv_url(j.get("id", ""))
        except Exception as exc:
            logger.warning("skip entry with invalid arxiv id: %s (%s)", j.get("id"), exc)
            continue
        j["_idv"] = idv
        j["_id"] = rawid
        j["_version"] = version
        updated_parsed = j.get("updated_parsed") or j.get("published_parsed")
        if not updated_parsed:
            logger.warning("skip entry missing updated/published time: %s", idv)
            continue
        j["_time"] = time.mktime(updated_parsed)
        j["_time_str"] = time.strftime("%b %d %Y", updated_parsed)
        # delete apparently spurious and redundant information
        j.pop("summary_detail", None)
        j.pop("title_detail", None)
        out.append(j)

    return out


def filter_latest_version(idvs):
    """
    for each idv filter the list down to only the most recent version
    """

    pid_to_v = OrderedDict()
    for idv in idvs:
        s = str(idv or "").strip()
        if not s:
            continue
        pid = ""
        v = 1
        if "v" not in s:
            # Accept version-less ids as v1, but only if it resembles an arXiv id.
            # arXiv ids are either new-style "YYYY.NNNNN" or old-style "archive/YYMMNNN".
            if "." not in s and "/" not in s:
                continue
            pid = s
            v = 1
        else:
            left, right = s.rsplit("v", 1)
            if left and right.isdigit():
                pid = left
                v = int(right)
            else:
                logger.warning("skip invalid idv '%s'", s)
                continue

        if not pid:
            continue

        try:
            pid_to_v[pid] = max(int(v), pid_to_v.get(pid, 0))
        except Exception as exc:
            logger.warning("skip invalid idv '%s': %s", s, exc)
            continue

    filt = [f"{pid}v{v}" for pid, v in pid_to_v.items()]
    return filt
