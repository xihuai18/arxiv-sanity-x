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
    assert ix >= 0, "bad url: " + url
    idv = url[ix + 1 :]  # extract just the id (and the version)
    parts = idv.split("v")
    assert len(parts) == 2, "error splitting id and version in idv string: " + idv
    return idv, parts[0], int(parts[1])


def parse_response(response):
    out = []
    parse = feedparser.parse(response)
    # for e in tqdm.tqdm(parse.entries, desc="Parsing papers"):
    for e in parse.entries:
        j = encode_feedparser_dict(e)
        # extract / parse id information
        idv, rawid, version = parse_arxiv_url(j["id"])
        j["_idv"] = idv
        j["_id"] = rawid
        j["_version"] = version
        j["_time"] = time.mktime(j["updated_parsed"])
        j["_time_str"] = time.strftime("%b %d %Y", j["updated_parsed"])
        # delete apparently spurious and redundant information
        del j["summary_detail"]
        del j["title_detail"]
        out.append(j)

    return out


def filter_latest_version(idvs):
    """
    for each idv filter the list down to only the most recent version
    """

    pid_to_v = OrderedDict()
    for idv in idvs:
        pid, v = idv.split("v")
        pid_to_v[pid] = max(int(v), pid_to_v.get(pid, 0))

    filt = [f"{pid}v{v}" for pid, v in pid_to_v.items()]
    return filt
