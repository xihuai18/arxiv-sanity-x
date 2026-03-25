"""
Utils for dealing with arxiv API and related processing
"""

import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict

import feedparser

from config import settings

logger = logging.getLogger(__name__)

_ARXIV_RETRY_MAX_TRIES = 3
_WITHDRAWN_BODY_MARKERS = (
    "this paper has been withdrawn",
    "this article has been withdrawn",
    "the paper has been withdrawn",
    "the article has been withdrawn",
    "withdrawn by the author",
    "withdrawn by the authors",
    "this paper has been retracted",
    "this article has been retracted",
    "retracted by the author",
    "retracted by the authors",
)


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
    return _open_arxiv_api_url(search_query)


def _open_arxiv_api_url(url: str) -> bytes:
    """Open an arXiv API URL with bounded retries."""

    logger.debug(f"arxiv url {url}")
    logger.debug(f"Searching arxiv for {url}")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "arxiv-sanity-x (+https://github.com/karpathy/arxiv-sanity-lite)",
        },
    )
    last_exc: Exception | None = None
    for attempt in range(_ARXIV_RETRY_MAX_TRIES):
        try:
            with urllib.request.urlopen(req, timeout=settings.arxiv.api_timeout) as resp:
                response = resp.read()
                if getattr(resp, "status", 200) != 200:
                    logger.error("arxiv did not return status 200 response")
                return response
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_exc = e
            if attempt >= _ARXIV_RETRY_MAX_TRIES - 1:
                break
            logger.warning(f"arxiv API request failed (attempt {attempt + 1}/{_ARXIV_RETRY_MAX_TRIES}): {e}")
            _sleep_backoff(attempt)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("arxiv API request failed")

    # Unreachable: kept to satisfy type checkers.
    return b""


def get_entries_by_ids(ids: list[str]) -> list[dict]:
    """Fetch one or more specific arXiv IDs via the public API."""

    normalized_ids = [str(pid or "").strip() for pid in ids if str(pid or "").strip()]
    if not normalized_ids:
        return []
    query = urllib.parse.urlencode({"id_list": ",".join(normalized_ids)})
    response = _open_arxiv_api_url(f"https://export.arxiv.org/api/query?{query}")
    return parse_response(response)


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


def _normalize_marker_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def is_withdrawn_entry(entry: dict | None) -> bool:
    """Best-effort withdrawn/retracted detection from arXiv API comments."""

    if not isinstance(entry, dict):
        return False

    comment = _normalize_marker_text(entry.get("arxiv_comment") or entry.get("comment"))
    return bool(comment and any(marker in comment for marker in _WITHDRAWN_BODY_MARKERS))


def resolve_latest_nonwithdrawn_version(
    latest_entry: dict | None,
    *,
    entries_getter=None,
    batch_size: int = 20,
) -> dict | None:
    """Return the latest non-withdrawn version for a paper, or None if none exists."""

    if not isinstance(latest_entry, dict):
        return None
    if not is_withdrawn_entry(latest_entry):
        return latest_entry

    raw_pid = str(latest_entry.get("_id") or "").strip()
    try:
        latest_version = int(latest_entry.get("_version") or 0)
    except Exception:
        latest_version = 0
    if not raw_pid or latest_version <= 1:
        return None

    candidate_ids = [f"{raw_pid}v{version}" for version in range(latest_version - 1, 0, -1)]
    getter = entries_getter or get_entries_by_ids
    entries_by_idv: dict[str, dict] = {}

    chunk_size = max(1, int(batch_size or 1))
    for start in range(0, len(candidate_ids), chunk_size):
        chunk = candidate_ids[start : start + chunk_size]
        for entry in getter(chunk) or []:
            if not isinstance(entry, dict):
                continue
            idv = str(entry.get("_idv") or "").strip()
            if idv:
                entries_by_idv[idv] = entry

    for candidate_id in candidate_ids:
        candidate = entries_by_idv.get(candidate_id)
        if candidate and not is_withdrawn_entry(candidate):
            return candidate
    return None


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
