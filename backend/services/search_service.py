"""Search and ranking services."""

from __future__ import annotations

import re
import time
from random import shuffle
from threading import Lock
from typing import Any

import numpy as np
from loguru import logger

from config import settings

SUMMARY_DEFAULT_SEMANTIC_WEIGHT = settings.summary.default_semantic_weight

from ..utils.cache import LRUCacheTTL

# Constants (from centralized settings)
RET_NUM = settings.search.ret_num
MAX_RESULTS = settings.search.max_results

# Query parsing patterns
TFIDF_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b|\b[a-zA-Z]+\-[a-zA-Z]+\b"
TFIDF_STOP_WORDS = "english"
_QUERY_SEP_RE = re.compile(r"[,;\uFF0C\u3001\uFF1B\uFF1A:/\\|\(\)\[\]{}]+")
_BOOLEAN_TOKENS = {"and", "or", "not"}
_ARXIV_ID_RE = re.compile(
    r"(?:(?:arxiv:)?)(?P<id>(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))(?:v(?P<v>\d+))?",
    re.IGNORECASE,
)


# Search caches
SVM_RANK_CACHE = LRUCacheTTL(maxsize=128, ttl_s=180.0)
SEARCH_RANK_CACHE = LRUCacheTTL(maxsize=256, ttl_s=120.0)
QUERY_EMBED_CACHE = LRUCacheTTL(maxsize=256, ttl_s=600.0)


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    text = (text or "").lower().strip()
    if not text:
        return ""
    text = text.replace("-", " ")
    text = _QUERY_SEP_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def normalize_text_loose(text: str) -> str:
    """Looser normalization for title-like matching."""
    s = (text or "").lower().strip()
    if not s:
        return ""
    s = re.sub(r"[\-\u2010\u2011\u2012\u2013\u2014]", " ", s)
    s = re.sub(r"[\:\;\,\.!\?\(\)\[\]\{\}\/\\\|\"\'\`\~\+\=\*\#\$\%\^\&\@]", " ", s)
    s = " ".join(s.split())
    return s


def extract_arxiv_ids(q: str) -> list[str]:
    """Return normalized arXiv IDs mentioned in the query."""
    if not q:
        return []
    ids = []
    for m in _ARXIV_ID_RE.finditer(q.strip()):
        pid = (m.group("id") or "").strip().lower()
        ver = (m.group("v") or "").strip()
        if not pid:
            continue
        ids.append(f"{pid}v{ver}" if ver else pid)
    seen = set()
    out = []
    for pid in ids:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def looks_like_cjk_query(q: str) -> bool:
    """Heuristic: query contains mostly CJK and little ASCII."""
    if not q:
        return False
    s = q.strip()
    if not s:
        return False
    cjk = 0
    ascii_alnum = 0
    for ch in s:
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF) or (0x3040 <= o <= 0x30FF) or (0xAC00 <= o <= 0xD7AF):
            cjk += 1
        elif ch.isascii() and (ch.isalnum() or ch in (".", "-", "/")):
            ascii_alnum += 1
    return cjk >= 2 and ascii_alnum <= max(2, len(s) // 12)


def parse_search_query(q: str) -> dict:
    """Parse a paper-search query with field filters and phrases."""
    raw = (q or "").strip()
    parsed = {
        "raw": raw,
        "raw_lower": raw.lower(),
        "norm": normalize_text(raw),
        "terms": [],
        "phrases": [],
        "neg_terms": [],
        "filters": {"ti": [], "au": [], "abs": [], "cat": [], "id": []},
        "filters_phrases": {"ti": [], "au": [], "abs": [], "cat": [], "id": []},
    }
    if not raw:
        return parsed

    work = raw
    for key, canon in (
        ("title", "ti"),
        ("ti", "ti"),
        ("author", "au"),
        ("au", "au"),
        ("abstract", "abs"),
        ("abs", "abs"),
        ("category", "cat"),
        ("cat", "cat"),
        ("id", "id"),
    ):
        pat = re.compile(rf"(?i)(?:^|\s){re.escape(key)}:\"([^\"]+)\"(?=\s|$)")
        while True:
            m = pat.search(work)
            if not m:
                break
            parsed["filters"][canon].append(m.group(1).strip())
            parsed["filters_phrases"][canon].append(m.group(1).strip())
            work = (work[: m.start()] + " " + work[m.end() :]).strip()

    for m in re.finditer(r"\"([^\"]+)\"", work):
        phr = m.group(1).strip()
        if phr:
            parsed["phrases"].append(phr)
    work = re.sub(r"\"([^\"]+)\"", " ", work)

    tokens = [t for t in re.split(r"\s+", work) if t]
    for tok in tokens:
        if tok.startswith("-") or tok.startswith("!"):
            val = tok[1:].strip()
            if val:
                parsed["neg_terms"].append(val)
            continue
        m = re.match(r"(?i)^(title|ti|author|au|abstract|abs|category|cat|id):(.+)$", tok)
        if m:
            k = m.group(1).lower()
            v = m.group(2).strip()
            canon = {
                "title": "ti",
                "ti": "ti",
                "author": "au",
                "au": "au",
                "abstract": "abs",
                "abs": "abs",
                "category": "cat",
                "cat": "cat",
                "id": "id",
            }[k]
            if v:
                parsed["filters"][canon].append(v)
            continue
        parsed["terms"].append(tok)

    def _split_terms(values):
        out = []
        for v in values:
            vs = normalize_text(v)
            if not vs:
                continue
            out.extend([p for p in vs.split() if p])
        return out

    parsed["filters_terms"] = {k: _split_terms(parsed["filters"][k]) for k in parsed["filters"]}
    parsed["filters_phrases_norm"] = {
        k: [normalize_text(p) for p in parsed["filters_phrases"][k] if p.strip()] for k in parsed["filters_phrases"]
    }
    parsed["terms_norm"] = [
        t for t in normalize_text(" ".join(parsed["terms"])).split() if t and t not in _BOOLEAN_TOKENS
    ]
    parsed["neg_terms_norm"] = [t for t in normalize_text(" ".join(parsed["neg_terms"])).split() if t]
    parsed["phrases_norm"] = [normalize_text(p) for p in parsed["phrases"] if p.strip()]
    return parsed


def apply_limit(pids: list[str], scores: list[float], limit: int | None) -> tuple[list[str], list[float]]:
    """Apply limit to results."""
    if limit is not None and len(pids) > limit:
        return pids[:limit], scores[:limit]
    return pids, scores


def random_rank(pids_all: list[str], limit: int | None = None) -> tuple[list[str], list[float]]:
    """Random ranking."""
    import random

    if limit is not None:
        try:
            k = min(int(limit), len(pids_all))
        except Exception:
            k = len(pids_all)
        if k <= 0:
            return [], []
        pids = random.sample(pids_all, k)
        return pids, [0.0] * len(pids)
    pids = list(pids_all)
    shuffle(pids)
    return pids, [0.0] * len(pids)


def time_rank(metas: dict[str, Any], limit: int | None = None) -> tuple[list[str], list[float]]:
    """Time-based ranking."""
    import heapq

    if limit is not None:
        try:
            k = min(int(limit), len(metas))
        except Exception:
            k = len(metas)
        if k <= 0:
            return [], []
        ms = heapq.nlargest(k, metas.items(), key=lambda kv: kv[1]["_time"])
    else:
        ms = sorted(metas.items(), key=lambda kv: kv[1]["_time"], reverse=True)
    tnow = time.time()
    pids = [k for k, v in ms]
    scores = [(tnow - v["_time"]) / 60 / 60 / 24 for k, v in ms]
    return pids, scores


def filter_by_time(
    pids: list[str], metas: dict[str, Any], time_filter: str, user_tagged_pids: set | None = None
) -> tuple[list[str], list[int]]:
    """Filter papers by time, keeping tagged papers."""
    if not time_filter:
        return pids, list(range(len(pids)))

    tnow = time.time()
    try:
        deltat = float(time_filter) * 60 * 60 * 24
    except Exception:
        logger.warning(f"Invalid time_filter '{time_filter}', skipping")
        return pids, list(range(len(pids)))

    tagged_set = user_tagged_pids or set()
    valid_indices = []
    valid_pids = []
    for i, pid in enumerate(pids):
        meta = metas.get(pid)
        if not meta:
            continue
        in_time_window = (tnow - meta["_time"]) < deltat
        is_tagged = pid in tagged_set
        if in_time_window or is_tagged:
            valid_indices.append(i)
            valid_pids.append(pid)
    return valid_pids, valid_indices


def is_title_like_query(parsed: dict) -> bool:
    """Heuristic: user pasted a paper title."""
    if not parsed:
        return False
    raw = (parsed.get("raw") or "").strip()
    if not raw:
        return False
    # Longer, multi-token queries are more likely titles.
    terms = parsed.get("terms_norm") or []
    if len(raw) >= 22 and len(terms) >= 3:
        return True
    # Punctuation typical in titles.
    if ":" in raw and len(raw) >= 16:
        return True
    return False


def title_candidate_scan(
    parsed: dict,
    *,
    get_pids_fn,
    get_papers_bulk_fn,
    paper_text_fields_fn,
    max_candidates: int = 500,
    max_scan: int = 120000,
    time_budget_s: float = 0.6,
) -> list[str]:
    """Bounded scan over titles to pull in strong title matches.

    This is a safety net for "paste full title" searches when TF-IDF recall misses
    due to tokenization/vocab/min_df effects.
    """
    q_norm = (parsed.get("norm") or "").strip()
    q_loose = normalize_text_loose(parsed.get("raw") or "")
    if not q_norm and not q_loose:
        return []

    # Prefer loose form for punctuation-insensitive matching.
    q_use = q_loose or q_norm

    import heapq

    start = time.time()
    heap: list[tuple[float, str]] = []
    all_pids = list(get_pids_fn() or [])
    if not all_pids:
        return []

    scan_n = min(len(all_pids), int(max_scan))
    chunk = 2000
    for i in range(0, scan_n, chunk):
        if time.time() - start > time_budget_s:
            break
        part = all_pids[i : i + chunk]
        pid_to_paper = get_papers_bulk_fn(part)
        for pid in part:
            p = pid_to_paper.get(pid)
            if p is None:
                continue
            fields = paper_text_fields_fn(p)
            title_loose = fields.get("title_norm_loose") or ""
            title_norm = fields.get("title_norm") or ""
            if not title_loose and not title_norm:
                continue

            s = 0.0
            # Exact loose match is very strong.
            if q_use and title_loose and q_use == title_loose:
                s = 2000.0
            elif q_use and title_loose and q_use in title_loose:
                # Substring match on loose title.
                s = 900.0 + min(200.0, 10.0 * (len(q_use) / max(1.0, len(title_loose))))
            elif q_norm and title_norm and q_norm in title_norm and len(q_norm) >= 18:
                s = 750.0
            else:
                continue

            if len(heap) < max_candidates:
                heapq.heappush(heap, (s, pid))
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (s, pid))

    heap.sort(reverse=True)
    return [pid for _, pid in heap]


def compute_paper_score_parsed(
    parsed: dict,
    paper: dict,
    pid: str,
    *,
    paper_text_fields_fn,
    get_metas_fn,
    now_ts: float | None = None,
) -> float:
    """Paper-oriented lexical scoring with optional field filters."""
    if not parsed or paper is None:
        return 0.0

    pid_lower = (pid or "").lower()
    fields = paper_text_fields_fn(paper)

    # If the user input looks like a full title, strongly boost exact/substring title matches.
    q_raw_lower = (parsed.get("raw_lower") or "").strip()
    q_norm_full = (parsed.get("norm") or "").strip()
    q_loose_full = normalize_text_loose(parsed.get("raw") or "")
    if q_norm_full and len(q_norm_full) >= 18:
        if q_norm_full == fields["title_norm"] or (
            q_loose_full and q_loose_full == (fields.get("title_norm_loose") or "")
        ):
            return 1800.0
        if (q_norm_full in fields["title_norm"]) or (
            q_loose_full and q_loose_full in (fields.get("title_norm_loose") or "")
        ):
            title_phrase_bonus = 700.0
            if q_raw_lower and q_raw_lower in fields["title_lower"]:
                title_phrase_bonus += 50.0
            score = title_phrase_bonus
        else:
            score = 0.0
    else:
        score = 0.0

    # Exact ID fast path
    for want in extract_arxiv_ids(parsed.get("raw", "")) + (parsed.get("filters") or {}).get("id", []):
        want_l = want.strip().lower()
        if not want_l:
            continue
        if want_l == pid_lower:
            return 2000.0
        if want_l in pid_lower or pid_lower in want_l:
            return 1200.0

    # Negation
    neg = parsed.get("neg_terms_norm") or []
    if neg:
        hay = " ".join((fields["title_norm"], fields["authors_norm"], fields["tags_norm"], fields["summary_norm"]))
        for t in neg:
            if t and t in hay:
                return -50.0

    f_terms = parsed.get("filters_terms") or {}
    f_phrases = parsed.get("filters_phrases_norm") or {}
    general = parsed.get("terms_norm") or []
    phrases = parsed.get("phrases_norm") or []

    has_field_filters = any(
        len((parsed.get("filters", {}) or {}).get(k, []) or []) > 0 for k in ("ti", "au", "abs", "cat")
    )

    title_terms = (f_terms.get("ti") or []) + ([] if has_field_filters else general)
    author_terms = (f_terms.get("au") or []) + ([] if has_field_filters else general)
    abs_terms = (f_terms.get("abs") or []) + ([] if has_field_filters else general)
    cat_terms = (f_terms.get("cat") or []) + ([] if has_field_filters else general)

    # Enforce explicit field filters
    if f_phrases.get("ti"):
        if any(ph and ph in fields["title_norm"] for ph in f_phrases["ti"]):
            score += 180.0
        elif f_terms.get("ti"):
            if not any(t and t in fields["title_norm"] for t in f_terms["ti"]):
                return 0.0
        else:
            return 0.0
    elif f_terms.get("ti"):
        if not any(t and t in fields["title_norm"] for t in f_terms["ti"]):
            return 0.0

    if f_phrases.get("au"):
        if any(ph and ph in fields["authors_norm"] for ph in f_phrases["au"]):
            score += 120.0
        elif f_terms.get("au"):
            if not any(t and t in fields["authors_norm"] for t in f_terms["au"]):
                return 0.0
        else:
            return 0.0
    elif f_terms.get("au"):
        if not any(t and t in fields["authors_norm"] for t in f_terms["au"]):
            return 0.0

    if f_phrases.get("abs"):
        if any(ph and ph in fields["summary_norm"] for ph in f_phrases["abs"]):
            score += 60.0
        elif f_terms.get("abs"):
            if not any(t and t in fields["summary_norm"] for t in f_terms["abs"]):
                return 0.0
        else:
            return 0.0
    elif f_terms.get("abs"):
        if not any(t and t in fields["summary_norm"] for t in f_terms["abs"]):
            return 0.0

    if f_phrases.get("cat"):
        if any(ph and ph in fields["tags_norm"] for ph in f_phrases["cat"]):
            score += 45.0
        elif f_terms.get("cat"):
            if not any(t and t in fields["tags_norm"] for t in f_terms["cat"]):
                return 0.0
        else:
            return 0.0
    elif f_terms.get("cat"):
        if not any(t and t in fields["tags_norm"] for t in f_terms["cat"]):
            return 0.0

    # Phrases: strongest in title, then abstract
    for ph in phrases:
        if not ph:
            continue
        if ph in fields["title_norm"]:
            score += 140.0
        elif ph in fields["summary_norm"]:
            score += 25.0

    # Implicit phrase boost
    if not phrases and general and len(general) >= 2 and not has_field_filters:
        general_phrase = " ".join(general)
        if general_phrase and general_phrase in fields["title_norm"]:
            score += 45.0
        elif general_phrase and general_phrase in fields["summary_norm"]:
            score += 12.0

    # Title scoring
    if title_terms:
        hits = sum(1 for t in title_terms if t and (t in fields["title_norm"]))
        if hits:
            score += 70.0 * (hits / max(1, len(title_terms)))
            freq = sum(min(2, fields["title_lower"].count(t)) for t in title_terms if t) / max(1, len(title_terms))
            score += 12.0 * freq
            if hits == len(title_terms) and len(title_terms) >= 2:
                score += 40.0

    # Author scoring
    if author_terms:
        hits = sum(1 for t in author_terms if t and (t in fields["authors_norm"]))
        if hits:
            score += 55.0 * (hits / max(1, len(author_terms)))

    # Category/tag scoring
    if cat_terms:
        hits = sum(1 for t in cat_terms if t and (t in fields["tags_norm"]))
        if hits:
            score += 35.0 * (hits / max(1, len(cat_terms)))

    # Abstract scoring (lower weight)
    if abs_terms:
        hits = sum(1 for t in abs_terms if t and (t in fields["summary_norm"]))
        if hits:
            score += 12.0 * (hits / max(1, len(abs_terms)))

    # Tiny recency tie-breaker
    try:
        metas = get_metas_fn() if get_metas_fn else {}
        meta = metas.get(pid)
        if meta and meta.get("_time"):
            now = float(now_ts) if now_ts is not None else time.time()
            age_days = max(0.0, (now - float(meta["_time"])) / 86400.0)
            score += 0.35 * (1.0 / (1.0 + age_days / 365.0))
    except Exception:
        pass

    # Boost coverage for multi-keyword queries
    if general and len(general) > 1 and not has_field_filters:
        hay = " ".join((fields["title_norm"], fields["authors_norm"], fields["tags_norm"], fields["summary_norm"]))
        matched = {t for t in general if t and t in hay}
        if matched:
            coverage = len(matched) / max(1, len(general))
            score += 18.0 * coverage
            if coverage == 1.0:
                score += 24.0

    return score


def lexical_rank_over_pids(
    pids: list[str],
    parsed: dict,
    *,
    get_papers_bulk_fn,
    paper_text_fields_fn,
    get_metas_fn,
    apply_limit_fn,
    limit: int | None = None,
) -> tuple[list[str], list[float]]:
    """Score a provided candidate pid list using lexical scoring."""
    if not pids:
        return [], []
    try:
        k = int(limit) if limit is not None else None
    except Exception:
        k = None

    pid_to_paper = get_papers_bulk_fn(pids)
    pairs: list[tuple[float, str]] = []
    now = time.time()
    for pid in pids:
        p = pid_to_paper.get(pid)
        if p is None:
            continue
        s = compute_paper_score_parsed(
            parsed,
            p,
            pid,
            paper_text_fields_fn=paper_text_fields_fn,
            get_metas_fn=get_metas_fn,
            now_ts=now,
        )
        if s > 0:
            pairs.append((s, pid))
    pairs.sort(reverse=True)
    out_pids = [pid for _, pid in pairs]
    out_scores = [float(s) for s, _ in pairs]
    if k is not None:
        out_pids, out_scores = apply_limit_fn(out_pids, out_scores, k)
    return out_pids, out_scores


def lexical_rank_fullscan(
    parsed: dict,
    *,
    get_pids_fn,
    get_papers_fn,
    get_papers_bulk_fn,
    paper_text_fields_fn,
    get_metas_fn,
    max_results: int,
    limit: int | None = None,
) -> tuple[list[str], list[float]]:
    """Lexical ranking by scanning the whole corpus."""
    all_pids = list(get_pids_fn() or [])
    if not all_pids:
        return [], []

    try:
        k = int(limit) if limit is not None else 200
    except Exception:
        k = 200
    k = max(1, min(k, int(max_results)))

    import heapq

    heap: list[tuple[float, str]] = []
    now = time.time()
    papers_cache = get_papers_fn()
    if isinstance(papers_cache, dict) and papers_cache:
        for pid in all_pids:
            p = papers_cache.get(pid)
            if p is None:
                continue
            s = compute_paper_score_parsed(
                parsed,
                p,
                pid,
                paper_text_fields_fn=paper_text_fields_fn,
                get_metas_fn=get_metas_fn,
                now_ts=now,
            )
            if s <= 0:
                continue
            if len(heap) < k:
                heapq.heappush(heap, (s, pid))
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (s, pid))
    else:
        chunk = 2000
        for i in range(0, len(all_pids), chunk):
            part = all_pids[i : i + chunk]
            pid_to_paper = get_papers_bulk_fn(part)
            for pid in part:
                p = pid_to_paper.get(pid)
                if p is None:
                    continue
                s = compute_paper_score_parsed(
                    parsed,
                    p,
                    pid,
                    paper_text_fields_fn=paper_text_fields_fn,
                    get_metas_fn=get_metas_fn,
                    now_ts=now,
                )
                if s <= 0:
                    continue
                if len(heap) < k:
                    heapq.heappush(heap, (s, pid))
                else:
                    if s > heap[0][0]:
                        heapq.heapreplace(heap, (s, pid))

    heap.sort(reverse=True)
    out_pids = [pid for _, pid in heap]
    out_scores = [float(s) for s, _ in heap]
    return out_pids, out_scores


def compute_paper_score_simple(q: str, qs: list[str], q_norm: str, qs_norm: list[str], paper: dict, pid: str) -> float:
    """Compute a simple relevance score for a single paper.

    This is the pre-refactor scoring function used by legacy_search_rank().
    Kept for backwards compatibility.
    """
    score = 0.0

    pid_lower = (pid or "").lower()
    if q == pid_lower:
        return 1000.0
    if q and (q in pid_lower or pid_lower in q):
        score += 500.0

    title = paper.get("title", "") or ""
    title_lower = title.lower()
    title_norm = normalize_text(title)

    if q and q in title_lower:
        score += 100.0
    elif q_norm and q_norm in title_norm:
        score += 90.0

    if len(qs) > 1:
        if all(qp in title_lower for qp in qs):
            score += 60.0
        elif all(qp in title_norm for qp in qs_norm):
            score += 50.0

    words_in_title = sum(1 for qp in qs if qp and qp in title_lower)
    words_in_title_norm = sum(1 for qp in qs_norm if qp and qp in title_norm)
    best_partial = max(words_in_title, words_in_title_norm)
    if best_partial > 0 and qs:
        partial_ratio = best_partial / len(qs)
        score += 30.0 * partial_ratio
        freq_score = sum(min(2, title_lower.count(qp)) for qp in qs if qp) / len(qs)
        score += 10.0 * freq_score

    authors = paper.get("authors") or []
    authors_str = " ".join(a.get("name", "") for a in authors if isinstance(a, dict)).lower()
    if q and q in authors_str:
        score += 80.0
    elif len(qs) > 1 and all(qp in authors_str for qp in qs):
        score += 50.0
    else:
        author_matches = sum(1 for qp in qs if qp and qp in authors_str)
        if author_matches > 0 and qs:
            score += 20.0 * (author_matches / len(qs))

    summary_text = paper.get("summary", "") or ""
    summary_lower = summary_text.lower()
    summary_norm = normalize_text(summary_text)

    if q and q in summary_lower:
        score += 15.0
    elif q_norm and q_norm in summary_norm:
        score += 12.0

    if len(qs) > 1:
        if all(qp in summary_lower for qp in qs):
            score += 8.0
        elif all(qp in summary_norm for qp in qs_norm):
            score += 6.0

    words_in_summary = sum(1 for qp in qs if qp and qp in summary_lower)
    if words_in_summary > 0 and qs:
        score += 3.0 * (words_in_summary / len(qs))
        freq_score = sum(min(3, summary_lower.count(qp)) for qp in qs if qp) / len(qs)
        score += 1.0 * freq_score

    return score


def count_match(q: str, pid_start: int, n_pids: int):
    """Count-based matching for legacy search (parallelizable worker)."""
    q = (q or "").strip()
    if not q:
        return []

    q_lower = q.lower()
    qs = q_lower.split()
    q_norm = normalize_text(q)
    qs_norm = q_norm.split()

    from aslite.repositories import PaperRepository
    from backend.services.data_service import get_pids as _get_pids

    sub_pairs = []
    with PaperRepository.open_readonly() as pdb:
        pids = _get_pids()[pid_start : pid_start + n_pids]
        chunk_size = 500
        for i in range(0, len(pids), chunk_size):
            batch = pids[i : i + chunk_size]
            batch_papers = pdb.get_many(batch)
            for pid in batch:
                paper = batch_papers.get(pid)
                if paper is None:
                    continue

                score = compute_paper_score_simple(q_lower, qs, q_norm, qs_norm, paper, pid)
                if score > 0:
                    sub_pairs.append((score, pid))

    return sub_pairs


def legacy_search_rank(q: str = "", limit: int | None = None):
    """Legacy count-based keyword search.

    This is slow and kept only for compatibility/testing; modern keyword search
    is handled by search_rank() in legacy (TF-IDF + lexical fusion).
    """
    q = (q or "").strip()
    if not q:
        return [], []

    from multiprocessing import Pool, cpu_count

    from backend.services.data_service import get_pids as _get_pids

    n_pids = len(_get_pids())
    chunk_size = 20000
    n_process = min(cpu_count() // 2, max(1, n_pids // chunk_size))
    n_process = max(1, n_process)

    with Pool(n_process) as pool:
        sub_pairs_list = pool.starmap(
            count_match,
            [(q, pid_start, chunk_size) for pid_start in range(0, n_pids, chunk_size)],
        )
        pairs = [pair for sublist in sub_pairs_list for pair in sublist]

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    scores = [float(p[0]) for p in pairs]
    return apply_limit(pids, scores, limit)


# -----------------------------------------------------------------------------
# SVM-based ranking
# -----------------------------------------------------------------------------


def svm_rank(
    tags: str = "",
    s_pids: str = "",
    C: float = None,
    logic: str = "and",
    time_filter: str = "",
    limit: int | None = None,
    *,
    # Dependency injection for testability
    get_features_fn=None,
    get_tags_fn=None,
    get_neg_tags_fn=None,
    get_metas_fn=None,
    compute_upload_features_fn=None,
    user: str | None = None,
) -> tuple[list[str], list[float], list[dict]]:
    """SVM-based paper ranking using user tags as training signal.

    Args:
        tags: Comma-separated tag names or 'all' for all tags
        s_pids: Comma-separated paper IDs to use as positive seeds
        C: SVM regularization parameter
        logic: 'and' or 'or' for combining multiple tags
        time_filter: Days to filter papers by time
        limit: Maximum number of results
        get_features_fn: Function to get features (for testing)
        get_tags_fn: Function to get user tags (for testing)
        get_neg_tags_fn: Function to get negative tags (for testing)
        get_metas_fn: Function to get paper metas (for testing)
        user: User identifier for cache key

    Returns:
        Tuple of (pids, scores, words) where words contains top SVM feature weights
    """
    import os

    from sklearn import svm as sklearn_svm

    from aslite.db import DICT_DB_FILE, FEATURES_FILE
    from config import settings

    SVM_C = settings.svm.c
    SVM_MAX_ITER = settings.svm.max_iter
    SVM_TOL = settings.svm.tol
    SVM_NEG_WEIGHT = settings.svm.neg_weight

    # Default dependency injection
    if get_features_fn is None:
        from backend.services.data_service import get_features_cached

        get_features_fn = get_features_cached
    if get_tags_fn is None:
        from backend.services.user_service import get_tags

        get_tags_fn = get_tags
    if get_neg_tags_fn is None:
        from backend.services.user_service import get_neg_tags

        get_neg_tags_fn = get_neg_tags
    if get_metas_fn is None:
        from backend.services.data_service import get_metas

        get_metas_fn = get_metas

    # Use default value
    if C is None:
        C = SVM_C

    # tag can be one tag or a few comma-separated tags or 'all' for all tags we have in db
    # pid can be a specific paper id to set as positive for a kind of nearest neighbor search
    if not (tags or s_pids):
        return [], [], []

    # Preload tag db for caching and smart time-filter logic
    tags_db = {}
    neg_tags_db = {}
    if tags:
        try:
            tags_db = get_tags_fn() or {}
        except Exception:
            tags_db = {}
        try:
            neg_tags_db = get_neg_tags_fn() or {}
        except Exception:
            neg_tags_db = {}

    # Fast path: cache (extend key when upload training samples are involved)
    cache_key = None
    try:
        dict_mtime = os.path.getmtime(DICT_DB_FILE) if os.path.exists(DICT_DB_FILE) else 0.0
        feat_mtime = os.path.getmtime(FEATURES_FILE) if os.path.exists(FEATURES_FILE) else 0.0

        upload_pids_in_signal = set()
        if tags:
            tags_filter_to = tags_db.keys() if tags == "all" else set(map(str.strip, tags.split(",")))
            for tag in tags_filter_to:
                for pid in tags_db.get(tag, set()) or ():
                    if isinstance(pid, str) and pid.startswith("up_"):
                        upload_pids_in_signal.add(pid)
                for pid in neg_tags_db.get(tag, set()) or ():
                    if isinstance(pid, str) and pid.startswith("up_"):
                        upload_pids_in_signal.add(pid)
        if s_pids:
            for pid in map(str.strip, s_pids.split(",")):
                if pid.startswith("up_"):
                    upload_pids_in_signal.add(pid)

        upload_fingerprint = None
        if upload_pids_in_signal:
            # If any upload feature file is missing, skip caching to avoid staleness.
            from hashlib import sha1

            try:
                from backend.services.upload_similarity_service import (
                    get_upload_features_path,
                )
            except Exception:
                get_upload_features_path = None

            missing = False
            parts = []
            if get_upload_features_path is not None:
                for upid in sorted(upload_pids_in_signal):
                    try:
                        fp = get_upload_features_path(upid)
                        if not fp.exists():
                            missing = True
                            break
                        parts.append(f"{upid}:{fp.stat().st_mtime_ns}")
                    except Exception:
                        missing = True
                        break
            else:
                missing = True

            if not missing:
                h = sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
                upload_fingerprint = (len(upload_pids_in_signal), h)

        if upload_pids_in_signal and upload_fingerprint is None:
            cache_key = None
        else:
            cache_key = (
                "svm",
                user,
                tags,
                s_pids,
                float(C),
                logic,
                time_filter,
                int(limit) if limit is not None else None,
                dict_mtime,
                feat_mtime,
                upload_fingerprint,
            )
            cached = SVM_RANK_CACHE.get(cache_key)
            if cached is not None:
                return cached
    except Exception:
        cache_key = None

    # Use intelligent cache to load features
    s_time = time.time()
    features = get_features_fn()
    x, pids = features["x"], features["pids"]

    # Collect all user-tagged paper IDs for smart time filtering
    user_tagged_pids = set()
    if tags:
        tags_filter_to = tags_db.keys() if tags == "all" else set(map(str.strip, tags.split(",")))
        for tag in tags_filter_to:
            if tag in tags_db:
                user_tagged_pids.update(tags_db[tag])
            if tag in neg_tags_db:
                user_tagged_pids.update(neg_tags_db[tag])

    if s_pids:
        user_tagged_pids.update(map(str.strip, s_pids.split(",")))

    # Apply smart time filtering: keep tagged papers and papers within time window
    if time_filter:
        metas = get_metas_fn()
        pids, time_valid_indices = filter_by_time(pids, metas, time_filter, user_tagged_pids)
        if not pids:
            return [], [], []
        x = x[time_valid_indices]
        logger.trace(
            f"After intelligent time filtering, kept {len(pids)} papers (including tagged papers and those within {time_filter} days)"
        )

    n, _d = x.shape

    # Construct the positive/negative sets (avoid building pid->idx for all papers)
    pos_weights = {}
    neg_weights = {}
    tags_filter_to = []

    if tags:
        tags_filter_to = list(tags_db.keys()) if tags == "all" else [t.strip() for t in tags.split(",") if t.strip()]
        if logic == "and":
            tag_counts = {}
            for tag in tags_filter_to:
                if tag not in tags_db:
                    continue
                for pid in tags_db[tag]:
                    tag_counts[pid] = tag_counts.get(pid, 0) + 1
            for pid, count in tag_counts.items():
                pos_weights[pid] = max(pos_weights.get(pid, 0.0), float(count))
        else:
            for tag in tags_filter_to:
                if tag not in tags_db:
                    continue
                for pid in tags_db[tag]:
                    pos_weights[pid] = max(pos_weights.get(pid, 0.0), 1.0)

        # Collect explicit negative samples for these tags
        for tag in tags_filter_to:
            if tag not in neg_tags_db:
                continue
            for pid in neg_tags_db[tag]:
                neg_weights[pid] = max(neg_weights.get(pid, 0.0), 1.0)

    if s_pids:
        pid_list = [p.strip() for p in s_pids.split(",") if p.strip()]
        seed_weight = 1.0
        if logic == "and" and tags_filter_to:
            seed_weight = float(len(tags_filter_to))
        for pid in pid_list:
            pos_weights[pid] = max(pos_weights.get(pid, 0.0), seed_weight)

    y = np.zeros(n, dtype=np.int8)
    sample_weight = np.ones(n, dtype=np.float32)
    found = 0
    if pos_weights:
        for i, pid in enumerate(pids):
            w = pos_weights.get(pid)
            if w:
                y[i] = 1
                sample_weight[i] = float(w)
                found += 1
    neg_found = 0
    if neg_weights:
        for i, pid in enumerate(pids):
            if y[i] == 1:
                continue
            w = neg_weights.get(pid)
            if w:
                sample_weight[i] = max(sample_weight[i], float(SVM_NEG_WEIGHT))
                neg_found += 1
    logger.trace(f"Found {found} positive and {neg_found} negative papers in current feature slice")
    e_time = time.time()

    logger.trace(f"feature loading/caching for {e_time - s_time:.5f}s")

    # Collect upload training samples (used for training only; candidates remain arXiv pids)
    upload_pids = set()
    if pos_weights:
        upload_pids.update([pid for pid in pos_weights.keys() if isinstance(pid, str) and pid.startswith("up_")])
    if neg_weights:
        upload_pids.update([pid for pid in neg_weights.keys() if isinstance(pid, str) and pid.startswith("up_")])

    upload_rows = []
    upload_y = []
    upload_sw = []
    upload_pos_found = 0
    if upload_pids:
        logger.info(f"SVM training includes {len(upload_pids)} uploaded paper(s): {sorted(upload_pids)}")
        if compute_upload_features_fn is None:
            from backend.services.upload_similarity_service import (
                compute_upload_features as compute_upload_features_fn,
            )

        try:
            import scipy.sparse as sp
        except Exception:
            sp = None

        if sp is not None:
            for upid in sorted(upload_pids):
                is_pos = upid in pos_weights
                is_neg = upid in neg_weights and not is_pos
                if not (is_pos or is_neg):
                    continue

                try:
                    feats = compute_upload_features_fn(upid)
                except Exception as e:
                    logger.warning(f"Failed to compute upload features for {upid}: {e}")
                    continue
                if not feats:
                    continue
                x_up = feats.get("x")
                if x_up is None:
                    continue

                try:
                    if not sp.issparse(x_up):
                        x_up = sp.csr_matrix(x_up)
                    if x_up.shape[0] != 1:
                        x_up = x_up[:1]
                    if x_up.shape[1] != x.shape[1]:
                        logger.warning(
                            f"Upload features dim mismatch for {upid}: got {x_up.shape[1]}, expected {x.shape[1]}"
                        )
                        continue
                except Exception:
                    continue

                upload_rows.append(x_up)
                if is_pos:
                    upload_y.append(1)
                    upload_sw.append(float(pos_weights.get(upid) or 1.0))
                    upload_pos_found += 1
                else:
                    upload_y.append(0)
                    upload_sw.append(float(max(1.0, float(SVM_NEG_WEIGHT))))

    if upload_rows:
        logger.info(
            f"Successfully loaded {len(upload_rows)} upload feature(s) for SVM training ({upload_pos_found} positive)"
        )

    if found + upload_pos_found == 0:
        return [], [], []  # there are no positives (even with uploads)

    if found == n:
        # If uploads introduce negative samples, we must train to leverage them.
        has_upload_negative = any(int(v) == 0 for v in (upload_y or []))
        if not has_upload_negative:
            scores = sample_weight
            if limit is not None:
                k = min(int(limit), len(scores))
                if k <= 0:
                    return [], [], []
                top_ix = np.argpartition(-scores, k - 1)[:k]
                top_ix = top_ix[np.argsort(-scores[top_ix])]
            else:
                top_ix = np.argsort(-scores)
            pids_out = [pids[int(ix)] for ix in top_ix]
            scores_out = [100 * float(scores[int(ix)]) for ix in top_ix]
            return pids_out, scores_out, []

    s_time = time.time()

    # If training data contains only one class, fallback to score-by-weight.
    try:
        if upload_rows:
            y_train_preview = np.concatenate([y, np.asarray(upload_y, dtype=np.int8)])
            if y_train_preview.min() == y_train_preview.max():
                scores = sample_weight
                if limit is not None:
                    k = min(int(limit), len(scores))
                    if k <= 0:
                        return [], [], []
                    top_ix = np.argpartition(-scores, k - 1)[:k]
                    top_ix = top_ix[np.argsort(-scores[top_ix])]
                else:
                    top_ix = np.argsort(-scores)
                pids_out = [pids[int(ix)] for ix in top_ix]
                scores_out = [100 * float(scores[int(ix)]) for ix in top_ix]
                return pids_out, scores_out, []
    except Exception:
        pass

    # classify - optimize parameters to accelerate training while maintaining accuracy
    clf = sklearn_svm.LinearSVC(
        class_weight="balanced",
        verbose=0,
        max_iter=SVM_MAX_ITER,
        tol=SVM_TOL,
        C=C,
        dual=False,  # Faster for high-dimensional features
        random_state=42,  # Ensure reproducible results
        fit_intercept=True,  # Usually improves accuracy
        multi_class="ovr",  # One-vs-rest, faster for binary classification
    )

    x_train = x
    y_train = y
    sample_weight_train = sample_weight
    if upload_rows:
        try:
            import scipy.sparse as sp

            x_train = sp.vstack([x] + upload_rows, format="csr")
            y_train = np.concatenate([y, np.asarray(upload_y, dtype=np.int8)])
            sample_weight_train = np.concatenate([sample_weight, np.asarray(upload_sw, dtype=np.float32)])
        except Exception as e:
            logger.warning(f"Failed to append upload training samples, fallback to in-slice only: {e}")
            x_train = x
            y_train = y
            sample_weight_train = sample_weight

    clf.fit(x_train, y_train, sample_weight=sample_weight_train)
    e_time = time.time()
    logger.trace(f"SVM fitting data for {e_time - s_time:.5f}s")

    s = clf.decision_function(x)
    if getattr(s, "ndim", 1) > 1:
        s = s.reshape(-1)
    e_time = time.time()
    logger.trace(f"SVM decsion function for {e_time - s_time:.5f}s")

    # Top-k selection without full sort (limit is already bounded by MAX_RESULTS in callers)
    if limit is not None:
        k = min(int(limit), len(s))
        if k <= 0:
            return [], [], []
        top_ix = np.argpartition(-s, k - 1)[:k]
        top_ix = top_ix[np.argsort(-s[top_ix])]
    else:
        top_ix = np.argsort(-s)

    pids_out = [pids[int(ix)] for ix in top_ix]
    scores_out = [100 * float(s[int(ix)]) for ix in top_ix]

    # get the words that score most positively and most negatively for the svm
    ivocab = {v: k for k, v in features["vocab"].items()}  # index to word mapping
    weights = clf.coef_[0] if getattr(clf.coef_, "ndim", 1) > 1 else clf.coef_

    # Only analyze TF-IDF part weights (vocab size), ignore embedding dimensions
    vocab_size = len(ivocab)
    tfidf_weights = weights[:vocab_size]  # Only TF-IDF weights

    sortix = np.argsort(-tfidf_weights)
    e_time = time.time()
    logger.trace(f"rank calculation for {e_time - s_time:.5f}s")

    words = []
    for ix in list(sortix[:40]) + list(sortix[-20:]):
        words.append(
            {
                "word": ivocab[ix],
                "weight": float(tfidf_weights[ix]),
            }
        )

    result = (pids_out, scores_out, words)
    if cache_key is not None:
        try:
            SVM_RANK_CACHE.set(cache_key, result)
        except Exception:
            pass
    return result


# -----------------------------------------------------------------------------
# TF-IDF Query Vectorizer
# -----------------------------------------------------------------------------

# Query-side TF-IDF vectorizer cache (reconstructed from features.p)
_tfidf_query_vectorizer = None
_tfidf_query_vectorizer_mtime = 0.0
_tfidf_query_vectorizer_lock = Lock()


def get_query_vectorizer(
    features: dict,
    features_file: str = None,
):
    """Reconstruct a query-side TF-IDF encoder from cached features.

    Args:
        features: Features dict containing vocab, idf, tfidf_params
        features_file: Path to features file for mtime checking

    Returns:
        Query vectorizer object or None if not available
    """
    global _tfidf_query_vectorizer, _tfidf_query_vectorizer_mtime

    import os

    from aslite.db import FEATURES_FILE

    if features_file is None:
        features_file = FEATURES_FILE

    try:
        current_mtime = os.path.getmtime(features_file)
    except Exception:
        current_mtime = 0.0

    with _tfidf_query_vectorizer_lock:
        if _tfidf_query_vectorizer is not None and _tfidf_query_vectorizer_mtime == current_mtime:
            return _tfidf_query_vectorizer

        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

        tfidf_params = features.get("tfidf_params", {})
        ngram_max = int(tfidf_params.get("ngram_max", 1))
        token_pattern = tfidf_params.get("token_pattern") or TFIDF_TOKEN_PATTERN
        stop_words = tfidf_params.get("stop_words", TFIDF_STOP_WORDS)
        if isinstance(stop_words, str) and stop_words.lower() in ("", "none", "null"):
            stop_words = None
        vocab = features.get("vocab")
        idf = features.get("idf")

        if vocab is None or idf is None:
            return None

        try:
            import scipy.sparse as sp
        except Exception as e:
            logger.warning(f"scipy is required for TF-IDF keyword search, fallback to legacy: {e}")
            return None

        class _QueryTfidf:
            def __init__(self, vocabulary, idf_values, ngram_max_val):
                cv_kwargs = {
                    "vocabulary": vocabulary,
                    "ngram_range": (1, ngram_max_val),
                    "dtype": np.int32,
                    "lowercase": True,
                    "strip_accents": "unicode",
                }
                if token_pattern:
                    cv_kwargs["token_pattern"] = token_pattern
                if stop_words:
                    cv_kwargs["stop_words"] = stop_words
                self._cv = CountVectorizer(**cv_kwargs)
                self._tfidf = TfidfTransformer(
                    norm="l2",
                    use_idf=True,
                    smooth_idf=True,
                    sublinear_tf=True,
                )
                idf_arr = np.asarray(idf_values, dtype=np.float32)
                self._tfidf.idf_ = idf_arr
                self._tfidf._idf_diag = sp.diags(
                    idf_arr,
                    offsets=0,
                    shape=(idf_arr.shape[0], idf_arr.shape[0]),
                    format="csr",
                )

            def transform(self, texts):
                counts = self._cv.transform(texts)
                return self._tfidf.transform(counts).astype(np.float32, copy=False)

        _tfidf_query_vectorizer = _QueryTfidf(vocab, idf, ngram_max)
        _tfidf_query_vectorizer_mtime = current_mtime
        return _tfidf_query_vectorizer


# -----------------------------------------------------------------------------
# High-level Search Functions
# -----------------------------------------------------------------------------


def search_rank(
    q: str = "",
    limit: int | None = None,
    *,
    get_features_fn=None,
    get_pids_fn=None,
    get_papers_bulk_fn=None,
    get_metas_fn=None,
    paper_exists_fn=None,
    paper_text_fields_fn=None,
) -> tuple[list[str], list[float]]:
    """
    Fast keyword search using TF-IDF dot-product against the precomputed matrix.
    Also runs legacy substring matching and merges results to catch papers
    not in the TF-IDF index (e.g., newly added papers).

    Args:
        q: Search query
        limit: Maximum number of results
        get_features_fn: Function to get features (for dependency injection)
        get_pids_fn: Function to get paper IDs
        get_papers_bulk_fn: Function to get papers in bulk
        get_metas_fn: Function to get paper metas
        paper_exists_fn: Function to check if paper exists
        paper_text_fields_fn: Function to build paper text fields

    Returns:
        Tuple of (pids, scores)
    """
    from tools.paper_summarizer import split_pid_version

    # Default dependency injection
    if get_features_fn is None:
        from backend.services.data_service import get_features_cached

        get_features_fn = get_features_cached
    if get_pids_fn is None:
        from backend.services.data_service import get_pids

        get_pids_fn = get_pids
    if get_papers_bulk_fn is None:
        from backend.services.data_service import get_papers_bulk

        get_papers_bulk_fn = get_papers_bulk
    if get_metas_fn is None:
        from backend.services.data_service import get_metas

        get_metas_fn = get_metas
    if paper_exists_fn is None:

        def paper_exists_fn(pid):
            metas = get_metas_fn()
            return pid in metas

    if paper_text_fields_fn is None:
        from backend.services.render_service import build_paper_text_fields

        paper_text_fields_fn = build_paper_text_fields

    q = (q or "").strip()
    if not q:
        return [], []

    parsed = parse_search_query(q)

    # If the query contains an explicit arXiv id, return it first (if present).
    mentioned_ids = extract_arxiv_ids(parsed.get("raw", ""))
    if mentioned_ids:
        for mid in mentioned_ids:
            raw_pid, _ = split_pid_version(mid)
            if paper_exists_fn(raw_pid):
                pid = raw_pid
                if mid != raw_pid and paper_exists_fn(mid):
                    pid = mid
                return [pid], [2000.0]

    # Fielded queries or phrases typically need lexical scoring
    has_field_filters = any(
        len(parsed.get("filters", {}).get(k, []) or []) > 0 for k in ("ti", "au", "abs", "cat", "id")
    )
    has_phrases = bool(parsed.get("phrases"))
    is_title_like = is_title_like_query(parsed)

    features = None
    if not has_field_filters:
        try:
            features = get_features_fn()
        except Exception:
            features = None

    cache_key = None
    try:
        from backend.services.data_service import get_features_file_mtime

        cache_key = (
            "kw_merged",
            q.lower(),
            int(limit) if limit is not None else None,
            float(get_features_file_mtime() or 0.0),
        )
        cached = SEARCH_RANK_CACHE.get(cache_key)
        if cached is not None:
            return cached
    except Exception:
        cache_key = None

    # If query is explicitly fielded, prefer lexical scan
    if has_field_filters or has_phrases:
        pids, scores = lexical_rank_fullscan(
            parsed,
            get_pids_fn=get_pids_fn,
            get_papers_fn=None,
            get_papers_bulk_fn=get_papers_bulk_fn,
            paper_text_fields_fn=paper_text_fields_fn,
            get_metas_fn=get_metas_fn,
            max_results=MAX_RESULTS,
            limit=limit,
        )
        if cache_key is not None:
            SEARCH_RANK_CACHE.set(cache_key, (pids, scores))
        return pids, scores

    tfidf_pids = []

    # Try TF-IDF candidate generation first
    if features:
        x_tfidf = features.get("x_tfidf")
        v = get_query_vectorizer(features) if x_tfidf is not None else None

        if x_tfidf is not None and v is not None:
            try:
                q_vec = v.transform([q])
                if q_vec.nnz > 0:
                    scores_sparse = (x_tfidf @ q_vec.T).tocoo()
                    if scores_sparse.nnz > 0:
                        rows = scores_sparse.row
                        vals = scores_sparse.data
                        cand_k = 2000
                        if limit is not None:
                            try:
                                cand_k = max(cand_k, int(limit) * 20)
                            except Exception:
                                pass
                        cand_k = min(cand_k, MAX_RESULTS)

                        if scores_sparse.nnz > cand_k:
                            top_ix = np.argpartition(-vals, cand_k - 1)[:cand_k]
                            top_ix = top_ix[np.argsort(-vals[top_ix])]
                        else:
                            top_ix = np.argsort(-vals)

                        tfidf_pids = [features["pids"][int(rows[i])] for i in top_ix]
            except Exception as e:
                logger.warning(f"TF-IDF keyword search failed: {e}")

    # If TF-IDF found too little, lexical scan fallback
    if (not tfidf_pids or len(tfidf_pids) < 30) or looks_like_cjk_query(parsed.get("raw", "")):
        pids, scores = lexical_rank_fullscan(
            parsed,
            get_pids_fn=get_pids_fn,
            get_papers_fn=None,
            get_papers_bulk_fn=get_papers_bulk_fn,
            paper_text_fields_fn=paper_text_fields_fn,
            get_metas_fn=get_metas_fn,
            max_results=MAX_RESULTS,
            limit=limit,
        )
        if cache_key is not None:
            SEARCH_RANK_CACHE.set(cache_key, (pids, scores))
        return pids, scores

    # Safety net for pasted full titles
    if is_title_like:
        extra = title_candidate_scan(
            parsed,
            get_pids_fn=get_pids_fn,
            get_papers_bulk_fn=get_papers_bulk_fn,
            paper_text_fields_fn=paper_text_fields_fn,
            max_candidates=400,
        )
        if extra:
            seen = set(tfidf_pids)
            for pid in extra:
                if pid not in seen:
                    tfidf_pids.append(pid)
                    seen.add(pid)

    # Rerank TF-IDF candidates using paper-oriented lexical scoring
    lex_pids, _ = lexical_rank_over_pids(
        list(tfidf_pids or []),
        parsed,
        get_papers_bulk_fn=get_papers_bulk_fn,
        paper_text_fields_fn=paper_text_fields_fn,
        get_metas_fn=get_metas_fn,
        apply_limit_fn=apply_limit,
        limit=None,
    )

    # Rank fusion (RRF)
    rrf_k = 60
    tfidf_rank = {pid: r for r, pid in enumerate(tfidf_pids)}
    lex_rank = {pid: r for r, pid in enumerate(lex_pids)}

    w_tfidf = 0.45
    w_lex = 0.55
    combined_scores = {}
    all_ids = set(tfidf_rank.keys()) | set(lex_rank.keys())
    for pid in all_ids:
        tr = tfidf_rank.get(pid)
        lr = lex_rank.get(pid)
        s = 0.0
        if tr is not None:
            s += w_tfidf / (rrf_k + tr)
        if lr is not None:
            s += w_lex / (rrf_k + lr)
        combined_scores[pid] = s

    merged = sorted(all_ids, key=lambda pid: combined_scores.get(pid, 0.0), reverse=True)
    pids = merged
    scores = [combined_scores[pid] * 100.0 for pid in pids]

    pids, scores = apply_limit(pids, scores, limit)

    if cache_key is not None:
        SEARCH_RANK_CACHE.set(cache_key, (pids, scores))

    return pids, scores


def hybrid_search_rank(
    q: str = "",
    limit: int | None = None,
    semantic_weight: float = None,
    *,
    search_rank_fn=None,
    semantic_search_fn=None,
) -> tuple[list[str], list[float], dict]:
    """
    Hybrid search: weighted Reciprocal Rank Fusion (RRF) of keyword + semantic rankings.

    Args:
        q: Search query
        limit: Maximum number of results
        semantic_weight: Weight for semantic search (0-1)
        search_rank_fn: Function for keyword search (for dependency injection)
        semantic_search_fn: Function for semantic search (for dependency injection)

    Returns:
        Tuple of (pids, scores, score_details)
    """
    if semantic_weight is None:
        semantic_weight = SUMMARY_DEFAULT_SEMANTIC_WEIGHT

    # Default dependency injection
    if search_rank_fn is None:
        search_rank_fn = search_rank
    if semantic_search_fn is None:
        from backend.services.semantic_service import semantic_search_rank

        semantic_search_fn = semantic_search_rank

    q = (q or "").strip()
    if not q:
        return [], [], {}

    try:
        from backend.services.data_service import get_features_file_mtime
        from backend.services.semantic_service import get_cached_embeddings_mtime

        cache_key = (
            "hyb",
            q.lower(),
            int(limit) if limit is not None else None,
            float(semantic_weight),
            float(get_features_file_mtime() or 0.0),
            float(get_cached_embeddings_mtime() or 0.0),
        )
        cached = SEARCH_RANK_CACHE.get(cache_key)
        if cached is not None:
            return cached
    except Exception:
        cache_key = None

    candidate_k = None
    if limit is not None:
        candidate_k = min(int(limit) * 2, MAX_RESULTS)

    keyword_pids, keyword_scores = search_rank_fn(q, limit=candidate_k)
    semantic_pids, semantic_scores = semantic_search_fn(q, limit=candidate_k)

    if not keyword_pids and not semantic_pids:
        return [], [], {}
    if not semantic_pids:
        return keyword_pids[:limit] if limit else keyword_pids, keyword_scores[:limit] if limit else keyword_scores, {}
    if not keyword_pids:
        return (
            semantic_pids[:limit] if limit else semantic_pids,
            semantic_scores[:limit] if limit else semantic_scores,
            {},
        )

    keyword_rank = {pid: i + 1 for i, pid in enumerate(keyword_pids)}
    semantic_rank = {pid: i + 1 for i, pid in enumerate(semantic_pids)}

    keyword_score_map = {pid: score for pid, score in zip(keyword_pids, keyword_scores)}
    semantic_score_map = {pid: score for pid, score in zip(semantic_pids, semantic_scores)}

    rrf_k = 60
    w_sem = float(np.clip(semantic_weight, 0.0, 1.0))
    w_kw = 1.0 - w_sem

    combined_scores = {}
    all_pids = set(keyword_rank.keys()) | set(semantic_rank.keys())

    for pid in all_pids:
        kr = keyword_rank.get(pid)
        sr = semantic_rank.get(pid)
        kw_rrf = (w_kw / (rrf_k + kr)) if kr is not None else 0.0
        sem_rrf = (w_sem / (rrf_k + sr)) if sr is not None else 0.0
        final = kw_rrf + sem_rrf
        combined_scores[pid] = final

    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    if limit:
        sorted_results = sorted_results[: int(limit)]

    pids = [pid for pid, _ in sorted_results]
    scores = [combined_scores[pid] * 100.0 for pid in pids]

    score_details = {}
    for pid in pids:
        kr = keyword_rank.get(pid)
        sr = semantic_rank.get(pid)
        kw_rrf = (w_kw / (rrf_k + kr)) if kr is not None else 0.0
        sem_rrf = (w_sem / (rrf_k + sr)) if sr is not None else 0.0
        final = kw_rrf + sem_rrf
        score_details[pid] = {
            "keyword_score": float(keyword_score_map.get(pid, 0.0)),
            "semantic_score": float(semantic_score_map.get(pid, 0.0)),
            "keyword_rank": int(kr) if kr is not None else None,
            "semantic_rank": int(sr) if sr is not None else None,
            "keyword_rrf": float(kw_rrf),
            "semantic_rrf": float(sem_rrf),
            "keyword_weight": w_kw,
            "semantic_weight": w_sem,
            "final_score": float(final * 100.0),
        }

    result = (pids, scores, score_details)
    if cache_key is not None:
        SEARCH_RANK_CACHE.set(cache_key, result)
    return result


def enhanced_search_rank(
    q: str = "",
    limit: int | None = None,
    search_mode: str = "keyword",
    semantic_weight: float = None,
) -> tuple[list[str], list[float]]:
    """
    Enhanced search function supporting multiple search modes.

    Args:
        q: Search query
        limit: Result count limit
        search_mode: Search mode ('keyword', 'semantic', 'hybrid')
        semantic_weight: Semantic search weight (0-1), only used in hybrid mode

    Returns:
        (paper_ids, scores) tuple
    """
    if semantic_weight is None:
        semantic_weight = SUMMARY_DEFAULT_SEMANTIC_WEIGHT

    if not q:
        return [], []

    if search_mode == "keyword":
        return search_rank(q, limit)

    elif search_mode == "semantic":
        from backend.services.semantic_service import semantic_search_rank

        return semantic_search_rank(q, limit)

    elif search_mode == "hybrid":
        pids, scores, _ = hybrid_search_rank(q, limit, semantic_weight)
        return pids, scores

    else:
        raise ValueError(f"Unknown search mode: {search_mode}")
