"""
This script is intended to wake up every 30 min or so (eg via cron),
it checks for any new arxiv papers via the arxiv API and stashes
them into a sqlite database.
"""

import argparse
import logging
import os
import random
import sys
import time

# Ensure repository root is importable when executing this file directly.
# e.g. `python tools/arxiv_daemon.py` would otherwise miss sibling packages like `aslite/`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tqdm

from aslite.arxiv import get_response, parse_response

# Repository layer for cleaner data access
from aslite.repositories import MetaRepository, PaperRepository

# arXiv AI tag groups (shared for collection and display)
# Note: these are default values, actual usage reads from settings.arxiv
CORE = ["cs.AI", "cs.LG", "stat.ML"]
LANG = ["cs.CL", "cs.IR", "cs.CV"]
AGENT = ["cs.MA", "cs.RO", "cs.HC", "cs.GT", "cs.NE"]
APP = ["cs.SE", "cs.CY"]
ALL_TAGS = CORE + LANG + AGENT + APP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arxiv Daemon")
    parser.add_argument("-n", "--num", type=int, default=2000, help="up to how many papers to fetch")
    parser.add_argument(
        "--num-total",
        type=int,
        default=-1,
        help="total papers to fetch across all categories; -1 to disable",
    )
    parser.add_argument("-s", "--start", type=int, default=0, help="start at what index")
    parser.add_argument(
        "-b",
        "--break-after",
        type=int,
        default=20,
        help="how many 0 new papers in a row would cause us to stop early? or 0 to disable.",
    )
    parser.add_argument("-i", "--init", action="store_true", default=False, help="init database")
    parser.add_argument(
        "-m",
        "--max_r",
        type=int,
        default=1000,
        help="how many results to query each time",
    )
    return parser


def run(args: argparse.Namespace, *, all_tags: list[str], empty_response_fallback: int) -> int:
    """Run one arXiv fetch cycle.

    Returns:
        0 if anything changed, 1 otherwise (kept for backward compatibility).
    """
    # query string of papers to look for
    q = "+OR+".join([f"cat:{tag}" for tag in all_tags])

    if args.init:
        keywords = [
            "Reinforcement+AND+Learning",
            "Offline+AND+Optimization",
            "Policy+AND+Optimization",
        ]
        q_lst = q.split("+OR+")
        q_lst = [q_i for q_i in q_lst if q_i]
        q_lst += keywords
    else:
        q_lst = q.split("+OR+")

    total_updated = 0
    total_fetched = 0
    stop_all = False
    category_stats = {}  # Track update stats per category
    for q_each in q_lst:
        if not q_each:
            continue
        if args.num_total > 0 and total_fetched >= args.num_total:
            break
        cat_new, cat_replace, cat_had = 0, 0, 0
        prevn = 0

        # fetch the latest papers
        zero_updates_in_a_row = 0
        break_p_each = False
        k = args.start
        while k < args.num:
            if args.num_total > 0:
                remaining = args.num_total - total_fetched
                if remaining <= 0:
                    stop_all = True
                    break
                max_r_request = min(args.max_r, remaining)
            else:
                max_r_request = args.max_r
            logging.debug("querying arxiv api for query %s at start_index %d", q_each, k)

            # attempt to fetch a batch of papers from arxiv api
            ntried = 0
            nempty = 0
            while True:
                try:
                    resp = get_response(search_query=q_each, start_index=k, max_r=max_r_request)
                    papers = parse_response(resp)
                    time.sleep(0.5)
                    logging.debug("fetch %d papers", len(papers))
                    if len(papers) > 0:
                        k += len(papers)
                        total_fetched += len(papers)
                        break  # otherwise we have to try again
                    nempty += 1
                    if args.break_after > 0 and nempty > args.break_after:
                        logging.error(
                            "ok we tried %d times, nothing is fetched. exitting.",
                            args.break_after,
                        )
                        # sys.exit()
                        break_p_each = True
                        break  # otherwise we have to try again
                    elif args.break_after == 0 and nempty >= empty_response_fallback:
                        logging.error(
                            "received empty responses %d times with break_after=0; stopping to avoid infinite loop",
                            nempty,
                        )
                        break_p_each = True
                        break  # otherwise we have to try again
                except Exception as e:
                    logging.warning(e)
                    logging.warning("will try again in a bit...")
                    ntried += 1
                    if args.break_after > 0 and ntried > args.break_after:
                        logging.error(
                            "ok we tried %d times, something is srsly wrong. exitting.",
                            args.break_after,
                        )
                        # sys.exit()
                        break_p_each = True
                        break  # otherwise we have to try again
                    time.sleep(2 + random.uniform(0, 4))
            if break_p_each:
                break

            # process the batch of retrieved papers
            nhad, nnew, nreplace = 0, 0, 0

            # batch write - open one connection at a time to avoid lock conflicts
            for retry in range(5):
                nhad_try, nnew_try, nreplace_try = 0, 0, 0
                metas_updates = {}
                papers_updates = {}
                try:
                    # First pass: compute stats and prepare batch updates using Repository
                    pids_to_check = [p["_id"] for p in papers]
                    existing = PaperRepository.get_by_ids(pids_to_check)

                    for p in tqdm.tqdm(papers, "Preparing updates", ncols=100, file=sys.stderr):
                        pid = p["_id"]
                        old = existing.get(pid)
                        if old is None:
                            nnew_try += 1
                        else:
                            if p["_time"] > old.get("_time", 0):
                                nreplace_try += 1
                            else:
                                nhad_try += 1

                        papers_updates[pid] = p
                        metas_updates[pid] = {"_time": p["_time"]}

                    # Batch write papers using Repository
                    PaperRepository.set_many(papers_updates)
                    prevn = PaperRepository.count()

                    # Then batch write metas
                    MetaRepository.save_many_no_commit(metas_updates)

                    nhad, nnew, nreplace = nhad_try, nnew_try, nreplace_try
                    break
                except Exception as e:
                    logging.warning(f"DB write failed (attempt {retry+1}): {e}")
                    time.sleep(2 + random.uniform(0, 3))
            else:
                logging.error("Failed to write to database after 5 retries")

            total_updated += nreplace + nnew
            cat_new += nnew
            cat_replace += nreplace
            cat_had += nhad

            # some diagnostic information on how things are coming along
            logging.debug(papers[0]["_time_str"])
            logging.debug(
                "k=%d, out of %d: had %d, replaced %d, new %d. now have: %d",
                k,
                len(papers),
                nhad,
                nreplace,
                nnew,
                prevn,
            )

            # early termination criteria
            if nnew == 0:
                zero_updates_in_a_row += 1
                if args.break_after > 0 and zero_updates_in_a_row >= args.break_after:
                    logging.debug(
                        "breaking out early, no new papers %d times in a row",
                        args.break_after,
                    )
                    break
                elif args.break_after > 0 and k == 0:
                    logging.debug("our very first call for the latest there were no new papers, exitting")
                    break
            else:
                zero_updates_in_a_row = 0

            # zzz
            time.sleep(1 + random.uniform(0, 3))
        if stop_all:
            break
        # Record stats for this category
        category_stats[q_each] = {"new": cat_new, "replace": cat_replace, "had": cat_had}
        print(f"[{q_each}] new: {cat_new}, replaced: {cat_replace}, existed: {cat_had}")

    # Print summary
    print("\n=== Category Summary ===")
    for cat, stats in category_stats.items():
        print(f"  {cat}: +{stats['new']} new, ~{stats['replace']} replaced, ={stats['had']} existed")
    print(f"Total updated: {total_updated}")

    # exit code: keep legacy behavior (1 means nothing updated)
    return 0 if total_updated > 0 else 1


def main(argv: list[str] | None = None) -> int:
    from config import settings

    all_tags = list(settings.arxiv.all_tags)
    empty_response_fallback = int(settings.arxiv.empty_response_fallback)

    log_level_name = settings.log_level.upper()
    log_level = getattr(logging, log_level_name, logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format="%(name)s %(levelname)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logging.debug(
        "Proxy HTTP_PROXY %s HTTPS_PROXY %s",
        os.environ.get("http_proxy", None),
        os.environ.get("https_proxy", None),
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.debug(args)

    return run(args, all_tags=all_tags, empty_response_fallback=empty_response_fallback)


if __name__ == "__main__":
    raise SystemExit(main())
