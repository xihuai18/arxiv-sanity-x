"""
This script is intended to wake up every 30 min or so (eg via cron),
it checks for any new arxiv papers via the arxiv API and stashes
them into a sqlite database.
"""

import sys
import time
import random
import logging
import tqdm
import argparse

from aslite.arxiv import get_response, parse_response
from aslite.db import get_papers_db, get_metas_db
import os


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logging.info(
        f"Proxy HTTP_PROXY {os.environ.get('http_proxy', None)} HTTPS_PROXY {os.environ.get('https_proxy', None)}"
    )

    parser = argparse.ArgumentParser(description="Arxiv Daemon")
    parser.add_argument(
        "-n", "--num", type=int, default=1000, help="up to how many papers to fetch"
    )
    parser.add_argument(
        "-s", "--start", type=int, default=0, help="start at what index"
    )
    parser.add_argument(
        "-b",
        "--break-after",
        type=int,
        default=5,
        help="how many 0 new papers in a row would cause us to stop early? or 0 to disable.",
    )
    parser.add_argument(
        "-i", "--init", action="store_true", default=False, help="init database"
    )
    parser.add_argument(
        "-m",
        "--max_r",
        type=int,
        default=1000,
        help="how many results to query each time",
    )
    args = parser.parse_args()
    print(args)
    """
    Quick note on the break_after argument: In a typical setting where one wants to update
    the papers database you'd choose a slightly higher num, but then break out early in case
    we've reached older papers that are already part of the database, to spare the arxiv API.
    """

    # query string of papers to look for
    q = "cat:cs.CV+OR+cat:cs.CL+OR+cat:cs.LG+OR+cat:cs.AI+OR+cat:cs.NE+OR+cat:cs.RO+OR+cat:stat.ML+OR+cat:cs.GT+OR+cat:cs.HC+OR+cat:cs.MA"
    # 20000
    # q = "cat:cs.CV+OR+cat:cs.CL+OR+cat:cs.RO"
    # 50000
    # q = "cat:cs.LG+OR+cat:cs.AI+OR+cat:stat.ML"
    # 10000
    # q = "cat:cs.NE+OR+cat:cs.GT+OR+cat:cs.HC+OR+cat:cs.MA"

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
        q_lst = [q]
    # example
    # https://export.arxiv.org/api/query?search_query=(cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.AI+OR+Large%20Language%20Model)&start=0&max_results=100

    total_updated = 0
    for q_each in q_lst:
        pdb = get_papers_db(flag="c")
        mdb = get_metas_db(flag="c")
        prevn = len(pdb)

        def store(p):
            pdb[p["_id"]] = p
            mdb[p["_id"]] = {"_time": p["_time"]}

        # fetch the latest papers
        zero_updates_in_a_row = 0
        break_p_each = False
        k = args.start
        while k < args.num:
            logging.info(
                "querying arxiv api for query %s at start_index %d" % (q_each, k)
            )

            # attempt to fetch a batch of papers from arxiv api
            ntried = 0
            nempty = 0
            while True:
                try:
                    resp = get_response(
                        search_query=q_each, start_index=k, max_r=args.max_r
                    )
                    papers = parse_response(resp)
                    time.sleep(0.5)
                    logging.info(f"fetch {len(papers)} papers")
                    if len(papers) > 0:
                        k += len(papers)
                        break  # otherwise we have to try again
                    nempty += 1
                    if nempty > 50:
                        logging.error(
                            "ok we tried 50 times, nothing is fetched. exitting."
                        )
                        # sys.exit()
                        break_p_each = True
                        break  # otherwise we have to try again
                except Exception as e:
                    logging.warning(e)
                    logging.warning("will try again in a bit...")
                    ntried += 1
                    if ntried > 50:
                        logging.error(
                            "ok we tried 50 times, something is srsly wrong. exitting."
                        )
                        # sys.exit()
                        break_p_each = True
                        break  # otherwise we have to try again
                    time.sleep(2 + random.uniform(0, 4))
            if break_p_each:
                break

            # process the batch of retrieved papers
            nhad, nnew, nreplace = 0, 0, 0
            for p in tqdm.tqdm(papers, "Updating database"):
                pid = p["_id"]
                if pid in pdb:
                    if p["_time"] > pdb[pid]["_time"]:
                        # replace, this one is newer
                        store(p)
                        nreplace += 1
                    else:
                        # we already had this paper, nothing to do
                        store(p)
                        nhad += 1
                else:
                    # new, simple store into database
                    store(p)
                    nnew += 1
            prevn = len(pdb)
            total_updated += nreplace + nnew

            # some diagnostic information on how things are coming along
            logging.info(papers[0]["_time_str"])
            logging.info(
                "k=%d, out of %d: had %d, replaced %d, new %d. now have: %d"
                % (k, len(papers), nhad, nreplace, nnew, prevn)
            )

            # early termination criteria
            if nnew == 0:
                zero_updates_in_a_row += 1
                if args.break_after > 0 and zero_updates_in_a_row >= args.break_after:
                    logging.info(
                        "breaking out early, no new papers %d times in a row"
                        % (args.break_after,)
                    )
                    break
                elif k == 0:
                    logging.info(
                        "our very first call for the latest there were no new papers, exitting"
                    )
                    break
            else:
                zero_updates_in_a_row = 0

            # zzz
            time.sleep(1 + random.uniform(0, 3))

    # exit with OK status if anything at all changed, but if nothing happened then raise 1
    sys.exit(0 if total_updated > 0 else 1)
