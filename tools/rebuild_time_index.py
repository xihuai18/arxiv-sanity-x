"""Rebuild the metas_time_index table from the metas table.

Usage:
    python -m tools rebuild_time_index
"""

import sys

from loguru import logger

from aslite.db import metas_time_index_count, rebuild_metas_time_index


def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    logger.info("Checking current metas_time_index status...")
    current_count = metas_time_index_count()
    logger.info(f"Current index entries: {current_count}")

    rebuild_metas_time_index()

    new_count = metas_time_index_count()
    logger.info(f"New index entries: {new_count}")


if __name__ == "__main__":
    main()
