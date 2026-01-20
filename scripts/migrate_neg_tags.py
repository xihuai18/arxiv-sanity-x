#!/usr/bin/env python3
"""Backup and migrate dict.db to include neg_tags table.

- Creates a timestamped backup of dict.db
- Ensures neg_tags table exists
- Initializes empty neg_tags entries for all existing users
- Prints current table list for verification
"""
import argparse
import datetime as dt
import os
import shutil
import sqlite3

from aslite.db import DICT_DB_FILE, get_neg_tags_db, get_tags_db


def list_tables(db_path: str):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def backup_db(db_path: str, backup_dir: str) -> str:
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = os.path.join(backup_dir, f"dict.db.{timestamp}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def migrate_neg_tags(db_path: str, dry_run: bool = False) -> int:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"dict.db not found: {db_path}")

    updated = 0
    if dry_run:
        return updated

    with get_tags_db(flag="c") as tags_db:
        with get_neg_tags_db(flag="c") as neg_tags_db:
            for user in tags_db.keys():
                if user not in neg_tags_db:
                    neg_tags_db[user] = {}
                    updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser(description="Backup and migrate dict.db to include neg_tags table")
    parser.add_argument("--db", default=DICT_DB_FILE, help="Path to dict.db (default from aslite.db)")
    parser.add_argument("--backup-dir", default=None, help="Directory to store backups (default: data/backup)")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without modifying DB")
    args = parser.parse_args()

    db_path = args.db
    backup_dir = args.backup_dir or os.path.join(os.path.dirname(db_path), "backup")

    print(f"dict.db path: {db_path}")
    print("Existing tables:", list_tables(db_path))

    if not os.path.exists(db_path):
        print("dict.db not found; aborting.")
        return 1

    if args.dry_run:
        print("Dry-run: no changes will be made.")
        return 0

    backup_path = backup_db(db_path, backup_dir)
    print(f"Backup created: {backup_path}")

    updated = migrate_neg_tags(db_path)
    print(f"Initialized neg_tags for {updated} users.")

    print("Tables after migration:", list_tables(db_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
