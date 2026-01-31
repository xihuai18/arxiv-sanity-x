"""
Repository Layer - High-level database operations abstraction.

This module provides a clean abstraction over raw database access functions,
encapsulating common business logic patterns.
This is an optional layer - existing code can continue to use get_*_db() directly.

Design principles:
- Each Repository corresponds to a business domain (Paper, Tag, ReadingList, etc.)
- Uses static methods, stateless design, easy to call
- Internally uses optimized batch query methods (get_many, items_with_prefix, etc.)
- Provides type hints, improves IDE support and code readability

Advantages:
1. Encapsulates common query patterns - avoids repetitive database access code
2. Centralized performance optimization - batch operations implemented at Repository layer
3. Easy to test - can mock Repository for unit testing
4. Separation of concerns - business logic separated from data access
5. Backward compatible - doesn't affect existing code, gradual migration

Usage examples:
    from aslite.repositories import PaperRepository, TagRepository

    # Batch fetch papers
    papers = PaperRepository.get_by_ids(['2301.00001', '2301.00002'])

    # Get user tags
    tags = TagRepository.get_user_tags('alice')

    # Get reading list
    reading_list = ReadingListRepository.get_user_reading_list('alice')

Migration strategy:
- New features should prefer Repository layer
- Existing code can migrate gradually, not mandatory
- Maintain compatibility with direct database access
"""

import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

from loguru import logger

from aslite.db import (
    get_combined_tags_db,
    get_email_db,
    get_keywords_db,
    get_metas_db,
    get_neg_tags_db,
    get_papers_db,
    get_readinglist_db,
    get_readinglist_index_db,
    get_summary_status_db,
    get_tags_db,
    get_uploaded_papers_db,
    get_uploaded_papers_index_db,
    update_metas_time_index,
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def readinglist_key(user: str, pid: str) -> str:
    """Generate reading list key."""
    return f"{user}::{pid}"


def summary_status_key(pid: str, model: str) -> str:
    """Generate summary status key."""
    return f"{pid}::{model}"


def task_status_key(task_id: str) -> str:
    """Generate task status key."""
    return f"task::{task_id}"


def summary_generation_epoch_key(pid: str, model: str) -> str:
    """Generate per-(pid, model) generation epoch key.

    This is used for cooperative cancellation: when an epoch is bumped, any in-flight
    summary jobs that were started/enqueued under an older epoch should abort and
    avoid writing caches.
    """
    return f"epoch::{pid}::{model}"


def parse_readinglist_key(key: str) -> Tuple[str, str]:
    """Parse reading list key into (user, pid)."""
    parts = key.split("::", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", ""


# -----------------------------------------------------------------------------
# Paper Repository
# -----------------------------------------------------------------------------


class PaperRepository:
    """Repository for paper-related operations."""

    @staticmethod
    def get_by_ids(pids: List[str]) -> Dict[str, dict]:
        """
        Batch fetch papers by IDs.

        Args:
            pids: List of paper IDs

        Returns:
            Dictionary mapping paper ID to paper data
        """
        if not pids:
            return {}
        with get_papers_db() as pdb:
            return pdb.get_many(pids)

    @staticmethod
    def get_by_ids_with_cache(pids: List[str], cache=None) -> Dict[str, dict]:
        """
        Batch fetch papers by IDs with optional cache update.

        Args:
            pids: List of paper IDs
            cache: Optional cache object with set(key, value) method

        Returns:
            Dictionary mapping paper ID to paper data
        """
        papers = PaperRepository.get_by_ids(pids)
        if cache is not None:
            for pid, paper in papers.items():
                try:
                    cache.set(pid, paper)
                except Exception:
                    pass
        return papers

    @staticmethod
    def get_by_id(pid: str) -> Optional[dict]:
        """
        Get a single paper by ID.

        Args:
            pid: Paper ID

        Returns:
            Paper data or None if not found
        """
        with get_papers_db() as pdb:
            return pdb.get(pid)

    @staticmethod
    def open_readonly():
        """Open a readonly papers DB handle (caller must close)."""
        return get_papers_db(flag="r")

    @staticmethod
    def count() -> int:
        """Return number of papers in the database."""
        with get_papers_db() as pdb:
            return len(pdb)

    @staticmethod
    def iter_all_papers():
        """Stream all paper items as (pid, paper) tuples."""
        t_start = time.time()
        logger.trace("[BLOCKING] PaperRepository.iter_all_papers: starting full table scan...")
        count = 0
        with get_papers_db() as pdb:
            for item in pdb.items():
                count += 1
                yield item
        logger.trace(
            f"[BLOCKING] PaperRepository.iter_all_papers: yielded {count} papers in {time.time() - t_start:.2f}s"
        )

    @staticmethod
    def get_recent_papers(days: int = 7, limit: Optional[int] = None) -> List[Tuple[str, dict]]:
        """
        Get papers from the last N days.

        Args:
            days: Number of days to look back
            limit: Maximum number of papers to return

        Returns:
            List of (paper_id, paper_data) tuples, sorted by time descending
        """
        cutoff = time.time() - days * 24 * 3600
        papers = []

        with get_papers_db() as pdb:
            for pid, p in pdb.items():
                if p.get("_time", 0) > cutoff:
                    papers.append((pid, p))

        # Sort by time descending
        papers.sort(key=lambda x: x[1].get("_time", 0), reverse=True)

        if limit:
            papers = papers[:limit]

        return papers

    @staticmethod
    def count_all() -> int:
        """Get total number of papers in database."""
        with get_papers_db() as pdb:
            return len(pdb)

    @staticmethod
    def save(pid: str, paper_data: dict):
        """
        Save or update a paper.

        Args:
            pid: Paper ID
            paper_data: Paper data dictionary
        """
        with get_papers_db(flag="c") as pdb:
            pdb[pid] = paper_data

    @staticmethod
    def save_many(papers: Dict[str, dict]):
        """
        Batch save multiple papers.

        Args:
            papers: Dictionary mapping paper ID to paper data
        """
        if not papers:
            return
        with get_papers_db(flag="c") as pdb:
            pdb.set_many(papers)

    @staticmethod
    def set_many(papers: Dict[str, dict]):
        """Alias for save_many to keep naming consistent with lower-level API."""
        PaperRepository.save_many(papers)


# -----------------------------------------------------------------------------
# Meta Repository
# -----------------------------------------------------------------------------


class MetaRepository:
    """Repository for paper metadata operations."""

    @staticmethod
    def get_by_ids(pids: List[str]) -> Dict[str, dict]:
        """Batch fetch metadata by paper IDs."""
        if not pids:
            return {}
        with get_metas_db() as mdb:
            return mdb.get_many(pids)

    @staticmethod
    def get_by_id(pid: str) -> Optional[dict]:
        """Get metadata for a single paper."""
        with get_metas_db() as mdb:
            return mdb.get(pid)

    @staticmethod
    def save_many(metas: Dict[str, dict]):
        """Batch save metadata and update time index."""
        if not metas:
            return
        with get_metas_db(flag="c") as mdb:
            mdb.set_many(metas)
        # Update time index
        MetaRepository._update_time_index(metas)

    @staticmethod
    def save_many_no_commit(metas: Dict[str, dict]):
        """Batch save metadata without autocommit (caller manages commit)."""
        if not metas:
            return
        with get_metas_db(flag="c", autocommit=False) as mdb:
            mdb.set_many(metas)
            mdb.commit()
        # Update time index
        MetaRepository._update_time_index(metas)

    @staticmethod
    def _update_time_index(metas: Dict[str, dict]):
        """Update the metas_time_index table with _time values from metas."""
        if not metas:
            return
        pid_time_pairs = []
        for pid, meta in metas.items():
            _time = meta.get("_time", 0) if isinstance(meta, dict) else 0
            pid_time_pairs.append((pid, _time))
        if pid_time_pairs:
            try:
                update_metas_time_index(pid_time_pairs)
            except Exception:
                # Don't fail the main operation if index update fails
                pass

    @staticmethod
    def iter_all_metas():
        """Stream all metadata items as (pid, meta) tuples."""
        t_start = time.time()
        logger.trace("[BLOCKING] MetaRepository.iter_all_metas: starting full table scan...")
        count = 0
        with get_metas_db() as mdb:
            for item in mdb.items():
                count += 1
                yield item
        logger.trace(f"[BLOCKING] MetaRepository.iter_all_metas: yielded {count} metas in {time.time() - t_start:.2f}s")

    @staticmethod
    def get_latest_n(n: int, use_index: bool = True) -> List[Tuple[str, dict]]:
        """Get the latest n papers sorted by _time descending.

        This method uses the metas_time_index table for efficient queries,
        falling back to full table scan if the index is not available.

        Args:
            n: Number of papers to fetch
            use_index: If True, try to use the time index table first

        Returns:
            List of (pid, meta) tuples, sorted by _time descending
        """
        if n <= 0:
            return []

        if use_index:
            from aslite.db import get_latest_pids_by_time, metas_time_index_count

            # Check if index has enough entries
            index_count = metas_time_index_count()
            if index_count > 0:
                # Get latest pids from index
                pid_time_pairs = get_latest_pids_by_time(n)
                if pid_time_pairs:
                    pids = [pid for pid, _time in pid_time_pairs]
                    # Batch fetch metas
                    metas_dict = MetaRepository.get_by_ids(pids)
                    # Return in order (index already sorted by _time desc)
                    result = []
                    for pid in pids:
                        if pid in metas_dict:
                            result.append((pid, metas_dict[pid]))
                    return result

        # Fallback: full table scan with heapq (original behavior)
        import heapq

        heap: List[Tuple[float, str, dict]] = []
        with get_metas_db() as mdb:
            for pid, meta in mdb.items():
                t = meta.get("_time", 0) if isinstance(meta, dict) else 0
                if len(heap) < n:
                    heapq.heappush(heap, (t, pid, meta))
                elif t > heap[0][0]:
                    heapq.heapreplace(heap, (t, pid, meta))

        # Sort by time descending
        return sorted([(pid, meta) for _t, pid, meta in heap], key=lambda x: x[1].get("_time", 0), reverse=True)


# -----------------------------------------------------------------------------
# Tag Repository
# -----------------------------------------------------------------------------


class TagRepository:
    """Repository for tag-related operations."""

    @staticmethod
    def get_user_tags(user: str) -> Dict[str, Set[str]]:
        """
        Get all tags for a user.

        Args:
            user: Username

        Returns:
            Dictionary mapping tag name to set of paper IDs
        """
        with get_tags_db() as tdb:
            return tdb.get(user, {})

    @staticmethod
    def get_all_tags() -> Dict[str, Dict[str, Set[str]]]:
        """Get all tags for all users."""
        with get_tags_db() as tdb:
            return {user: tags for user, tags in tdb.items()}

    @staticmethod
    def get_user_neg_tags(user: str) -> Dict[str, Set[str]]:
        """Get all negative tags for a user (convenience wrapper)."""
        return NegativeTagRepository.get_user_neg_tags(user)

    @staticmethod
    def get_user_combined_tags(user: str) -> Set[str]:
        """Get all combined tags for a user (convenience wrapper)."""
        return CombinedTagRepository.get_user_combined_tags(user)

    @staticmethod
    def get_all_users_tags() -> Dict[str, Dict[str, Set[str]]]:
        """
        Get tags for all users.

        Returns:
            Dictionary mapping username to their tags
        """
        with get_tags_db() as tdb:
            return {user: tags for user, tags in tdb.items()}

    @staticmethod
    def add_paper_to_tag(user: str, pid: str, tag: str):
        """
        Add a paper to a user's tag.

        Args:
            user: Username
            pid: Paper ID
            tag: Tag name
        """
        with get_tags_db(flag="c") as tdb:
            tags = tdb.get(user, {})
            if tag not in tags:
                tags[tag] = set()
            tags[tag].add(pid)
            tdb[user] = tags

    @staticmethod
    def add_paper_to_tag_and_remove_neg(user: str, pid: str, tag: str):
        """
        Add a paper to a user's tag and remove it from negative tag if present.

        This mirrors the legacy behavior in serve.py.
        """
        with get_tags_db(flag="c") as tdb:
            with get_neg_tags_db(flag="c") as ntdb:
                tags = tdb.get(user, {})
                neg_tags = ntdb.get(user, {})

                if tag not in tags:
                    tags[tag] = set()
                tags[tag].add(pid)

                if tag in neg_tags and pid in neg_tags[tag]:
                    neg_tags[tag].discard(pid)
                    if not neg_tags[tag]:
                        del neg_tags[tag]
                    ntdb[user] = neg_tags

                tdb[user] = tags

    @staticmethod
    def remove_paper_from_tag(user: str, pid: str, tag: str) -> bool:
        """
        Remove a paper from a user's tag.

        Args:
            user: Username
            pid: Paper ID
            tag: Tag name

        Returns:
            True if paper was removed, False if not found
        """
        with get_tags_db(flag="c") as tdb:
            tags = tdb.get(user, {})
            if tag in tags and pid in tags[tag]:
                tags[tag].discard(pid)
                if not tags[tag]:  # Remove empty tag
                    del tags[tag]
                tdb[user] = tags
                return True
            return False

    @staticmethod
    def remove_paper_from_tag_verbose(user: str, pid: str, tag: str) -> str:
        """
        Remove a paper from a user's tag with legacy response messages.

        Returns:
            "ok" if removed, otherwise legacy error message.
        """
        with get_tags_db(flag="c") as tdb:
            if user not in tdb:
                return r"user has no library of tags ¯\_(ツ)_/¯"

            tags = tdb[user]
            if tag not in tags:
                return f"user doesn't have the tag {tag}"

            if pid in tags[tag]:
                tags[tag].remove(pid)
                if len(tags[tag]) == 0:
                    del tags[tag]
                tdb[user] = tags
                return "ok"

            return f"user doesn't have paper {pid} in tag {tag}"

    @staticmethod
    def delete_tag(user: str, tag: str) -> bool:
        """
        Delete an entire tag for a user.

        Args:
            user: Username
            tag: Tag name

        Returns:
            True if tag was deleted, False if not found
        """
        with get_tags_db(flag="c") as tdb:
            tags = tdb.get(user, {})
            if tag in tags:
                del tags[tag]
                tdb[user] = tags
                return True
            return False

    @staticmethod
    def delete_tag_full(user: str, tag: str) -> str:
        """
        Delete a tag and clean up combined/negative tags.

        Returns:
            "ok" if deleted, otherwise legacy error message.
        """
        with get_tags_db(flag="c") as tdb:
            with get_combined_tags_db(flag="c") as ctdb:
                with get_neg_tags_db(flag="c") as ntdb:
                    if user not in tdb:
                        return "user does not have a library"

                    tags = tdb[user]
                    if tag not in tags:
                        return "user does not have this tag"

                    del tags[tag]
                    tdb[user] = tags

                    combined = ctdb.get(user)
                    if combined:
                        for ctag in list(combined):
                            if tag in ctag.split(","):
                                combined.remove(ctag)
                        ctdb[user] = combined

                    neg_tags = ntdb.get(user)
                    if neg_tags and tag in neg_tags:
                        del neg_tags[tag]
                        ntdb[user] = neg_tags

        return "ok"

    @staticmethod
    def rename_tag(user: str, old_tag: str, new_tag: str) -> bool:
        """
        Rename a tag for a user.

        Args:
            user: Username
            old_tag: Current tag name
            new_tag: New tag name

        Returns:
            True if tag was renamed, False if old tag not found
        """
        with get_tags_db(flag="c") as tdb:
            tags = tdb.get(user, {})
            if old_tag in tags:
                pids = tags[old_tag]
                del tags[old_tag]
                if new_tag not in tags:
                    tags[new_tag] = pids
                else:
                    tags[new_tag] = tags[new_tag].union(pids)
                tdb[user] = tags
                return True
            return False

    @staticmethod
    def rename_tag_full(user: str, old_tag: str, new_tag: str) -> str:
        """
        Rename a tag and update negative/combined tags.

        Returns:
            "ok" if renamed, otherwise legacy error message.
        """
        with get_tags_db(flag="c") as tdb:
            with get_neg_tags_db(flag="c") as ntdb:
                if user not in tdb:
                    return "user does not have a library"

                tags = tdb[user]
                if old_tag not in tags:
                    return "user does not have this tag"

                pids = tags[old_tag]
                del tags[old_tag]
                if new_tag not in tags:
                    tags[new_tag] = pids
                else:
                    tags[new_tag] = tags[new_tag].union(pids)
                tdb[user] = tags

                neg_tags = ntdb.get(user)
                if neg_tags and old_tag in neg_tags:
                    o_pids = neg_tags[old_tag]
                    del neg_tags[old_tag]
                    if new_tag not in neg_tags:
                        neg_tags[new_tag] = o_pids
                    else:
                        neg_tags[new_tag] = neg_tags[new_tag].union(o_pids)
                    ntdb[user] = neg_tags

        with get_combined_tags_db(flag="c") as ctdb:
            combined = ctdb.get(user)
            if combined:
                for ctag in list(combined):
                    if old_tag in (ctag_split := ctag.split(",")):
                        ctag_split = [ct_s if ct_s != old_tag else new_tag for ct_s in ctag_split]
                        combined.remove(ctag)
                        combined.add(",".join(ctag_split))
                ctdb[user] = combined

        return "ok"

    @staticmethod
    def create_tag(user: str, tag: str) -> str:
        """
        Create an empty tag for a user.

        Returns:
            "ok" if created, or legacy error message.
        """
        with get_tags_db(flag="c") as tdb:
            tags = tdb.get(user, {})
            if tag in tags:
                return "user has repeated tag"
            tags[tag] = set()
            tdb[user] = tags
        return "ok"

    @staticmethod
    def get_papers_by_tag(user: str, tag: str) -> Set[str]:
        """
        Get all paper IDs for a specific tag.

        Args:
            user: Username
            tag: Tag name

        Returns:
            Set of paper IDs
        """
        with get_tags_db() as tdb:
            tags = tdb.get(user, {})
            return tags.get(tag, set())

    @staticmethod
    def set_tag_label(user: str, pid: str, tag: str, label: int):
        """
        Set tag label for a paper (positive, negative, or neutral).

        Args:
            user: Username
            pid: Paper ID
            tag: Tag name
            label: 1 for positive, -1 for negative, 0 for neutral (remove)
        """
        with get_tags_db(flag="c") as tags_db:
            with get_neg_tags_db(flag="c") as neg_tags_db:
                if user not in tags_db:
                    tags_db[user] = {}
                if user not in neg_tags_db:
                    neg_tags_db[user] = {}

                pos_d = tags_db[user]
                neg_d = neg_tags_db[user]

                if label == 1:
                    # Add to positive, remove from negative
                    if tag not in pos_d:
                        pos_d[tag] = set()
                    pos_d[tag].add(pid)
                    if tag in neg_d and pid in neg_d[tag]:
                        neg_d[tag].discard(pid)
                        if len(neg_d[tag]) == 0:
                            del neg_d[tag]
                elif label == -1:
                    # Add to negative, remove from positive
                    if tag not in neg_d:
                        neg_d[tag] = set()
                    neg_d[tag].add(pid)
                    if tag in pos_d and pid in pos_d[tag]:
                        pos_d[tag].discard(pid)
                        if len(pos_d[tag]) == 0:
                            del pos_d[tag]
                else:  # label == 0
                    # Remove from both
                    if tag in pos_d and pid in pos_d[tag]:
                        pos_d[tag].discard(pid)
                        if len(pos_d[tag]) == 0:
                            del pos_d[tag]
                    if tag in neg_d and pid in neg_d[tag]:
                        neg_d[tag].discard(pid)
                        if len(neg_d[tag]) == 0:
                            del neg_d[tag]

                tags_db[user] = pos_d
                neg_tags_db[user] = neg_d


# -----------------------------------------------------------------------------
# Negative Tag Repository
# -----------------------------------------------------------------------------


class NegativeTagRepository:
    """Repository for negative tag operations."""

    @staticmethod
    def get_user_neg_tags(user: str) -> Dict[str, Set[str]]:
        """Get all negative tags for a user."""
        with get_neg_tags_db() as ntdb:
            return ntdb.get(user, {})

    @staticmethod
    def add_paper_to_neg_tag(user: str, pid: str, tag: str):
        """Add a paper to a user's negative tag."""
        with get_neg_tags_db(flag="c") as ntdb:
            neg_tags = ntdb.get(user, {})
            if tag not in neg_tags:
                neg_tags[tag] = set()
            neg_tags[tag].add(pid)
            ntdb[user] = neg_tags

    @staticmethod
    def remove_paper_from_neg_tag(user: str, pid: str, tag: str) -> bool:
        """Remove a paper from a user's negative tag."""
        with get_neg_tags_db(flag="c") as ntdb:
            neg_tags = ntdb.get(user, {})
            if tag in neg_tags and pid in neg_tags[tag]:
                neg_tags[tag].discard(pid)
                if not neg_tags[tag]:
                    del neg_tags[tag]
                ntdb[user] = neg_tags
                return True
            return False

    @staticmethod
    def ensure_users(users: Iterable[str]) -> int:
        """Ensure negative tag entries exist for users; returns count created."""
        updated = 0
        with get_neg_tags_db(flag="c") as ntdb:
            for user in users:
                if user not in ntdb:
                    ntdb[user] = {}
                    updated += 1
        return updated


# -----------------------------------------------------------------------------
# Combined Tag Repository
# -----------------------------------------------------------------------------


class CombinedTagRepository:
    """Repository for combined tag operations."""

    @staticmethod
    def get_user_combined_tags(user: str) -> Set[str]:
        """
        Get all combined tags for a user.

        Args:
            user: Username

        Returns:
            Set of combined tag strings (e.g., "RL,NLP")
        """
        with get_combined_tags_db() as ctdb:
            return ctdb.get(user, set())

    @staticmethod
    def get_all_combined_tags() -> Dict[str, Set[str]]:
        """Get all combined tags for all users."""
        with get_combined_tags_db() as ctdb:
            return {user: ctags for user, ctags in ctdb.items()}

    @staticmethod
    def add_combined_tag(user: str, combined_tag: str):
        """
        Add a combined tag for a user.

        Args:
            user: Username
            combined_tag: Combined tag string (e.g., "RL,NLP")
        """
        with get_combined_tags_db(flag="c") as ctdb:
            ctags = ctdb.get(user, set())
            ctags.add(combined_tag)
            ctdb[user] = ctags

    @staticmethod
    def remove_combined_tag(user: str, combined_tag: str) -> bool:
        """Remove a combined tag for a user."""
        with get_combined_tags_db(flag="c") as ctdb:
            ctags = ctdb.get(user, set())
            if combined_tag in ctags:
                ctags.discard(combined_tag)
                ctdb[user] = ctags
                return True
            return False

    @staticmethod
    def rename_combined_tag(user: str, old_tag: str, new_tag: str) -> bool:
        """
        Rename a combined tag for a user.

        Args:
            user: Username
            old_tag: Old combined tag string
            new_tag: New combined tag string

        Returns:
            True if renamed successfully, False if old tag not found
        """
        with get_combined_tags_db(flag="c") as ctdb:
            ctags = ctdb.get(user, set())
            if old_tag not in ctags:
                return False
            ctags.discard(old_tag)
            ctags.add(new_tag)
            ctdb[user] = ctags
            return True

    @staticmethod
    def has_combined_tag(user: str, combined_tag: str) -> bool:
        """Check if user has a specific combined tag."""
        with get_combined_tags_db() as ctdb:
            ctags = ctdb.get(user, set())
            return combined_tag in ctags


# -----------------------------------------------------------------------------
# Keyword Repository
# -----------------------------------------------------------------------------


class KeywordRepository:
    """Repository for keyword tracking operations."""

    @staticmethod
    def get_user_keywords(user: str) -> Dict[str, Set[str]]:
        """
        Get all keywords for a user.

        Args:
            user: Username

        Returns:
            Dictionary mapping keyword to set of paper IDs
        """
        with get_keywords_db() as kdb:
            return kdb.get(user, {})

    @staticmethod
    def get_all_keywords() -> Dict[str, Dict[str, Set[str]]]:
        """Get all keywords for all users."""
        with get_keywords_db() as kdb:
            return {user: keywords for user, keywords in kdb.items()}

    @staticmethod
    def add_keyword(user: str, keyword: str):
        """
        Add a keyword for a user.

        Args:
            user: Username
            keyword: Keyword to track
        """
        with get_keywords_db(flag="c") as kdb:
            keywords = kdb.get(user, {})
            if keyword not in keywords:
                keywords[keyword] = set()
            kdb[user] = keywords

    @staticmethod
    def remove_keyword(user: str, keyword: str) -> bool:
        """Remove a keyword for a user."""
        with get_keywords_db(flag="c") as kdb:
            keywords = kdb.get(user, {})
            if keyword in keywords:
                del keywords[keyword]
                kdb[user] = keywords
                return True
            return False

    @staticmethod
    def add_paper_to_keyword(user: str, keyword: str, pid: str):
        """Add a paper to a keyword's tracking set."""
        with get_keywords_db(flag="c") as kdb:
            keywords = kdb.get(user, {})
            if keyword not in keywords:
                keywords[keyword] = set()
            keywords[keyword].add(pid)
            kdb[user] = keywords

    @staticmethod
    def rename_keyword(user: str, old_keyword: str, new_keyword: str) -> str:
        """
        Rename a keyword for a user.

        Returns:
            "ok" if renamed, otherwise legacy error message.
        """
        with get_keywords_db(flag="c") as kdb:
            keywords = kdb.get(user, {})
            if not keywords:
                return "user does not have a library"
            if old_keyword not in keywords:
                return "user does not have this keyword"

            # Merge if new keyword exists, otherwise rename
            if new_keyword in keywords:
                keywords[new_keyword] = keywords[new_keyword].union(keywords[old_keyword])
            else:
                keywords[new_keyword] = keywords[old_keyword]
            del keywords[old_keyword]
            kdb[user] = keywords
        return "ok"


# -----------------------------------------------------------------------------
# Reading List Repository
# -----------------------------------------------------------------------------


class ReadingListRepository:
    """Repository for reading list operations."""

    @staticmethod
    def get_user_reading_list(user: str) -> Dict[str, dict]:
        """
        Get all reading list items for a user.

        Args:
            user: Username

        Returns:
            Dictionary mapping paper ID to reading list item data
        """
        # Try index first
        with get_readinglist_index_db() as idx_db:
            indexed_pids = idx_db.get(user, [])

        if indexed_pids:
            with get_readinglist_db() as rldb:
                rl_keys = [readinglist_key(user, pid) for pid in indexed_pids]
                fetched = rldb.get_many(rl_keys)
                result = {}
                for pid in indexed_pids:
                    rl_key = readinglist_key(user, pid)
                    if rl_key in fetched:
                        result[pid] = fetched[rl_key]
                return result

        # Fallback: prefix scan (and rebuild index)
        result = {}
        pids = []
        with get_readinglist_db() as rldb:
            prefix = f"{user}::"
            for key, value in rldb.items_with_prefix(prefix):
                _, pid = parse_readinglist_key(key)
                if pid:
                    result[pid] = value
                    pids.append(pid)

        if pids:
            try:
                with get_readinglist_index_db(flag="c") as idx_db:
                    idx_db[user] = list(dict.fromkeys(pids))
            except Exception:
                pass

        return result

    @staticmethod
    def get_reading_list_item(user: str, pid: str) -> Optional[dict]:
        """
        Get a specific reading list item.

        Args:
            user: Username
            pid: Paper ID

        Returns:
            Reading list item data or None if not found
        """
        rl_key = readinglist_key(user, pid)
        with get_readinglist_db() as rldb:
            return rldb.get(rl_key)

    @staticmethod
    def add_to_reading_list(user: str, pid: str, item_data: dict):
        """
        Add a paper to user's reading list.

        Args:
            user: Username
            pid: Paper ID
            item_data: Reading list item data
        """
        rl_key = readinglist_key(user, pid)
        with get_readinglist_db(flag="c") as rldb:
            rldb[rl_key] = item_data

        # Update index
        with get_readinglist_index_db(flag="c") as idx_db:
            indexed_pids = idx_db.get(user, [])
            if pid not in indexed_pids:
                indexed_pids.append(pid)
                idx_db[user] = indexed_pids

    @staticmethod
    def remove_from_reading_list(user: str, pid: str) -> bool:
        """
        Remove a paper from user's reading list.

        Args:
            user: Username
            pid: Paper ID

        Returns:
            True if item was removed, False if not found
        """
        rl_key = readinglist_key(user, pid)
        removed = False

        with get_readinglist_db(flag="c") as rldb:
            if rl_key in rldb:
                del rldb[rl_key]
                removed = True

        if removed:
            # Update index
            with get_readinglist_index_db(flag="c") as idx_db:
                indexed_pids = idx_db.get(user, [])
                if pid in indexed_pids:
                    indexed_pids.remove(pid)
                    idx_db[user] = indexed_pids

        return removed

    @staticmethod
    def update_reading_list_item(user: str, pid: str, updates: dict):
        """
        Update fields in a reading list item.

        Args:
            user: Username
            pid: Paper ID
            updates: Dictionary of fields to update
        """
        rl_key = readinglist_key(user, pid)
        with get_readinglist_db(flag="c") as rldb:
            item = rldb.get(rl_key, {})
            item.update(updates)
            rldb[rl_key] = item


# -----------------------------------------------------------------------------
# Summary Status Repository
# -----------------------------------------------------------------------------


class SummaryStatusRepository:
    """Repository for summary status tracking."""

    @staticmethod
    def get_status(pid: str, model: str = None) -> Optional[dict]:
        """
        Get summary status for a paper.

        Args:
            pid: Paper ID
            model: Model name (optional, uses default if not provided)

        Returns:
            Status data or None if not found
        """
        key = summary_status_key(pid, model)
        with get_summary_status_db() as sdb:
            return sdb.get(key)

    @staticmethod
    def set_status(pid: str, model: str, status: str, error: str = None, **extra):
        """
        Set summary status for a paper.

        Args:
            pid: Paper ID
            model: Model name
            status: Status string (e.g., 'ok', 'error', 'queued')
            error: Error message if any
        """
        key = summary_status_key(pid, model)
        with get_summary_status_db(flag="c") as sdb:
            existing = sdb.get(key, {})
            if not isinstance(existing, dict):
                existing = {}
            updates = {
                "status": status,
                "last_error": error,
                "updated_time": time.time(),
            }
            if extra:
                updates.update(extra)
            for field, value in updates.items():
                if value is None:
                    existing.pop(field, None)
                else:
                    existing[field] = value
            sdb[key] = existing

    @staticmethod
    def update_status(pid: str, model: str, updates: dict):
        """
        Update specific fields in summary status.

        Args:
            pid: Paper ID
            model: Model name
            updates: Dictionary of fields to update
        """
        key = summary_status_key(pid, model)
        with get_summary_status_db(flag="c") as sdb:
            status = sdb.get(key, {})
            status.update(updates)
            sdb[key] = status

    @staticmethod
    def delete_status(pid: str, model: str) -> bool:
        """Delete summary status for a paper."""
        key = summary_status_key(pid, model)
        with get_summary_status_db(flag="c") as sdb:
            if key in sdb:
                del sdb[key]
                return True
            return False

    @staticmethod
    def get_all_items(limit: Optional[int] = None):
        """
        Get all summary status items.

        Args:
            limit: Maximum number of items to return (None for all)

        Returns:
            List of (key, value) tuples
        """
        with get_summary_status_db() as sdb:
            if limit is None or limit <= 0:
                return list(sdb.items())
            out = []
            for i, item in enumerate(sdb.items()):
                if i >= limit:
                    break
                out.append(item)
            return out

    @staticmethod
    def get_items_with_prefix(prefix: str, limit: Optional[int] = None):
        """
        Get all items with a specific key prefix.

        Args:
            prefix: Key prefix to filter by
            limit: Maximum number of items to return (None for all)

        Returns:
            List of (key, value) tuples
        """
        with get_summary_status_db() as sdb:
            if limit is None or limit <= 0:
                return list(sdb.items_with_prefix(prefix))
            out = []
            for i, item in enumerate(sdb.items_with_prefix(prefix)):
                if i >= limit:
                    break
                out.append(item)
            return out

    @staticmethod
    def get_task_status(task_id: str) -> Optional[dict]:
        """
        Get status for a specific task.

        Args:
            task_id: Task ID

        Returns:
            Task status data or None if not found
        """
        key = f"task::{task_id}"
        with get_summary_status_db() as sdb:
            return sdb.get(key)

    @staticmethod
    def set_task_status(task_id: str, status: str, error: Optional[str] = None, **extra):
        """
        Set status for a specific task.

        Args:
            task_id: Task ID
            status: Status string
            error: Error message if any
            extra: Additional fields to store
        """
        key = f"task::{task_id}"
        with get_summary_status_db(flag="c") as sdb:
            existing = sdb.get(key, {})
            if not isinstance(existing, dict):
                existing = {}
            updates = {
                "status": status,
                "error": error,
                "updated_time": time.time(),
            }
            if extra:
                updates.update(extra)
            for field, value in updates.items():
                if value is None:
                    existing.pop(field, None)
                else:
                    existing[field] = value
            sdb[key] = existing

    @staticmethod
    def get_generation_epoch(pid: str, model: str) -> int:
        """Get current cancellation epoch for a (pid, model) pair."""
        pid = (pid or "").strip()
        model = (model or "").strip()
        if not pid or not model:
            return 0
        key = summary_generation_epoch_key(pid, model)
        with get_summary_status_db() as sdb:
            val = sdb.get(key)
        try:
            return int(val or 0)
        except Exception:
            return 0

    @staticmethod
    def bump_generation_epoch(pid: str, model: str) -> int:
        """Increment cancellation epoch for a (pid, model) pair and return new value."""
        pid = (pid or "").strip()
        model = (model or "").strip()
        if not pid or not model:
            return 0
        key = summary_generation_epoch_key(pid, model)
        with get_summary_status_db(flag="c") as sdb:
            cur = sdb.get(key)
            try:
                cur_i = int(cur or 0)
            except Exception:
                cur_i = 0
            cur_i += 1
            sdb[key] = cur_i
            return cur_i


# -----------------------------------------------------------------------------
# User Repository
# -----------------------------------------------------------------------------


class UserRepository:
    """Repository for user-related operations."""

    @staticmethod
    def _coerce_email_list(value) -> List[str]:
        """Coerce legacy/new email value into a normalized list[str]."""
        if value is None:
            return []
        if isinstance(value, str):
            v = value.strip()
            return [v] if v else []
        if isinstance(value, (list, tuple, set)):
            out: List[str] = []
            for item in value:
                if not isinstance(item, str):
                    continue
                v = item.strip()
                if v:
                    out.append(v)
            return out
        return []

    @staticmethod
    def get_email(user: str) -> Optional[str]:
        """Get primary email address for a user (legacy API)."""
        with get_email_db() as edb:
            value = edb.get(user)
        emails = UserRepository._coerce_email_list(value)
        return emails[0] if emails else ""

    @staticmethod
    def get_emails(user: str) -> List[str]:
        """Get all email addresses for a user."""
        with get_email_db() as edb:
            value = edb.get(user)
        return UserRepository._coerce_email_list(value)

    @staticmethod
    def get_all_emails() -> Dict[str, str]:
        """Get primary email addresses for all users (legacy API)."""
        with get_email_db() as edb:
            items = list(edb.items())
        out: Dict[str, str] = {}
        for user, value in items:
            emails = UserRepository._coerce_email_list(value)
            out[user] = emails[0] if emails else ""
        return out

    @staticmethod
    def get_all_email_lists() -> Dict[str, List[str]]:
        """Get email address lists for all users."""
        with get_email_db() as edb:
            items = list(edb.items())
        return {user: UserRepository._coerce_email_list(value) for user, value in items}

    @staticmethod
    def set_email(user: str, email: str):
        """Set primary email address for a user (legacy API)."""
        email = (email or "").strip()
        if email:
            UserRepository.set_emails(user, [email])
        else:
            UserRepository.set_emails(user, [])

    @staticmethod
    def set_emails(user: str, emails: List[str]):
        """Set email addresses for a user."""
        normalized: List[str] = []
        seen = set()
        for email in emails or []:
            if not isinstance(email, str):
                continue
            e = email.strip()
            if not e:
                continue
            key = e.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(e)
        with get_email_db(flag="c") as edb:
            edb[user] = normalized

    @staticmethod
    def get_all_users() -> List[str]:
        """Get list of all users (from tags database)."""
        with get_tags_db() as tdb:
            return list(tdb.keys())


# -----------------------------------------------------------------------------
# Uploaded Paper Repository
# -----------------------------------------------------------------------------


class UploadedPaperRepository:
    """Repository for uploaded paper operations."""

    # Store an owner+sha256 -> pid mapping in the same table for atomic duplicate detection.
    # This avoids introducing a new DB/table and keeps writes transactional.
    _SHA256_KEY_PREFIX = "sha256::"

    @staticmethod
    def sha256_mapping_key(owner: str, sha256: str) -> str:
        """Return the internal mapping key for (owner, sha256)."""
        return f"{UploadedPaperRepository._SHA256_KEY_PREFIX}{owner}::{sha256}"

    @staticmethod
    def get(pid: str) -> Optional[dict]:
        """Get an uploaded paper by PID.

        Args:
            pid: Upload PID (e.g., up_V1StGXR8_Z5j)

        Returns:
            Uploaded paper data or None if not found
        """
        with get_uploaded_papers_db() as updb:
            return updb.get(pid)

    @staticmethod
    def save(pid: str, data: dict):
        """Save or update an uploaded paper.

        Args:
            pid: Upload PID
            data: Paper data dictionary
        """
        with get_uploaded_papers_db(flag="c") as updb:
            updb[pid] = data

    @staticmethod
    def delete(pid: str) -> bool:
        """Delete an uploaded paper.

        Args:
            pid: Upload PID

        Returns:
            True if deleted, False if not found
        """
        with get_uploaded_papers_db(flag="c") as updb:
            try:
                del updb[pid]
                return True
            except KeyError:
                return False

    @staticmethod
    def get_by_owner(owner: str) -> Dict[str, dict]:
        """Get all uploaded papers for a user.

        Args:
            owner: Username

        Returns:
            Dictionary mapping PID to paper data
        """
        # Try index first
        with get_uploaded_papers_index_db() as idx_db:
            indexed_pids = idx_db.get(owner, [])

        if indexed_pids:
            with get_uploaded_papers_db() as updb:
                fetched = updb.get_many(indexed_pids)
                result = {
                    pid: data for pid, data in fetched.items() if isinstance(data, dict) and data.get("owner") == owner
                }

            # Self-heal the index in case it contains stale PIDs (deleted) or
            # mismatched ownership. This keeps counts/quota checks consistent.
            try:
                desired = list(dict.fromkeys([pid for pid in indexed_pids if pid in result]))
                current = list(dict.fromkeys(indexed_pids))
                if desired != current:
                    with get_uploaded_papers_index_db(flag="c") as idx_db:
                        idx_db[owner] = desired
            except Exception:
                pass

            return result

        # Fallback: full scan and rebuild index
        result = {}
        pids = []
        with get_uploaded_papers_db() as updb:
            for pid, data in updb.items():
                if isinstance(data, dict) and data.get("owner") == owner:
                    result[pid] = data
                    pids.append(pid)

        if pids:
            try:
                with get_uploaded_papers_index_db(flag="c") as idx_db:
                    idx_db[owner] = list(dict.fromkeys(pids))
            except Exception:
                pass

        return result

    @staticmethod
    def get_by_sha256(owner: str, sha256: str) -> Optional[Tuple[str, dict]]:
        """Find an uploaded paper by SHA256 hash for a specific owner.

        Args:
            owner: Username
            sha256: File SHA256 hash

        Returns:
            Tuple of (pid, data) if found, None otherwise
        """
        sha_key = UploadedPaperRepository.sha256_mapping_key(owner, sha256)

        # Fast path: use mapping key if present.
        try:
            with get_uploaded_papers_db() as updb:
                mapped_pid = updb.get(sha_key)
        except Exception:
            mapped_pid = None

        if isinstance(mapped_pid, str) and mapped_pid:
            record = UploadedPaperRepository.get(mapped_pid)
            if isinstance(record, dict) and record.get("owner") == owner and record.get("sha256") == sha256:
                return mapped_pid, record

            # Best-effort self-heal: mapping points to missing/mismatched record.
            try:
                with get_uploaded_papers_db(flag="c", autocommit=False) as updb:
                    with updb.transaction():
                        cur = updb.get(sha_key)
                        if cur == mapped_pid:
                            del updb[sha_key]
            except Exception:
                pass

        # Fallback: scan the owner's uploads (bounded by MAX_UPLOADS_PER_USER).
        papers = UploadedPaperRepository.get_by_owner(owner)
        for pid, data in papers.items():
            if data.get("sha256") == sha256:
                # Best-effort: write back mapping for future fast path.
                try:
                    with get_uploaded_papers_db(flag="c", autocommit=False) as updb:
                        with updb.transaction():
                            if not updb.get(sha_key):
                                updb[sha_key] = pid
                except Exception:
                    pass
                return pid, data
        return None

    @staticmethod
    def remove_sha256_mapping(owner: str, sha256: str, pid: str | None = None) -> None:
        """Remove the sha256 mapping key if it currently points to pid (or unconditionally if pid is None)."""
        sha_key = UploadedPaperRepository.sha256_mapping_key(owner, sha256)
        try:
            with get_uploaded_papers_db(flag="c", autocommit=False) as updb:
                with updb.transaction():
                    cur = updb.get(sha_key)
                    if cur is None:
                        return
                    if pid is None or cur == pid:
                        del updb[sha_key]
        except Exception:
            return

    @staticmethod
    def add_to_index(owner: str, pid: str):
        """Add a PID to the user's index.

        Args:
            owner: Username
            pid: Upload PID
        """
        # Use a transaction to avoid lost updates under concurrent uploads.
        with get_uploaded_papers_index_db(flag="c", autocommit=False) as idx_db:
            with idx_db.transaction():
                indexed_pids = idx_db.get(owner, [])
                if pid not in indexed_pids:
                    indexed_pids.append(pid)
                    idx_db[owner] = indexed_pids

    @staticmethod
    def remove_from_index(owner: str, pid: str):
        """Remove a PID from the user's index.

        Args:
            owner: Username
            pid: Upload PID
        """
        with get_uploaded_papers_index_db(flag="c", autocommit=False) as idx_db:
            with idx_db.transaction():
                indexed_pids = idx_db.get(owner, [])
                if pid in indexed_pids:
                    indexed_pids.remove(pid)
                    idx_db[owner] = indexed_pids

    @staticmethod
    def update(pid: str, updates: dict) -> bool:
        """Update specific fields in an uploaded paper.

        Only updates existing records - will NOT create new records if pid doesn't exist.
        This prevents "resurrection" of deleted records during concurrent operations.

        Args:
            pid: Upload PID
            updates: Dictionary of fields to update

        Returns:
            True if record was updated, False if record doesn't exist
        """
        with get_uploaded_papers_db(flag="c") as updb:
            data = updb.get(pid)
            # Only update if record exists and is a valid dict
            if not isinstance(data, dict):
                return False
            data.update(updates)
            data["updated_time"] = time.time()
            updb[pid] = data
            return True

    @staticmethod
    def exists(pid: str) -> bool:
        """Check if an uploaded paper exists.

        Args:
            pid: Upload PID

        Returns:
            True if exists
        """
        with get_uploaded_papers_db() as updb:
            return pid in updb

    @staticmethod
    def count_by_owner(owner: str) -> int:
        """Count uploaded papers for a user.

        Args:
            owner: Username

        Returns:
            Number of uploaded papers
        """
        # Count actual existing papers rather than trusting a potentially stale index.
        return len(UploadedPaperRepository.get_by_owner(owner))
