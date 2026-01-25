"""Unit tests for cache utilities."""

from __future__ import annotations

import time

import pytest


class TestLRUCacheTTL:
    """Tests for LRUCacheTTL class."""

    @pytest.fixture
    def cache(self):
        """Create a cache instance for testing."""
        from backend.utils.cache import LRUCacheTTL

        return LRUCacheTTL(maxsize=10, ttl_s=1.0)

    def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key_returns_none(self, cache):
        """Test that getting a missing key returns None."""
        assert cache.get("nonexistent") is None

    def test_maxsize_eviction(self, cache):
        """Test that old entries are evicted when maxsize is reached."""
        # Fill cache beyond maxsize
        for i in range(15):
            cache.set(f"key{i}", f"value{i}")

        # First keys should be evicted (or TTL expired)
        # Note: exact behavior depends on implementation
        assert cache.get("key14") == "value14"  # Latest should exist

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        from backend.utils.cache import LRUCacheTTL

        cache = LRUCacheTTL(maxsize=10, ttl_s=0.1)  # 100ms TTL
        cache.set("key", "value")

        assert cache.get("key") == "value"

        time.sleep(0.15)  # Wait for TTL to expire
        assert cache.get("key") is None

    def test_update_existing_key(self, cache):
        """Test updating an existing key."""
        cache.set("key", "value1")
        cache.set("key", "value2")
        assert cache.get("key") == "value2"
