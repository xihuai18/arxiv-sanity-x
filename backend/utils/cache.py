"""Small in-memory cache helpers."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict


class LRUCacheTTL:
    def __init__(self, maxsize: int = 256, ttl_s: float = 120.0):
        self._maxsize = maxsize
        self._ttl_s = ttl_s
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if now - ts > self._ttl_s:
                try:
                    del self._data[key]
                except KeyError:
                    pass
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key, value):
        now = time.time()
        with self._lock:
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)
