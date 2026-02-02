"""Unit tests for memory usage parsing fallback."""

from __future__ import annotations

import builtins
import io


def test_get_memory_usage_mb_proc_parsing_is_defensive(monkeypatch):
    import backend.utils.memory_limit as ml

    # Force the function to use the /proc fallback.
    monkeypatch.setattr(ml.resource, "getrusage", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("nope")))

    def fake_open(_path, *_a, **_k):
        # Include a malformed VmRSS line and then a valid one.
        return io.StringIO("VmRSS:\nVmRSS:\t2048 kB\n")

    monkeypatch.setattr(builtins, "open", fake_open)

    assert ml.get_memory_usage_mb() == 2.0
