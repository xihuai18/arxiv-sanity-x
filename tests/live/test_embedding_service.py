"""Live tests for embedding service.

Important: keep this file FAST.

The embedding service can be slow or hang under load (model warmup, queueing,
GPU contention, etc.). To avoid the whole test suite getting stuck, we only
perform a lightweight availability check here (port open).

If you want to run deeper embedding/semantic checks, add a separate opt-in test
file guarded by an environment variable.
"""

from __future__ import annotations

from tests.service_detection import is_embedding_service_available


class TestEmbeddingServiceAvailability:
    """Tests for embedding service availability detection."""

    def test_can_detect_embedding_service(self):
        """Test that we can detect embedding service status."""
        result = is_embedding_service_available()
        assert isinstance(result, bool)
        if result:
            print("✓ Embedding service is available")
        else:
            print("✗ Embedding service is not available")
