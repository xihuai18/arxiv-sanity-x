"""Unit tests for SSE (Server-Sent Events) utilities."""

from __future__ import annotations

import queue


class TestRegisterUserStream:
    """Tests for register_user_stream function."""

    def test_register_user_stream_returns_queue(self):
        """Test that register_user_stream returns a Queue."""
        from backend.utils.sse import register_user_stream, unregister_user_stream

        q = register_user_stream("test_user")
        try:
            assert isinstance(q, queue.Queue)
        finally:
            unregister_user_stream("test_user", q)

    def test_register_multiple_streams_for_same_user(self):
        """Test registering multiple streams for the same user."""
        from backend.utils.sse import register_user_stream, unregister_user_stream

        q1 = register_user_stream("test_user")
        q2 = register_user_stream("test_user")
        try:
            assert isinstance(q1, queue.Queue)
            assert isinstance(q2, queue.Queue)
            assert q1 is not q2  # Should be different queues
        finally:
            unregister_user_stream("test_user", q1)
            unregister_user_stream("test_user", q2)


class TestUnregisterUserStream:
    """Tests for unregister_user_stream function."""

    def test_unregister_user_stream_no_error(self):
        """Test that unregister_user_stream doesn't raise errors."""
        from backend.utils.sse import register_user_stream, unregister_user_stream

        q = register_user_stream("test_user")
        # Should not raise any errors
        unregister_user_stream("test_user", q)

    def test_unregister_nonexistent_stream_no_error(self):
        """Test that unregistering a non-existent stream doesn't raise errors."""
        from backend.utils.sse import unregister_user_stream

        q = queue.Queue()
        # Should not raise any errors
        unregister_user_stream("nonexistent_user", q)


class TestEmitUserEvent:
    """Tests for emit_user_event function."""

    def test_emit_user_event_to_registered_user(self):
        """Test emitting event to a registered user."""
        from backend.utils.sse import (
            emit_user_event,
            register_user_stream,
            unregister_user_stream,
        )

        q = register_user_stream("test_user")
        try:
            emit_user_event("test_user", {"type": "test", "data": "hello"})

            # Check if event was received
            event = q.get(timeout=1)
            assert isinstance(event, dict)
            assert event["type"] == "test"
            assert event["data"] == "hello"
            assert "ts" in event
        finally:
            unregister_user_stream("test_user", q)

    def test_emit_user_event_to_nonexistent_user_no_error(self):
        """Test that emitting to non-existent user doesn't raise errors."""
        from backend.utils.sse import emit_user_event

        # Should not raise any errors
        emit_user_event("nonexistent_user", {"type": "test"})

    def test_emit_user_event_with_none_user(self):
        """Test emitting event with None user."""
        from backend.utils.sse import emit_user_event

        # Should not raise any errors
        emit_user_event(None, {"type": "test"})


class TestEmitAllEvent:
    """Tests for emit_all_event function."""

    def test_emit_all_event_no_error(self):
        """Test that emit_all_event doesn't raise errors."""
        from backend.utils.sse import emit_all_event

        # Should not raise any errors
        emit_all_event({"type": "broadcast", "data": "hello all"})

    def test_emit_all_event_reaches_registered_users(self):
        """Test that emit_all_event reaches registered users."""
        from backend.utils.sse import (
            emit_all_event,
            register_user_stream,
            unregister_user_stream,
        )

        q = register_user_stream("test_user")
        try:
            emit_all_event({"type": "broadcast", "data": "hello"})

            # Check if event was received
            event = q.get(timeout=1)
            assert isinstance(event, dict)
            assert event["type"] == "broadcast"
            assert event["data"] == "hello"
            assert "ts" in event
        finally:
            unregister_user_stream("test_user", q)
