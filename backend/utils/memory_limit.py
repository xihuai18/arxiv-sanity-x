"""
Memory limit utilities for preventing OOM (Out of Memory) issues.

Usage:
    from backend.utils.memory_limit import set_memory_limit, check_memory_usage

    # Set memory limit (MB)
    set_memory_limit(2048)

    # Check current memory usage
    usage_mb = check_memory_usage()
"""

from __future__ import annotations

import os
import resource
import sys
from typing import Callable

from loguru import logger


def set_memory_limit(max_mb: int) -> bool:
    """Set memory limit for current process using resource.setrlimit.

    Args:
        max_mb: Maximum memory in MB. 0 or negative means no limit.

    Returns:
        True if limit was set successfully, False otherwise.
    """
    if max_mb <= 0:
        return False

    try:
        max_bytes = max_mb * 1024 * 1024
        # RLIMIT_AS limits the total address space (virtual memory)
        # This is more reliable than RLIMIT_DATA for Python processes
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # Set both soft and hard limits
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        logger.info(f"Memory limit set to {max_mb}MB (was soft={soft}, hard={hard})")
        return True
    except (ValueError, resource.error) as e:
        logger.warning(f"Failed to set memory limit: {e}")
        return False
    except AttributeError:
        # Windows doesn't support RLIMIT_AS
        logger.warning("Memory limit not supported on this platform")
        return False


def get_memory_usage_mb() -> float:
    """Get current memory usage of this process in MB.

    Returns:
        Memory usage in MB, or -1 if unable to determine.
    """
    try:
        # Try using resource module (Unix)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        else:
            return usage.ru_maxrss / 1024
    except Exception:
        pass

    try:
        # Fallback: read from /proc on Linux
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # VmRSS is in KB
                    return int(line.split()[1]) / 1024
    except Exception:
        pass

    return -1


def check_memory_and_exit_if_exceeded(max_mb: int, exit_code: int = 137) -> None:
    """Check memory usage and exit if exceeded.

    This is useful for worker processes that should be restarted when
    memory usage is too high.

    Args:
        max_mb: Maximum memory in MB.
        exit_code: Exit code to use (137 = killed by signal, mimics OOM killer).
    """
    if max_mb <= 0:
        return

    usage = get_memory_usage_mb()
    if usage > 0 and usage > max_mb:
        logger.warning(f"Memory usage {usage:.1f}MB exceeds limit {max_mb}MB, exiting")
        os._exit(exit_code)


def create_memory_check_callback(max_mb: int) -> Callable[[], None] | None:
    """Create a callback function for periodic memory checking.

    Args:
        max_mb: Maximum memory in MB. 0 means no limit.

    Returns:
        Callback function or None if no limit.
    """
    if max_mb <= 0:
        return None

    def check_memory():
        check_memory_and_exit_if_exceeded(max_mb)

    return check_memory
