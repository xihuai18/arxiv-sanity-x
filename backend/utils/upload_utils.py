"""
Upload utility functions for uploaded papers feature.

This module provides helper functions for handling uploaded paper IDs
and related operations.
"""

import hashlib
import re
from pathlib import Path

# Upload PID format: up_<nanoid-12>
# Character set: [a-zA-Z0-9_-]
# Total length: 15 characters (up_ + 12 char nanoid)
UPLOAD_PID_PREFIX = "up_"
UPLOAD_PID_PATTERN = re.compile(r"^up_[a-zA-Z0-9_-]{12}$")


def is_upload_pid(pid: str) -> bool:
    """Check if pid is an uploaded paper ID.

    Args:
        pid: Paper ID to check

    Returns:
        True if pid starts with 'up_' prefix
    """
    return bool(pid and pid.startswith(UPLOAD_PID_PREFIX))


def validate_upload_pid(pid: str) -> bool:
    """Validate upload PID format strictly.

    Args:
        pid: Paper ID to validate

    Returns:
        True if pid matches the exact upload PID format
    """
    return bool(pid and UPLOAD_PID_PATTERN.match(pid))


def generate_upload_pid() -> str:
    """Generate a new upload PID using nanoid.

    Returns:
        A new upload PID in format up_<nanoid-12>
    """
    try:
        from nanoid import generate

        # Use URL-safe alphabet
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-"
        nanoid = generate(alphabet, 12)
        return f"{UPLOAD_PID_PREFIX}{nanoid}"
    except ImportError:
        # Fallback to secrets if nanoid not available
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits + "_-"
        nanoid = "".join(secrets.choice(alphabet) for _ in range(12))
        return f"{UPLOAD_PID_PREFIX}{nanoid}"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA256 hash of bytes.

    Args:
        data: Bytes to hash

    Returns:
        Hex-encoded SHA256 hash
    """
    return hashlib.sha256(data).hexdigest()


def get_upload_pdf_path(pid: str, data_dir: str) -> Path:
    """Get the path to an uploaded PDF file.

    Args:
        pid: Upload PID
        data_dir: Base data directory

    Returns:
        Path to the PDF file
    """
    return Path(data_dir) / "uploads" / pid / "original.pdf"


def get_upload_dir(pid: str, data_dir: str) -> Path:
    """Get the upload directory for a PID.

    Args:
        pid: Upload PID
        data_dir: Base data directory

    Returns:
        Path to the upload directory
    """
    return Path(data_dir) / "uploads" / pid


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe storage.

    Removes:
    - Path separators (/, \\)
    - Null bytes
    - Parent directory references (..)
    - Control characters (ASCII 0x00-0x1F, 0x7F)
    - Carriage return and line feed

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove control characters (ASCII 0x00-0x1F and 0x7F) including \r\n
    filename = re.sub(r"[\x00-\x1f\x7f]", "", filename)
    # Remove path separators, null bytes, and parent directory references
    filename = filename.replace("/", "_").replace("\\", "_").replace("\x00", "")
    filename = filename.replace("..", "_")
    # Remove leading/trailing whitespace and dots
    filename = filename.strip().strip(".")
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        max_name_len = 255 - len(ext) - 1 if ext else 255
        filename = name[:max_name_len] + ("." + ext if ext else "")
    return filename or "unnamed"
