"""SatQuery AI — Security utilities.

File validation, filename sanitization, and upload security checks.
"""

import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from app.utils.logging import get_logger

logger = get_logger("security")

# Magic bytes for supported file formats
MAGIC_BYTES = {
    ".tif": [b"\x49\x49\x2A\x00", b"\x4D\x4D\x00\x2A"],  # TIFF (little/big endian)
    ".tiff": [b"\x49\x49\x2A\x00", b"\x4D\x4D\x00\x2A"],
    ".png": [b"\x89\x50\x4E\x47"],
    ".jpg": [b"\xFF\xD8\xFF"],
    ".jpeg": [b"\xFF\xD8\xFF"],
}


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a filename to prevent path traversal and special character issues.

    Args:
        filename: Original filename from upload.
        max_length: Maximum allowed filename length.

    Returns:
        Sanitized filename with UUID prefix.
    """
    # Extract basename only (prevent path traversal)
    basename = os.path.basename(filename)

    # Remove any non-alphanumeric characters except dots, hyphens, underscores
    safe_name = re.sub(r"[^\w\-.]", "_", basename)

    # Remove leading dots (prevent hidden files)
    safe_name = safe_name.lstrip(".")

    # Truncate if too long (preserve extension)
    name, ext = os.path.splitext(safe_name)
    if len(safe_name) > max_length:
        name = name[: max_length - len(ext) - 37]  # 37 = UUID + underscore

    # Prefix with UUID for uniqueness
    unique_name = f"{uuid.uuid4().hex[:12]}_{name}{ext}"

    return unique_name


def validate_file_extension(
    filename: str, allowed_extensions: List[str]
) -> Tuple[bool, str]:
    """Validate that a file has an allowed extension.

    Args:
        filename: Filename to check.
        allowed_extensions: List of allowed extensions (e.g., ['.tif', '.png']).

    Returns:
        Tuple of (is_valid, message).
    """
    _, ext = os.path.splitext(filename.lower())

    if ext not in allowed_extensions:
        return False, (
            f"Unsupported file format: {ext}. "
            f"Accepted formats: {', '.join(allowed_extensions)}"
        )
    return True, "Valid file extension."


def validate_file_size(
    file_size_bytes: int, max_size_mb: int
) -> Tuple[bool, str]:
    """Validate that a file does not exceed the size limit.

    Args:
        file_size_bytes: File size in bytes.
        max_size_mb: Maximum allowed size in megabytes.

    Returns:
        Tuple of (is_valid, message).
    """
    max_bytes = max_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        size_mb = file_size_bytes / (1024 * 1024)
        return False, (
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed size ({max_size_mb} MB)."
        )
    return True, "Valid file size."


def validate_magic_bytes(
    file_path: Path, expected_extension: str
) -> Tuple[bool, str]:
    """Validate file magic bytes match the expected format.

    Args:
        file_path: Path to the file.
        expected_extension: Expected file extension.

    Returns:
        Tuple of (is_valid, message).
    """
    ext = expected_extension.lower()
    if ext not in MAGIC_BYTES:
        # No magic byte check available for this extension
        return True, "No magic byte validation available for this format."

    try:
        with open(file_path, "rb") as f:
            header = f.read(8)

        expected_headers = MAGIC_BYTES[ext]
        for expected in expected_headers:
            if header[: len(expected)] == expected:
                return True, "Magic bytes match expected format."

        return False, (
            f"File content does not match expected format ({ext}). "
            "The file may be corrupted or mislabelled."
        )
    except Exception as e:
        logger.error("magic_byte_validation_error", error=str(e), path=str(file_path))
        return False, f"Could not validate file content: {str(e)}"
