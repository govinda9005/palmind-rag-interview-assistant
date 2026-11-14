"""Input validation utilities for PalmMind AI backend."""

import os
from typing import List
from app.utils.exceptions import (
    EmptyFileException,
    FileSizeExceededException,
    UnsupportedFileTypeException,
)


def validate_file_upload(
    filename: str,
    file_bytes: bytes,
    allowed_extensions: List[str],
    max_size_mb: float,
) -> str:
    """Validate uploaded file extension, non-emptiness, and size limits.

    Returns the normalized file extension (e.g. 'pdf' or 'txt').
    """
    if not filename or not filename.strip():
        raise UnsupportedFileTypeException(file_type="unknown", allowed=allowed_extensions)

    _, ext = os.path.splitext(filename)
    clean_ext = ext.lstrip(".").lower()

    if clean_ext not in [e.lower() for e in allowed_extensions]:
        raise UnsupportedFileTypeException(file_type=clean_ext, allowed=allowed_extensions)

    if not file_bytes or len(file_bytes) == 0:
        raise EmptyFileException(filename=filename)

    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise FileSizeExceededException(
            filename=filename,
            file_size_mb=file_size_mb,
            max_size_mb=max_size_mb,
        )

    return clean_ext

