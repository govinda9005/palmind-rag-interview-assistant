"""Input validation utilities for PalmMind AI backend."""

from datetime import date, datetime, time, timezone
import os
import re
from typing import List, Optional

from pydantic import EmailStr, TypeAdapter, ValidationError

from app.utils.exceptions import (
    BookingValidationException,
    EmptyFileException,
    FileSizeExceededException,
    UnsupportedFileTypeException,
)

# Precompile common email regex pattern for fast secondary check
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

# Supported candidate date string formats
DATE_FORMATS = [
    "%Y-%m-%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
]

# Supported candidate time string formats
TIME_FORMATS = [
    "%H:%M",
    "%H:%M:%S",
    "%I:%M %p",
    "%I:%M%p",
    "%I %p",
    "%I%p",
]


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


def validate_candidate_name(name: str) -> str:
    """Validate candidate full name.

    Args:
        name: Raw candidate name string.

    Returns:
        Cleaned name string.

    Raises:
        BookingValidationException: If name is invalid.
    """
    if not name or not isinstance(name, str):
        raise BookingValidationException("Name cannot be empty.")

    clean_name = name.strip()
    if not clean_name:
        raise BookingValidationException("Name cannot be empty.")

    if len(clean_name) < 2:
        raise BookingValidationException("Candidate name must be at least 2 characters long.")

    if len(clean_name) > 255:
        raise BookingValidationException("Candidate name exceeds maximum length of 255 characters.")

    return clean_name


def validate_candidate_email(email_str: str) -> str:
    """Deterministically validate email address using Pydantic TypeAdapter and regex.

    Args:
        email_str: Raw email string.

    Returns:
        Normalized lowercase email string.

    Raises:
        BookingValidationException: If email is malformed.
    """
    if not email_str or not isinstance(email_str, str):
        raise BookingValidationException("Email address cannot be empty.")

    clean_email = email_str.strip().lower()

    if not EMAIL_REGEX.match(clean_email):
        raise BookingValidationException(
            f"'{clean_email}' is not a valid email address. Expected format: user@domain.com."
        )

    try:
        adapter = TypeAdapter(EmailStr)
        validated = adapter.validate_python(clean_email)
        return str(validated)
    except ValidationError as exc:
        raise BookingValidationException(
            f"'{clean_email}' is not a valid email address.",
            details={"error": str(exc)},
        ) from exc


def parse_and_validate_date(
    date_str: str,
    reference_date: Optional[date] = None,
) -> date:
    """Parse a date string across multiple formats and reject past dates.

    Assumed timezone: Server local / UTC date.

    Args:
        date_str: Date representation (e.g. '2026-11-20', '20 November 2026').
        reference_date: Baseline date to compare for past date check (defaults to current date).

    Returns:
        Parsed datetime.date instance.

    Raises:
        BookingValidationException: If date cannot be parsed or is in the past.
    """
    if not date_str or not isinstance(date_str, str):
        raise BookingValidationException("Interview date cannot be empty.")

    clean_date_str = date_str.strip()
    parsed_date: Optional[date] = None

    for fmt in DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(clean_date_str, fmt).date()
            break
        except ValueError:
            continue

    if parsed_date is None:
        raise BookingValidationException(
            f"Could not parse interview date '{clean_date_str}'. "
            "Please provide a valid date format such as 'YYYY-MM-DD' or '20 November 2026'."
        )

    ref = reference_date or datetime.now(timezone.utc).date()
    if parsed_date < ref:
        raise BookingValidationException(
            f"The selected interview date '{parsed_date.isoformat()}' is in the past. "
            "Please choose a future date."
        )

    return parsed_date


def parse_and_validate_time(time_str: str) -> time:
    """Parse a time string across 12-hour and 24-hour formats and normalize to datetime.time.

    Args:
        time_str: Time representation (e.g. '10:30 AM', '14:30', '10:30').

    Returns:
        Parsed datetime.time instance.

    Raises:
        BookingValidationException: If time cannot be parsed.
    """
    if not time_str or not isinstance(time_str, str):
        raise BookingValidationException("Interview time cannot be empty.")

    clean_time_str = time_str.strip().upper()
    parsed_time: Optional[time] = None

    for fmt in TIME_FORMATS:
        try:
            parsed_time = datetime.strptime(clean_time_str, fmt).time()
            break
        except ValueError:
            continue

    if parsed_time is None:
        raise BookingValidationException(
            f"Could not parse interview time '{clean_time_str}'. "
            "Please provide a valid time format such as '10:30 AM' or '14:00'."
        )

    return parsed_time
