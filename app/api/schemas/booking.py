"""Pydantic schemas for structured interview booking."""

from datetime import date, datetime, time
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RawBookingExtraction(BaseModel):
    """Raw structured booking information extracted from user utterances by LLM."""

    intent: Optional[str] = Field(
        default="interview_booking",
        description="Detected user intent: 'interview_booking', 'document_rag', or 'other'",
    )
    name: Optional[str] = Field(default=None, description="Full name of the candidate")
    email: Optional[str] = Field(default=None, description="Candidate email address")
    interview_date: Optional[str] = Field(
        default=None,
        description="Interview date mentioned by user (e.g. YYYY-MM-DD or '20 November 2026')",
    )
    interview_time: Optional[str] = Field(
        default=None,
        description="Interview time mentioned by user (e.g. HH:MM or '10:30 AM')",
    )


class BookingDraft(BaseModel):
    """Temporary in-progress booking state preserved across turns in Redis."""

    name: Optional[str] = None
    email: Optional[str] = None
    interview_date: Optional[str] = None  # Normalized YYYY-MM-DD
    interview_time: Optional[str] = None  # Normalized HH:MM
    status: str = "collecting"  # "collecting", "confirmed"
    last_field_asked: Optional[str] = None


class BookingValidated(BaseModel):
    """Strictly validated booking payload ready for SQLite persistence."""

    name: str = Field(..., min_length=2, max_length=255)
    email: EmailStr
    interview_date: date
    interview_time: time

    @property
    def formatted_date(self) -> str:
        return self.interview_date.isoformat()

    @property
    def formatted_time(self) -> str:
        return self.interview_time.strftime("%H:%M")


class BookingResponse(BaseModel):
    """Structured response representing a confirmed interview booking."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    interview_date: str
    interview_time: str
    status: str
    created_at: Optional[datetime] = None
