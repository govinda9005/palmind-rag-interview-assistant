"""Pydantic models for structured interview booking."""

from datetime import date, time
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class BookingExtraction(BaseModel):
    """Structured booking details extracted from conversation turns."""

    name: Optional[str] = Field(default=None, description="Full name of the candidate")
    email: Optional[EmailStr] = Field(default=None, description="Candidate email address")
    interview_date: Optional[date] = Field(default=None, description="Target interview date (YYYY-MM-DD)")
    interview_time: Optional[time] = Field(default=None, description="Target interview time (HH:MM)")


class BookingResponse(BaseModel):
    """Booking confirmation schema."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    interview_date: date
    interview_time: time
    status: str

