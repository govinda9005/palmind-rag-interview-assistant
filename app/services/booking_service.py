"""Conversational Interview Booking Service using Gemini and Redis.

Coordinates multi-turn slot filling, deterministic validation in Python,
draft state preservation in Redis, and atomic persistence to SQLite.
"""

from datetime import date, datetime, time, timezone
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from app.api.schemas.booking import BookingDraft, BookingResponse, BookingValidated, RawBookingExtraction
from app.api.schemas.chat import ChatResponse
from app.core.config import Settings, get_settings
from app.repositories.booking_repository import BookingRepository
from app.services.memory_service import MemoryService, get_memory_service
from app.utils.exceptions import BookingValidationException, RAGException
from app.utils.validators import (
    parse_and_validate_date,
    parse_and_validate_time,
    validate_candidate_email,
    validate_candidate_name,
)

logger = logging.getLogger("palmind.booking")

# Assumed Timezone: Assumed server UTC / local date for deterministic validation.
ASSUMED_TIMEZONE = "UTC"

BOOKING_PROMPT_SYSTEM = """You are an interview booking assistant for PalmMind AI.
Analyze the user message and extract interview booking information.
Return a JSON object conforming strictly to this structure:
{
  "intent": "interview_booking" or "document_rag" or "other",
  "name": "full candidate name or null",
  "email": "candidate email address or null",
  "interview_date": "date string as mentioned by user or null",
  "interview_time": "time string as mentioned by user or null"
}
Rules:
- If the user wants to schedule, book, or arrange an interview, intent must be "interview_booking".
- If the user is responding with their name, email, date, or time in a booking context, intent must be "interview_booking".
- If the user asks a question about documents or policies, intent must be "document_rag".
- Do not invent facts or mock values. Use null if not specified.
"""


class BookingService:
    """Manages conversational interview scheduling, validation, and persistence."""

    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        gemini_client: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.memory_service = memory_service or get_memory_service()
        self._gemini_client = gemini_client

    def _get_gemini_client(self) -> Any:
        """Lazy-load or return configured Google GenAI Client."""
        if self._gemini_client is not None:
            return self._gemini_client

        api_key = self.settings.GEMINI_API_KEY
        if not api_key:
            logger.warning("GEMINI_API_KEY not configured. Falling back to heuristic extraction.")
            return None

        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=api_key)
            return self._gemini_client
        except Exception as exc:
            logger.warning("Could not initialize Google GenAI client: %s. Using heuristic fallback.", exc)
            return None

    def is_booking_active(self, session_id: str) -> bool:
        """Check if an active booking draft exists in Redis for this session."""
        draft = self.memory_service.get_booking_draft(session_id)
        return draft is not None and draft.get("status") == "collecting"

    def detect_intent_and_extract(
        self,
        message: str,
        session_id: str,
        current_draft: Optional[Dict[str, Any]] = None,
    ) -> RawBookingExtraction:
        """Extract booking information and classify intent using Gemini or robust heuristic fallback."""
        client = self._get_gemini_client()
        raw = RawBookingExtraction()

        if client is not None:
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    system_instruction=BOOKING_PROMPT_SYSTEM,
                )
                prompt = (
                    f"User message: {message}\n"
                    f"Current draft state: {json.dumps(current_draft or {})}"
                )
                response = client.models.generate_content(
                    model=self.settings.GEMINI_MODEL,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    parsed = json.loads(response.text.strip())
                    raw = RawBookingExtraction(
                        intent=parsed.get("intent", "interview_booking"),
                        name=parsed.get("name"),
                        email=parsed.get("email"),
                        interview_date=parsed.get("interview_date"),
                        interview_time=parsed.get("interview_time"),
                    )
                    logger.info("Gemini structured extraction: %s", raw.model_dump())
                    return raw
            except Exception as exc:
                logger.warning("Gemini booking extraction failed (%s). Falling back to heuristic.", exc)

        # Fallback Heuristic Extraction
        return self._heuristic_extract(message, current_draft)

    def _heuristic_extract(
        self,
        message: str,
        current_draft: Optional[Dict[str, Any]] = None,
    ) -> RawBookingExtraction:
        """Rule-based extractor for testing and offline resilience."""
        msg = message.strip()
        msg_lower = msg.lower()

        # Check intent
        booking_keywords = ["schedule", "book", "interview", "appointment", "slot"]
        has_booking_kw = any(kw in msg_lower for kw in booking_keywords)
        is_active = current_draft is not None and current_draft.get("status") == "collecting"

        intent = "interview_booking" if (has_booking_kw or is_active) else "document_rag"

        name: Optional[str] = None
        email: Optional[str] = None
        interview_date: Optional[str] = None
        interview_time: Optional[str] = None

        # Email regex pattern
        email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", msg)
        if email_match:
            email = email_match.group(0).rstrip(".")

        # Time regex pattern (e.g. 10:30 AM, 14:00, 10:30)
        time_match = re.search(r"\b(\d{1,2}:\d{2}(?:\s*(?:AM|PM|am|pm))?|\d{1,2}\s*(?:AM|PM|am|pm))\b", msg)
        if time_match:
            interview_time = time_match.group(0)

        # Date regex patterns (e.g. 2026-11-20, 20 November 2026, 20/11/2026)
        date_patterns = [
            r"\b(\d{4}-\d{2}-\d{2})\b",
            r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b",
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})\b",
            r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
        ]
        for pat in date_patterns:
            date_match = re.search(pat, msg, re.IGNORECASE)
            if date_match:
                interview_date = date_match.group(0)
                break

        # Slot-filling based on last_field_asked
        last_field = current_draft.get("last_field_asked") if current_draft else None
        if last_field == "name" and not name and not email and not interview_date and not interview_time:
            # Clean possible greeting prefixes
            clean = re.sub(r"^(?:my name is|i am|this is)\s+", "", msg, flags=re.IGNORECASE).strip()
            clean = clean.rstrip(".").strip()
            if len(clean) >= 2 and not any(kw in clean.lower() for kw in ["schedule", "book", "policy", "what"]):
                name = clean

        elif last_field == "email" and not email and not interview_date and not interview_time:
            # Clean any trailing punctuation
            candidate_email = msg.strip().rstrip(".")
            email = candidate_email

        elif last_field == "interview_date" and not interview_date and not email and not interview_time:
            interview_date = msg.strip().rstrip(".")

        elif last_field == "interview_time" and not interview_time and not email and not interview_date:
            interview_time = msg.strip().rstrip(".")

        return RawBookingExtraction(
            intent=intent,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
        )

    def process_message(
        self,
        session_id: str,
        message: str,
        db: Session,
        reference_date: Optional[date] = None,
    ) -> ChatResponse:
        """Process conversational booking turn, slot fill, validate, and persist when complete."""
        raw_draft = self.memory_service.get_booking_draft(session_id)
        if raw_draft is None:
            draft = BookingDraft()
        else:
            draft = BookingDraft(**raw_draft)

        # Extract information from incoming message
        extraction = self.detect_intent_and_extract(
            message=message,
            session_id=session_id,
            current_draft=draft.model_dump(),
        )

        # Handle user changing or updating previously provided fields (Scenario G)
        if extraction.interview_date:
            try:
                valid_date = parse_and_validate_date(extraction.interview_date, reference_date=reference_date)
                draft.interview_date = valid_date.isoformat()
            except BookingValidationException as exc:
                # If the user explicitly provided an invalid date, inform them immediately
                return self._respond_error(session_id, message, exc.message, draft)

        if extraction.email:
            try:
                valid_email = validate_candidate_email(extraction.email)
                draft.email = valid_email
            except BookingValidationException as exc:
                return self._respond_error(session_id, message, exc.message, draft)

        if extraction.interview_time:
            try:
                valid_time = parse_and_validate_time(extraction.interview_time)
                draft.interview_time = valid_time.strftime("%H:%M")
            except BookingValidationException as exc:
                return self._respond_error(session_id, message, exc.message, draft)

        if extraction.name:
            try:
                valid_name = validate_candidate_name(extraction.name)
                draft.name = valid_name
            except BookingValidationException as exc:
                return self._respond_error(session_id, message, exc.message, draft)

        # Process input for the specific field that was requested in the previous turn
        last_field = draft.last_field_asked

        # If user explicitly updated a different field (e.g. changing date while time was asked),
        # do not force-parse the whole message into last_field
        updated_another_field = bool(
            (last_field != "name" and extraction.name)
            or (last_field != "email" and extraction.email)
            or (last_field != "interview_date" and extraction.interview_date)
            or (last_field != "interview_time" and extraction.interview_time)
        )

        if not updated_another_field:
            if last_field == "name" and not draft.name:
                try:
                    valid_name = validate_candidate_name(message)
                    draft.name = valid_name
                except BookingValidationException as exc:
                    return self._respond_error(session_id, message, exc.message, draft)

            elif last_field == "email" and not draft.email:
                try:
                    valid_email = validate_candidate_email(message)
                    draft.email = valid_email
                except BookingValidationException as exc:
                    return self._respond_error(session_id, message, exc.message, draft)

            elif last_field == "interview_date" and not draft.interview_date:
                try:
                    valid_date = parse_and_validate_date(message, reference_date=reference_date)
                    draft.interview_date = valid_date.isoformat()
                except BookingValidationException as exc:
                    return self._respond_error(session_id, message, exc.message, draft)

            elif last_field == "interview_time" and not draft.interview_time:
                try:
                    valid_time = parse_and_validate_time(message)
                    draft.interview_time = valid_time.strftime("%H:%M")
                except BookingValidationException as exc:
                    return self._respond_error(session_id, message, exc.message, draft)

        # Determine next missing field to ask (One field at a time)
        if not draft.name:
            draft.last_field_asked = "name"
            self.memory_service.save_booking_draft(session_id, draft.model_dump())
            reply = "Sure! May I know your full name?"
            self.memory_service.add_turn(session_id, message, reply)
            return ChatResponse(session_id=session_id, response=reply, intent="interview_booking")

        if not draft.email:
            draft.last_field_asked = "email"
            self.memory_service.save_booking_draft(session_id, draft.model_dump())
            reply = "What is your email address?"
            self.memory_service.add_turn(session_id, message, reply)
            return ChatResponse(session_id=session_id, response=reply, intent="interview_booking")

        if not draft.interview_date:
            draft.last_field_asked = "interview_date"
            self.memory_service.save_booking_draft(session_id, draft.model_dump())
            reply = "Which date would you prefer?"
            self.memory_service.add_turn(session_id, message, reply)
            return ChatResponse(session_id=session_id, response=reply, intent="interview_booking")

        if not draft.interview_time:
            draft.last_field_asked = "interview_time"
            self.memory_service.save_booking_draft(session_id, draft.model_dump())
            reply = "What time would you prefer?"
            self.memory_service.add_turn(session_id, message, reply)
            return ChatResponse(session_id=session_id, response=reply, intent="interview_booking")

        # All 4 fields collected! Deterministic final validation
        try:
            date_obj = datetime.strptime(draft.interview_date, "%Y-%m-%d").date()
            time_obj = datetime.strptime(draft.interview_time, "%H:%M").time()
            validated = BookingValidated(
                name=draft.name,
                email=draft.email,
                interview_date=date_obj,
                interview_time=time_obj,
            )
        except Exception as exc:
            logger.error("Final validation failed on draft: %s", exc)
            return self._respond_error(
                session_id,
                message,
                f"Booking validation failed: {str(exc)}",
                draft,
            )

        # Database transaction & Duplicate check
        repo = BookingRepository(db)
        duplicate = repo.find_duplicate(
            email=validated.email,
            interview_date=validated.formatted_date,
            interview_time=validated.formatted_time,
        )

        if duplicate:
            logger.info("Found existing booking for %s on %s at %s", validated.email, validated.formatted_date, validated.formatted_time)
            self.memory_service.clear_booking_draft(session_id)
            reply = (
                f"An interview booking request for {validated.email} on "
                f"{validated.formatted_date} at {validated.formatted_time} has already been recorded."
            )
            self.memory_service.add_turn(session_id, message, reply)
            return ChatResponse(
                session_id=session_id,
                response=reply,
                intent="interview_booking",
                action_info={
                    "booking_id": duplicate.id,
                    "name": duplicate.name,
                    "email": duplicate.email,
                    "interview_date": duplicate.interview_date,
                    "interview_time": duplicate.interview_time,
                    "status": duplicate.status,
                    "duplicate": True,
                },
            )

        try:
            booking = repo.create(
                name=validated.name,
                email=validated.email,
                interview_date=validated.formatted_date,
                interview_time=validated.formatted_time,
                status="confirmed",
            )
            logger.info("Interview booking committed to SQLite: id=%s", booking.id)
        except Exception as exc:
            db.rollback()
            logger.error("Failed to commit booking to database: %s", exc)
            reply = "A database error occurred while recording your interview booking. Please try again."
            return ChatResponse(
                session_id=session_id,
                response=reply,
                intent="interview_booking",
            )

        # Clear draft only after successful database commit
        self.memory_service.clear_booking_draft(session_id)

        # Standard confirmation message
        reply = "Your interview booking request has been recorded successfully."
        self.memory_service.add_turn(session_id, message, reply)

        return ChatResponse(
            session_id=session_id,
            response=reply,
            intent="interview_booking",
            action_info={
                "booking_id": booking.id,
                "name": booking.name,
                "email": booking.email,
                "interview_date": booking.interview_date,
                "interview_time": booking.interview_time,
                "status": booking.status,
            },
        )

    def _respond_error(
        self,
        session_id: str,
        user_message: str,
        error_message: str,
        draft: BookingDraft,
    ) -> ChatResponse:
        """Preserve draft and prompt user with the validation error."""
        self.memory_service.save_booking_draft(session_id, draft.model_dump())
        self.memory_service.add_turn(session_id, user_message, error_message)
        return ChatResponse(
            session_id=session_id,
            response=error_message,
            intent="interview_booking",
        )


# Singleton instance for application reuse
booking_service = BookingService()


def get_booking_service() -> BookingService:
    """FastAPI dependency provider for BookingService."""
    return booking_service
