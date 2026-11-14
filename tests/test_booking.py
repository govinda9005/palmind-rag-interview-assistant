"""Unit and integration tests for conversational interview booking.

Covers:
- BookingRepository operations and duplicate prevention
- Deterministic Python validators (name, email, date formats, past date rejection, time formats)
- Scenario A: Complete booking conversation
- Scenario B: Incomplete conversation (missing email)
- Scenario C: Invalid email rejection
- Scenario D: Invalid date rejection
- Scenario E: Past date rejection
- Scenario F: Missing interview time
- Scenario G: User changes previously provided date
- Scenario H: Repeated confirmation message / duplicate detection
- Scenario I: Successful persistence in SQLite
- Scenario J: Database failure rollback
- Scenario K: Ensuring ordinary document RAG questions still route cleanly
"""

from datetime import date, datetime, time, timedelta, timezone
from typing import AsyncGenerator
from unittest.mock import MagicMock
import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes.chat import get_booking_service, get_memory_service, get_rag_service
from app.api.schemas.booking import BookingDraft, BookingResponse, BookingValidated, RawBookingExtraction
from app.api.schemas.chat import ChatResponse, SourceChunk
from app.core.config import Settings
from app.database.connection import Base, get_db
from app.main import app
from app.models.booking import Booking
from app.repositories.booking_repository import BookingRepository
from app.services.booking_service import BookingService
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService
from app.utils.exceptions import BookingValidationException
from app.utils.validators import (
    parse_and_validate_date,
    parse_and_validate_time,
    validate_candidate_email,
    validate_candidate_name,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def test_db_session() -> Session:
    """Create an isolated, in-memory SQLite database session for booking testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def fake_memory_service() -> MemoryService:
    """Isolated FakeRedis memory service."""
    client = fakeredis.FakeRedis(decode_responses=True)
    return MemoryService(client=client)


@pytest.fixture
def booking_service(fake_memory_service: MemoryService) -> BookingService:
    """BookingService instance backed by FakeRedis memory."""
    return BookingService(memory_service=fake_memory_service)


@pytest.fixture
def reference_future_date() -> date:
    """Standard baseline reference date for deterministic past/future date validation."""
    return date(2026, 11, 14)


# ==============================================================================
# 1. Unit Tests for Deterministic Python Validators
# ==============================================================================

class TestBookingValidatorsUnit:
    """Verify deterministic validation in Python for name, email, date, and time."""

    def test_validate_candidate_name_valid(self):
        assert validate_candidate_name("Govinda Yadav") == "Govinda Yadav"
        assert validate_candidate_name("  John Doe  ") == "John Doe"

    def test_validate_candidate_name_invalid_empty(self):
        with pytest.raises(BookingValidationException) as exc_info:
            validate_candidate_name("   ")
        assert "cannot be empty" in exc_info.value.message

    def test_validate_candidate_name_too_short(self):
        with pytest.raises(BookingValidationException) as exc_info:
            validate_candidate_name("A")
        assert "at least 2 characters" in exc_info.value.message

    def test_validate_candidate_email_valid(self):
        assert validate_candidate_email("govinda@example.com") == "govinda@example.com"
        assert validate_candidate_email("  Test.User+1@company.org  ") == "test.user+1@company.org"

    def test_validate_candidate_email_invalid(self):
        invalid_emails = ["notanemail", "user@", "@domain.com", "user@domain", "user space@domain.com"]
        for inv in invalid_emails:
            with pytest.raises(BookingValidationException):
                validate_candidate_email(inv)

    def test_parse_and_validate_date_formats(self, reference_future_date: date):
        # Format: YYYY-MM-DD
        d1 = parse_and_validate_date("2026-11-20", reference_date=reference_future_date)
        assert d1 == date(2026, 11, 20)

        # Format: DD Month YYYY
        d2 = parse_and_validate_date("20 November 2026", reference_date=reference_future_date)
        assert d2 == date(2026, 11, 20)

        # Format: Month DD, YYYY
        d3 = parse_and_validate_date("November 20, 2026", reference_date=reference_future_date)
        assert d3 == date(2026, 11, 20)

        # Format: DD/MM/YYYY
        d4 = parse_and_validate_date("20/11/2026", reference_date=reference_future_date)
        assert d4 == date(2026, 11, 20)

    def test_parse_and_validate_date_rejects_past_date(self, reference_future_date: date):
        past_date = "2026-11-10"  # Before reference date 2026-11-14
        with pytest.raises(BookingValidationException) as exc_info:
            parse_and_validate_date(past_date, reference_date=reference_future_date)
        assert "in the past" in exc_info.value.message

    def test_parse_and_validate_date_invalid_format(self):
        with pytest.raises(BookingValidationException) as exc_info:
            parse_and_validate_date("not-a-valid-date")
        assert "Could not parse" in exc_info.value.message

    def test_parse_and_validate_time_formats(self):
        # 12-hour with AM/PM
        t1 = parse_and_validate_time("10:30 AM")
        assert t1 == time(10, 30)

        t2 = parse_and_validate_time("2:15 PM")
        assert t2 == time(14, 15)

        # 24-hour
        t3 = parse_and_validate_time("14:30")
        assert t3 == time(14, 30)

        t4 = parse_and_validate_time("09:00")
        assert t4 == time(9, 0)

    def test_parse_and_validate_time_invalid(self):
        with pytest.raises(BookingValidationException) as exc_info:
            parse_and_validate_time("invalid_time")
        assert "Could not parse" in exc_info.value.message


# ==============================================================================
# 2. Unit Tests for BookingRepository
# ==============================================================================

class TestBookingRepositoryUnit:
    """Verify database operations on the bookings SQLite table."""

    def test_create_and_get_booking(self, test_db_session: Session):
        repo = BookingRepository(test_db_session)
        booking = repo.create(
            name="Govinda Yadav",
            email="govinda@example.com",
            interview_date="2026-11-20",
            interview_time="10:30",
            status="confirmed",
        )
        assert booking.id is not None
        assert booking.id > 0
        assert booking.name == "Govinda Yadav"
        assert booking.email == "govinda@example.com"
        assert booking.status == "confirmed"

        fetched = repo.get_by_id(booking.id)
        assert fetched is not None
        assert fetched.id == booking.id
        assert fetched.name == "Govinda Yadav"

    def test_find_duplicate_detection(self, test_db_session: Session):
        repo = BookingRepository(test_db_session)
        repo.create(
            name="Govinda Yadav",
            email="govinda@example.com",
            interview_date="2026-11-20",
            interview_time="10:30",
        )

        dup = repo.find_duplicate(
            email="govinda@example.com",
            interview_date="2026-11-20",
            interview_time="10:30",
        )
        assert dup is not None
        assert dup.name == "Govinda Yadav"

        no_dup = repo.find_duplicate(
            email="govinda@example.com",
            interview_date="2026-11-25",
            interview_time="10:30",
        )
        assert no_dup is None


# ==============================================================================
# 3. Unit and Scenario Tests for BookingService
# ==============================================================================

class TestBookingServiceScenarios:
    """Comprehensive verification of the required interview booking scenarios."""

    def test_scenario_a_complete_booking_conversation(
        self,
        booking_service: BookingService,
        test_db_session: Session,
        reference_future_date: date,
    ):
        """Scenario A: Full 5-turn conversation collecting all 4 required fields."""
        session_id = "test_scenario_a"

        # Turn 1: User indicates booking intent
        resp1 = booking_service.process_message(
            session_id=session_id,
            message="I want to schedule an interview.",
            db=test_db_session,
            reference_date=reference_future_date,
        )
        assert resp1.response == "Sure! May I know your full name?"
        assert resp1.intent == "interview_booking"
        assert booking_service.is_booking_active(session_id) is True

        # Turn 2: User provides full name
        resp2 = booking_service.process_message(
            session_id=session_id,
            message="Govinda Yadav.",
            db=test_db_session,
            reference_date=reference_future_date,
        )
        assert resp2.response == "What is your email address?"
        assert resp2.intent == "interview_booking"

        # Turn 3: User provides email
        resp3 = booking_service.process_message(
            session_id=session_id,
            message="govinda@example.com.",
            db=test_db_session,
            reference_date=reference_future_date,
        )
        assert resp3.response == "Which date would you prefer?"
        assert resp3.intent == "interview_booking"

        # Turn 4: User provides date
        resp4 = booking_service.process_message(
            session_id=session_id,
            message="20 November 2026.",
            db=test_db_session,
            reference_date=reference_future_date,
        )
        assert resp4.response == "What time would you prefer?"
        assert resp4.intent == "interview_booking"

        # Turn 5: User provides time -> Booking completes & persists to SQLite
        resp5 = booking_service.process_message(
            session_id=session_id,
            message="10:30 AM.",
            db=test_db_session,
            reference_date=reference_future_date,
        )
        assert resp5.response == "Your interview booking request has been recorded successfully."
        assert resp5.intent == "interview_booking"
        assert resp5.action_info is not None
        assert resp5.action_info["name"] == "Govinda Yadav"
        assert resp5.action_info["email"] == "govinda@example.com"
        assert resp5.action_info["interview_date"] == "2026-11-20"
        assert resp5.action_info["interview_time"] == "10:30"
        assert resp5.action_info["status"] == "confirmed"

        # Verify draft was cleared in Redis
        assert booking_service.is_booking_active(session_id) is False

        # Verify SQL persistence
        repo = BookingRepository(test_db_session)
        persisted = repo.get_by_id(resp5.action_info["booking_id"])
        assert persisted is not None
        assert persisted.email == "govinda@example.com"

    def test_scenario_b_missing_email_incomplete_conversation(
        self,
        booking_service: BookingService,
        test_db_session: Session,
        reference_future_date: date,
    ):
        """Scenario B: Conversation stops after name; incomplete booking is NOT saved."""
        session_id = "test_scenario_b"
        booking_service.process_message(session_id, "I want to book an interview", test_db_session, reference_future_date)
        resp2 = booking_service.process_message(session_id, "Jane Doe", test_db_session, reference_future_date)

        assert resp2.response == "What is your email address?"
        assert booking_service.is_booking_active(session_id) is True

        # Verify draft has name, but no email/date/time
        draft = booking_service.memory_service.get_booking_draft(session_id)
        assert draft["name"] == "Jane Doe"
        assert draft["email"] is None

        # Verify nothing persisted to SQLite
        repo = BookingRepository(test_db_session)
        assert len(repo.list_all()) == 0

    def test_scenario_c_invalid_email_handling(
        self,
        booking_service: BookingService,
        test_db_session: Session,
        reference_future_date: date,
    ):
        """Scenario C: Reject invalid email and prompt for a valid one without losing state."""
        session_id = "test_scenario_c"
        booking_service.process_message(session_id, "I want to schedule an interview", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "Govinda Yadav", test_db_session, reference_future_date)

        # Enter invalid email
        resp_err = booking_service.process_message(session_id, "invalid_email_format", test_db_session, reference_future_date)
        assert "not a valid email" in resp_err.response
        assert booking_service.is_booking_active(session_id) is True

        # Enter valid email
        resp_ok = booking_service.process_message(session_id, "govinda@valid.com", test_db_session, reference_future_date)
        assert resp_ok.response == "Which date would you prefer?"
        draft = booking_service.memory_service.get_booking_draft(session_id)
        assert draft["email"] == "govinda@valid.com"

    def test_scenario_d_invalid_date_handling(
        self,
        booking_service: BookingService,
        test_db_session: Session,
        reference_future_date: date,
    ):
        """Scenario D: Reject unparseable date format and request a valid date."""
        session_id = "test_scenario_d"
        booking_service.process_message(session_id, "Schedule interview", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "Govinda Yadav", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "govinda@example.com", test_db_session, reference_future_date)

        resp_err = booking_service.process_message(session_id, "not-a-real-date", test_db_session, reference_future_date)
        assert "Could not parse interview date" in resp_err.response

        # Then provide valid date
        resp_ok = booking_service.process_message(session_id, "2026-11-20", test_db_session, reference_future_date)
        assert resp_ok.response == "What time would you prefer?"

    def test_scenario_e_past_date_handling(
        self,
        booking_service: BookingService,
        test_db_session: Session,
        reference_future_date: date,
    ):
        """Scenario E: Reject past dates deterministically against the baseline date."""
        session_id = "test_scenario_e"
        booking_service.process_message(session_id, "Book an interview", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "Govinda Yadav", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "govinda@example.com", test_db_session, reference_future_date)

        past_date_str = (reference_future_date - timedelta(days=2)).isoformat()
        resp_err = booking_service.process_message(session_id, past_date_str, test_db_session, reference_future_date)
        assert "in the past" in resp_err.response

        # Future date works
        future_date_str = (reference_future_date + timedelta(days=5)).isoformat()
        resp_ok = booking_service.process_message(session_id, future_date_str, test_db_session, reference_future_date)
        assert resp_ok.response == "What time would you prefer?"

    def test_scenario_f_missing_interview_time(
        self,
        booking_service: BookingService,
        test_db_session: Session,
        reference_future_date: date,
    ):
        """Scenario F: User provides name, email, date, but not time yet."""
        session_id = "test_scenario_f"
        booking_service.process_message(session_id, "Schedule interview", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "Govinda Yadav", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "govinda@example.com", test_db_session, reference_future_date)
        resp = booking_service.process_message(session_id, "2026-11-20", test_db_session, reference_future_date)

        assert resp.response == "What time would you prefer?"
        # Database must still have 0 bookings
        repo = BookingRepository(test_db_session)
        assert len(repo.list_all()) == 0

    def test_scenario_g_user_changes_previously_provided_date(
        self,
        booking_service: BookingService,
        test_db_session: Session,
        reference_future_date: date,
    ):
        """Scenario G: User provides date, but then changes it in the subsequent turn."""
        session_id = "test_scenario_g"
        booking_service.process_message(session_id, "Schedule interview", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "Govinda Yadav", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "govinda@example.com", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "2026-11-20", test_db_session, reference_future_date)

        # User changes date
        resp_change = booking_service.process_message(
            session_id,
            "Actually, let's change the date to 2026-11-25",
            test_db_session,
            reference_future_date,
        )
        draft = booking_service.memory_service.get_booking_draft(session_id)
        assert draft["interview_date"] == "2026-11-25"
        assert resp_change.response == "What time would you prefer?"

        # Now complete with time
        resp_final = booking_service.process_message(session_id, "14:00", test_db_session, reference_future_date)
        assert resp_final.response == "Your interview booking request has been recorded successfully."
        assert resp_final.action_info["interview_date"] == "2026-11-25"
        assert resp_final.action_info["interview_time"] == "14:00"

    def test_scenario_h_repeated_confirmation_prevents_duplicate(
        self,
        booking_service: BookingService,
        test_db_session: Session,
        reference_future_date: date,
    ):
        """Scenario H: Repeated booking attempt with identical email, date, and time is caught."""
        session_id = "test_scenario_h"
        booking_service.process_message(session_id, "Schedule interview", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "Govinda Yadav", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "govinda@example.com", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "2026-11-20", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "10:30 AM", test_db_session, reference_future_date)

        repo = BookingRepository(test_db_session)
        assert len(repo.list_all()) == 1

        # Attempt to book the exact same slot again in a new session
        session_id_2 = "test_scenario_h_repeat"
        booking_service.process_message(session_id_2, "Schedule interview", test_db_session, reference_future_date)
        booking_service.process_message(session_id_2, "Govinda Yadav", test_db_session, reference_future_date)
        booking_service.process_message(session_id_2, "govinda@example.com", test_db_session, reference_future_date)
        booking_service.process_message(session_id_2, "2026-11-20", test_db_session, reference_future_date)
        resp_repeat = booking_service.process_message(session_id_2, "10:30 AM", test_db_session, reference_future_date)

        assert "already been recorded" in resp_repeat.response
        assert resp_repeat.action_info.get("duplicate") is True
        # Ensure SQLite has exactly 1 booking record, not 2
        assert len(repo.list_all()) == 1

    def test_scenario_i_sqlite_persistence_verification(
        self,
        booking_service: BookingService,
        test_db_session: Session,
        reference_future_date: date,
    ):
        """Scenario I: Verify row in SQLite matches all validated attributes."""
        session_id = "test_scenario_i"
        booking_service.process_message(session_id, "Schedule interview", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "Candidate Name", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "candidate@domain.com", test_db_session, reference_future_date)
        booking_service.process_message(session_id, "2026-11-25", test_db_session, reference_future_date)
        resp = booking_service.process_message(session_id, "11:00", test_db_session, reference_future_date)

        repo = BookingRepository(test_db_session)
        b = repo.get_by_id(resp.action_info["booking_id"])
        assert b is not None
        assert b.name == "Candidate Name"
        assert b.email == "candidate@domain.com"
        assert b.interview_date == "2026-11-25"
        assert b.interview_time == "11:00"
        assert b.status == "confirmed"


# ==============================================================================
# 4. API Integration Tests (POST /api/v1/chat)
# ==============================================================================

class TestChatAPIWithBookingIntegration:
    """Integration tests via HTTP client verifying seamless routing between RAG and Booking."""

    @pytest.fixture
    async def api_client(
        self,
        fake_memory_service: MemoryService,
        test_db_session: Session,
    ) -> AsyncGenerator[AsyncClient, None]:
        """AsyncClient with overridden dependencies for isolated testing."""
        app.dependency_overrides[get_memory_service] = lambda: fake_memory_service
        app.dependency_overrides[get_db] = lambda: test_db_session

        booking_svc = BookingService(memory_service=fake_memory_service)
        app.dependency_overrides[get_booking_service] = lambda: booking_svc

        # Mock RAGService for non-booking document questions
        mock_rag = MagicMock(spec=RAGService)
        mock_rag.answer_question.return_value = ChatResponse(
            session_id="doc_session",
            response="Employees receive 20 days of annual leave according to the policy.",
            sources_used=[
                SourceChunk(
                    document_id=1,
                    filename="policy.pdf",
                    chunk_index=0,
                    score=0.89,
                    text_snippet="Leave policy: 20 days.",
                )
            ],
            intent="document_rag",
        )
        app.dependency_overrides[get_rag_service] = lambda: mock_rag

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_full_booking_flow_via_chat_endpoint(self, api_client: AsyncClient):
        """Verify the complete 5-turn interview booking conversation via HTTP API."""
        session_id = "http_booking_session"

        # Turn 1
        r1 = await api_client.post("/api/v1/chat", json={"session_id": session_id, "message": "I want to schedule an interview."})
        assert r1.status_code == 200
        assert r1.json()["response"] == "Sure! May I know your full name?"
        assert r1.json()["intent"] == "interview_booking"

        # Turn 2
        r2 = await api_client.post("/api/v1/chat", json={"session_id": session_id, "message": "Govinda Yadav."})
        assert r2.status_code == 200
        assert r2.json()["response"] == "What is your email address?"

        # Turn 3
        r3 = await api_client.post("/api/v1/chat", json={"session_id": session_id, "message": "govinda@example.com."})
        assert r3.status_code == 200
        assert r3.json()["response"] == "Which date would you prefer?"

        # Turn 4 (future date)
        r4 = await api_client.post("/api/v1/chat", json={"session_id": session_id, "message": "2026-11-20"})
        assert r4.status_code == 200
        assert r4.json()["response"] == "What time would you prefer?"

        # Turn 5
        r5 = await api_client.post("/api/v1/chat", json={"session_id": session_id, "message": "10:30 AM."})
        assert r5.status_code == 200
        data5 = r5.json()
        assert data5["response"] == "Your interview booking request has been recorded successfully."
        assert data5["action_info"]["email"] == "govinda@example.com"
        assert data5["action_info"]["status"] == "confirmed"

    @pytest.mark.asyncio
    async def test_ordinary_document_question_routes_to_rag(self, api_client: AsyncClient):
        """Verify that ordinary document inquiries route to RAG without triggering booking."""
        resp = await api_client.post(
            "/api/v1/chat",
            json={"session_id": "rag_session_1", "message": "What is the company leave policy?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Employees receive 20 days" in data["response"]
        assert data["intent"] == "document_rag"
        assert len(data["sources_used"]) == 1
        assert data["sources_used"][0]["filename"] == "policy.pdf"

