"""Repository layer for Interview Booking database operations."""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from app.models.booking import Booking
from app.core.logging import setup_logging

logger = setup_logging()


class BookingRepository:
    """Encapsulates database access logic for Booking records."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        name: str,
        email: str,
        interview_date: str,
        interview_time: str,
        status: str = "confirmed",
    ) -> Booking:
        """Create and persist a new Booking record.

        Args:
            name: Candidate full name.
            email: Candidate email address.
            interview_date: Normalized date string (YYYY-MM-DD).
            interview_time: Normalized time string (HH:MM).
            status: Booking status (default: 'confirmed').

        Returns:
            The created Booking instance with populated ID.
        """
        booking = Booking(
            name=name.strip(),
            email=email.strip().lower(),
            interview_date=interview_date.strip(),
            interview_time=interview_time.strip(),
            status=status,
        )
        self.db.add(booking)
        self.db.commit()
        self.db.refresh(booking)
        logger.info(
            "Created Booking record in DB: id=%s email=%s date=%s time=%s",
            booking.id,
            booking.email,
            booking.interview_date,
            booking.interview_time,
        )
        return booking

    def get_by_id(self, booking_id: int) -> Optional[Booking]:
        """Fetch a booking by primary key ID."""
        stmt = select(Booking).where(Booking.id == booking_id)
        return self.db.scalars(stmt).first()

    def find_duplicate(
        self,
        email: str,
        interview_date: str,
        interview_time: str,
    ) -> Optional[Booking]:
        """Find an existing active booking with matching email, date, and time.

        Used to prevent accidental duplicate booking creation.
        """
        stmt = select(Booking).where(
            Booking.email == email.strip().lower(),
            Booking.interview_date == interview_date.strip(),
            Booking.interview_time == interview_time.strip(),
        )
        return self.db.scalars(stmt).first()

    def list_all(self, skip: int = 0, limit: int = 100) -> List[Booking]:
        """List bookings ordered by creation date descending."""
        stmt = select(Booking).order_by(desc(Booking.created_at)).offset(skip).limit(limit)
        return list(self.db.scalars(stmt).all())

    def list_by_email(self, email: str) -> List[Booking]:
        """List all bookings for a given candidate email."""
        stmt = (
            select(Booking)
            .where(Booking.email == email.strip().lower())
            .order_by(desc(Booking.created_at))
        )
        return list(self.db.scalars(stmt).all())

