"""Database initialization and schema migration helpers."""

from app.core.logging import setup_logging
from app.database.connection import Base, engine
# Import all models to ensure they register with Base.metadata
from app.models import Document, Chunk, Booking  # noqa: F401

logger = setup_logging()


def init_db() -> None:
    """Create all database tables if they do not exist."""
    logger.info("Initializing SQLite database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")

