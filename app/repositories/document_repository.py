"""Repository layer for Document database operations."""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from app.models.document import Document
from app.core.logging import setup_logging

logger = setup_logging()


class DocumentRepository:
    """Encapsulates database access logic for Document records."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        chunking_strategy: str,
        total_chunks: int,
        character_count: int,
    ) -> Document:
        """Create and persist a new Document record."""
        doc = Document(
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            chunking_strategy=chunking_strategy,
            total_chunks=total_chunks,
            character_count=character_count,
        )
        self.db.add(doc)
        self.db.flush()  # Flush to populate doc.id without ending transaction
        logger.info("Created Document record in DB: id=%s filename=%s", doc.id, filename)
        return doc

    def get_by_id(self, document_id: int) -> Optional[Document]:
        """Fetch a document by its primary key ID."""
        stmt = select(Document).where(Document.id == document_id)
        return self.db.scalars(stmt).first()

    def get_by_filename(self, filename: str) -> Optional[Document]:
        """Fetch the most recent document matching a filename."""
        stmt = (
            select(Document)
            .where(Document.filename == filename)
            .order_by(desc(Document.created_at))
        )
        return self.db.scalars(stmt).first()

    def list_all(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """List documents ordered by creation date descending."""
        stmt = select(Document).order_by(desc(Document.created_at)).offset(skip).limit(limit)
        return list(self.db.scalars(stmt).all())

    def delete(self, document_id: int) -> bool:
        """Delete a document by ID. Cascades deletion to associated chunks."""
        doc = self.get_by_id(document_id)
        if not doc:
            return False
        self.db.delete(doc)
        self.db.flush()
        logger.info("Deleted Document record from DB: id=%s", document_id)
        return True

