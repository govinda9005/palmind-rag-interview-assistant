"""Repository layer for Chunk database operations."""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.chunk import Chunk
from app.services.chunking_service import TextChunk
from app.core.logging import setup_logging

logger = setup_logging()


class ChunkRepository:
    """Encapsulates database access logic for Chunk records."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def bulk_create(self, document_id: int, chunks: List[TextChunk]) -> List[Chunk]:
        """Bulk insert chunks for an ingested document."""
        chunk_entities: List[Chunk] = []
        for c in chunks:
            entity = Chunk(
                document_id=document_id,
                chunk_index=c.chunk_index,
                content=c.content,
                character_count=c.character_count,
                start_char=c.start_char,
                end_char=c.end_char,
            )
            chunk_entities.append(entity)

        self.db.add_all(chunk_entities)
        self.db.flush()
        logger.info("Persisted %d Chunk records for document_id=%s", len(chunk_entities), document_id)
        return chunk_entities

    def get_by_document_id(self, document_id: int) -> List[Chunk]:
        """Fetch all chunks for a document ordered by chunk_index."""
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(self.db.scalars(stmt).all())

    def get_by_id(self, chunk_id: int) -> Optional[Chunk]:
        """Fetch a single chunk by ID."""
        stmt = select(Chunk).where(Chunk.id == chunk_id)
        return self.db.scalars(stmt).first()

