"""Document service orchestrating extraction, chunking, embedding, and storage."""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.api.schemas.document import ChunkingStrategyEnum, DocumentUploadResponse
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.models.document import Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.chunking_service import (
    ChunkingService,
    TextChunk,
    chunking_service as default_chunking_service,
)
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.extraction_service import (
    ExtractionService,
    extraction_service as default_extraction_service,
)
from app.services.vector_service import VectorService, get_vector_service
from app.utils.exceptions import (
    AppException,
    DocumentProcessingException,
    EmptyFileException,
)
from app.utils.validators import validate_file_upload

logger = setup_logging()


class DocumentService:
    """Coordinates the document ingestion pipeline across SQL and Vector databases."""

    def __init__(
        self,
        db: Session,
        embedding_service: Optional[EmbeddingService] = None,
        vector_service: Optional[VectorService] = None,
        chunking_service: Optional[ChunkingService] = None,
        extraction_service: Optional[ExtractionService] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.db = db
        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_service = vector_service or get_vector_service()
        self.chunking_service = chunking_service or default_chunking_service
        self.extraction_service = extraction_service or default_extraction_service
        self.settings = settings or get_settings()

    def ingest_document(
        self,
        filename: str,
        file_bytes: bytes,
        chunking_strategy: ChunkingStrategyEnum = ChunkingStrategyEnum.RECURSIVE,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> DocumentUploadResponse:
        """Execute the end-to-end document ingestion pipeline.

        Pipeline steps:
        1. Validate file extension, non-emptiness, and size limits.
        2. Extract raw text with encoding and format safety.
        3. Chunk text using the selected strategy (fixed or recursive).
        4. Generate 384-d dense vector embeddings for all chunks.
        5. Persist Document and Chunk metadata in SQLite.
        6. Upsert vectors and payloads into Qdrant.
        7. Maintain ACID atomicity: rollback DB and delete vectors on failure.
        """
        # Step 1: Validation
        clean_ext = validate_file_upload(
            filename=filename,
            file_bytes=file_bytes,
            allowed_extensions=self.settings.ALLOWED_FILE_TYPES,
            max_size_mb=self.settings.MAX_UPLOAD_SIZE_MB,
        )

        # Step 2: Extraction
        extracted_text = self.extraction_service.extract_text(file_bytes=file_bytes, filename=filename)
        if not extracted_text or not extracted_text.strip():
            raise EmptyFileException(filename=filename, details={"reason": "No readable text extracted"})

        char_count = len(extracted_text)

        # Step 3: Chunking
        chunks: List[TextChunk] = self.chunking_service.chunk_text(
            text=extracted_text,
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            raise DocumentProcessingException(
                message=f"Chunking yielded 0 chunks for document '{filename}'.",
                status_code=400,
            )

        # Step 4: Embedding Generation
        chunk_contents = [c.content for c in chunks]
        embeddings = self.embedding_service.generate_embeddings(chunk_contents)

        if len(embeddings) != len(chunks):
            raise DocumentProcessingException(
                message="Embedding count does not match chunk count.",
                status_code=500,
            )

        # Step 5 & 6: Relational + Vector Storage with rollback compensation
        created_doc_id: Optional[int] = None
        vectors_inserted = False

        try:
            # 5a. Persist Document record
            doc = self.doc_repo.create(
                filename=filename,
                file_type=clean_ext,
                file_size=len(file_bytes),
                chunking_strategy=chunking_strategy.value,
                total_chunks=len(chunks),
                character_count=char_count,
            )
            created_doc_id = doc.id

            # 5b. Persist Chunk records
            self.chunk_repo.bulk_create(document_id=doc.id, chunks=chunks)

            # 6. Upsert into Qdrant
            self.vector_service.upsert_chunks(
                document_id=doc.id,
                filename=filename,
                chunks=chunks,
                embeddings=embeddings,
            )
            vectors_inserted = True

            # Commit DB transaction
            self.db.commit()
            self.db.refresh(doc)
            logger.info(
                "Ingestion successful: doc_id=%d filename='%s' chunks=%d vectors=%d strategy=%s",
                doc.id,
                filename,
                len(chunks),
                len(embeddings),
                chunking_strategy.value,
            )

            return DocumentUploadResponse(
                success=True,
                document_id=doc.id,
                filename=doc.filename,
                extracted_character_count=char_count,
                number_of_chunks=len(chunks),
                selected_chunking_strategy=chunking_strategy,
                number_of_embeddings_stored=len(embeddings),
                message="Document uploaded, chunked, embedded, and indexed successfully.",
            )

        except Exception as exc:
            # Rollback DB transaction
            self.db.rollback()
            logger.error("Failed to ingest document '%s': %s. Rolling back DB.", filename, exc)

            # Clean up Qdrant vectors if inserted
            if vectors_inserted and created_doc_id is not None:
                try:
                    self.vector_service.delete_document_vectors(document_id=created_doc_id)
                    logger.info("Cleaned up orphaned Qdrant vectors for doc_id=%s", created_doc_id)
                except Exception as cleanup_err:
                    logger.warning(
                        "Failed to clean up Qdrant vectors for doc_id=%s: %s",
                        created_doc_id,
                        cleanup_err,
                    )

            if isinstance(exc, AppException):
                raise exc

            raise DocumentProcessingException(
                message=f"Failed to ingest document '{filename}'. Processing error encountered.",
                details={"error_type": exc.__class__.__name__},
            ) from exc

    def get_document(self, document_id: int) -> Optional[Document]:
        """Retrieve a document by its ID."""
        return self.doc_repo.get_by_id(document_id)

    def list_documents(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """List all indexed documents."""
        return self.doc_repo.list_all(skip=skip, limit=limit)

    def delete_document(self, document_id: int) -> bool:
        """Delete a document, its SQL chunks, and its vector embeddings."""
        doc = self.doc_repo.get_by_id(document_id)
        if not doc:
            return False

        # Delete from Qdrant first
        self.vector_service.delete_document_vectors(document_id=document_id)

        # Delete from SQLite
        self.doc_repo.delete(document_id)
        self.db.commit()
        logger.info("Completely purged document id=%s from SQLite and Qdrant", document_id)
        return True
