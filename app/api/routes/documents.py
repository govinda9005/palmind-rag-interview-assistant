"""API route definitions for document ingestion and management."""

from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.schemas.document import ChunkingStrategyEnum, DocumentUploadResponse
from app.database.connection import get_db
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    """Dependency provider for DocumentService."""
    return DocumentService(db=db)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description=(
        "Upload a PDF or TXT file. The system extracts its text, applies the selected "
        "chunking strategy (fixed or recursive), generates 384-d embeddings using "
        "all-MiniLM-L6-v2, and indexes the content in both SQLite and Qdrant."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="The document file (PDF or TXT) to upload"),
    chunking_strategy: ChunkingStrategyEnum = Form(
        default=ChunkingStrategyEnum.RECURSIVE,
        description="Selectable chunking strategy: 'fixed' or 'recursive'",
    ),
    chunk_size: Optional[int] = Form(
        default=None,
        description="Optional chunk size in characters (uses system defaults if omitted)",
    ),
    chunk_overlap: Optional[int] = Form(
        default=None,
        description="Optional chunk overlap in characters (uses system defaults if omitted)",
    ),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """Thin route handler delegating the ingestion pipeline to DocumentService."""
    file_bytes = await file.read()
    return service.ingest_document(
        filename=file.filename or "uploaded_document",
        file_bytes=file_bytes,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@router.get(
    "/",
    summary="List ingested documents",
    description="Retrieve a list of all ingested documents stored in SQLite.",
)
def list_documents(
    skip: int = 0,
    limit: int = 100,
    service: DocumentService = Depends(get_document_service),
) -> List[dict]:
    """Retrieve indexed documents metadata."""
    docs = service.list_documents(skip=skip, limit=limit)
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "chunking_strategy": doc.chunking_strategy,
            "total_chunks": doc.total_chunks,
            "character_count": doc.character_count,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        }
        for doc in docs
    ]


@router.get(
    "/{document_id}",
    summary="Get document details and chunks",
    description="Retrieve document metadata and all associated chunks by document ID.",
)
def get_document(
    document_id: int,
    service: DocumentService = Depends(get_document_service),
) -> dict:
    """Retrieve a single document and its ordered chunks."""
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found.",
        )
    return {
        "id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "file_size": doc.file_size,
        "chunking_strategy": doc.chunking_strategy,
        "total_chunks": doc.total_chunks,
        "character_count": doc.character_count,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "chunks": [
            {
                "id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "character_count": chunk.character_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for chunk in doc.chunks
        ],
    }


@router.delete(
    "/{document_id}",
    summary="Delete document",
    description="Delete a document, all its chunks from SQLite, and its vectors from Qdrant.",
)
def delete_document(
    document_id: int,
    service: DocumentService = Depends(get_document_service),
) -> dict:
    """Purge a document from both SQLite and Qdrant."""
    deleted = service.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found.",
        )
    return {"success": True, "message": f"Document {document_id} deleted successfully."}
