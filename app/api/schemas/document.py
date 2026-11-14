"""Pydantic models for document ingestion."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategyEnum(str, Enum):
    """Supported chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    model_config = ConfigDict(from_attributes=True)

    success: bool = True
    document_id: int
    filename: str
    extracted_character_count: int
    number_of_chunks: int
    selected_chunking_strategy: ChunkingStrategyEnum
    number_of_embeddings_stored: int
    message: Optional[str] = "Document uploaded, chunked, embedded, and indexed successfully."

