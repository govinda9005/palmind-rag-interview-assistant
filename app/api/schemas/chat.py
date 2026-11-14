"""Pydantic models for conversational RAG and interview booking."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming user chat request."""

    session_id: str = Field(..., min_length=1, description="Unique conversation session identifier")
    message: str = Field(..., min_length=1, description="User message or question")


class SourceChunk(BaseModel):
    """Retrieved document chunk context used for RAG grounding."""

    document_id: int
    filename: str
    chunk_index: int
    score: float
    text_snippet: str


class ChatResponse(BaseModel):
    """Assistant chat response."""

    session_id: str
    response: str
    sources_used: List[SourceChunk] = Field(default_factory=list)
    intent: Optional[str] = None
    action_info: Optional[Dict[str, Any]] = None

