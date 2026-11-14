"""API route definitions for conversational RAG."""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, status

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.services.memory_service import MemoryService, get_memory_service
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/chat", tags=["Conversational RAG"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with documents using custom RAG",
    description=(
        "Ask a question grounded in the indexed documents. The system computes query embeddings, "
        "retrieves top relevant chunks from Qdrant, builds a grounded prompt with Redis multi-turn "
        "memory, and calls Google Gemini to generate an accurate answer with source metadata citations."
    ),
)
def chat_endpoint(
    request: ChatRequest,
    service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """Conversational RAG endpoint returning grounded answers and source citations."""
    return service.answer_question(
        query=request.message,
        session_id=request.session_id,
    )


@router.get(
    "/history/{session_id}",
    summary="Get session chat history",
    description="Retrieve chronologically ordered message history stored in Redis for a specific session.",
)
def get_session_history(
    session_id: str,
    memory: MemoryService = Depends(get_memory_service),
) -> List[Dict[str, Any]]:
    """Retrieve full chronological conversation history for a session."""
    return memory.get_history(session_id=session_id)


@router.delete(
    "/history/{session_id}",
    summary="Clear session chat history",
    description="Purge all conversation history for a specific session from Redis.",
)
def clear_session_history(
    session_id: str,
    memory: MemoryService = Depends(get_memory_service),
) -> Dict[str, Any]:
    """Delete conversation history for a session."""
    deleted = memory.clear_session(session_id=session_id)
    return {"session_id": session_id, "cleared": deleted}

