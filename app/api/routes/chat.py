"""API route definitions for conversational RAG and interview booking."""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.database.connection import get_db
from app.services.booking_service import BookingService, get_booking_service
from app.services.memory_service import MemoryService, get_memory_service
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/chat", tags=["Conversational RAG & Booking"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with documents or book an interview",
    description=(
        "Ask a question grounded in indexed documents using custom RAG, or "
        "schedule an interview through multi-turn conversational booking."
    ),
)
def chat_endpoint(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    booking_service: BookingService = Depends(get_booking_service),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """Conversational endpoint routing between interview booking and custom document RAG."""
    # 1. If an active booking flow is already in progress for this session, continue booking
    if booking_service.is_booking_active(session_id=request.session_id):
        return booking_service.process_message(
            session_id=request.session_id,
            message=request.message,
            db=db,
        )

    # 2. Check if the incoming utterance initiates an interview booking
    extraction = booking_service.detect_intent_and_extract(
        message=request.message,
        session_id=request.session_id,
    )
    if extraction.intent == "interview_booking":
        return booking_service.process_message(
            session_id=request.session_id,
            message=request.message,
            db=db,
        )

    # 3. Otherwise, execute custom document RAG with Redis multi-turn memory
    return rag_service.answer_question(
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
    memory.clear_booking_draft(session_id=session_id)
    return {"session_id": session_id, "cleared": deleted}
