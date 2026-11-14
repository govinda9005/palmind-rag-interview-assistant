"""API route definitions for conversational RAG."""

from fastapi import APIRouter, Depends, status

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/chat", tags=["Conversational RAG"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with documents using custom RAG",
    description=(
        "Ask a question grounded in the indexed documents. The system computes query embeddings, "
        "retrieves top relevant chunks from Qdrant, builds a grounded prompt, and calls Google Gemini "
        "to generate an accurate answer with source metadata citations."
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

