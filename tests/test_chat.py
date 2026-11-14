"""Unit and integration tests for custom Conversational RAG service and chat endpoint."""

from unittest.mock import MagicMock
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes.chat import get_rag_service
from app.api.schemas.chat import ChatResponse
from app.main import app
from app.services.rag_service import RAGService, SYSTEM_INSTRUCTION
from app.services.vector_service import ScoredChunk
from app.utils.exceptions import RAGException


# ==============================================================================
# Unit Tests for RAGService
# ==============================================================================

class TestRAGServiceUnit:
    """Unit tests for individual RAG pipeline components."""

    def test_build_prompt_structure_and_citations(self):
        """Verify build_prompt constructs formatted context with citations and grounding rules."""
        service = RAGService()
        chunks = [
            ScoredChunk(
                document_id=1,
                filename="company_policy.pdf",
                chunk_index=0,
                chunk_text="PalmMind AI offers flexible remote work for engineering interns.",
                score=0.885,
            ),
            ScoredChunk(
                document_id=1,
                filename="company_policy.pdf",
                chunk_index=1,
                chunk_text="Core working hours are 10:00 AM to 4:00 PM NPT.",
                score=0.742,
            ),
        ]
        query = "What are the working hours at PalmMind AI?"
        prompt = service.build_prompt(query=query, chunks=chunks)

        # Assert grounding instructions
        assert "Retrieved Document Context:" in prompt
        assert "company_policy.pdf" in prompt
        assert "Chunk: 0" in prompt
        assert "Chunk: 1" in prompt
        assert "0.885" in prompt
        assert "flexible remote work" in prompt
        assert "10:00 AM to 4:00 PM" in prompt
        assert "User Question: What are the working hours at PalmMind AI?" in prompt
        assert "Answer the question using ONLY the facts" in prompt
        assert "Answer:" in prompt

    def test_build_prompt_empty_chunks(self):
        """Verify build_prompt handles empty chunks gracefully."""
        service = RAGService()
        prompt = service.build_prompt(query="Any questions?", chunks=[])
        assert "No relevant context found in indexed documents." in prompt
        assert "User Question: Any questions?" in prompt

    def test_retrieve_context_empty_query(self):
        """Verify retrieve_context returns empty list for empty/whitespace query."""
        service = RAGService()
        assert service.retrieve_context("") == []
        assert service.retrieve_context("   \n\t  ") == []

    def test_retrieve_context_delegates_to_embedding_and_vector(self):
        """Verify retrieve_context computes embedding and calls vector service."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embedding.return_value = [0.1] * 384

        mock_vector_service = MagicMock()
        mock_chunk = ScoredChunk(
            document_id=10,
            filename="tech_stack.txt",
            chunk_index=0,
            chunk_text="We use FastAPI and Qdrant.",
            score=0.91,
        )
        mock_vector_service.search_similar.return_value = [mock_chunk]

        service = RAGService(
            embedding_service=mock_embedding_service,
            vector_service=mock_vector_service,
        )

        results = service.retrieve_context(query="What is the tech stack?", top_k=3, score_threshold=0.3)
        assert len(results) == 1
        assert results[0].filename == "tech_stack.txt"
        mock_embedding_service.generate_embedding.assert_called_once_with("What is the tech stack?")
        mock_vector_service.search_similar.assert_called_once_with(
            query_vector=[0.1] * 384,
            top_k=3,
            score_threshold=0.3,
            document_id=None,
        )

    def test_generate_response_mocked_gemini(self):
        """Verify generate_response invokes Gemini Client and returns stripped text."""
        mock_gemini_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "  PalmMind AI uses FastAPI and Qdrant.  "
        mock_gemini_client.models.generate_content.return_value = mock_response

        service = RAGService(gemini_client=mock_gemini_client)
        answer = service.generate_response(prompt="Test prompt")

        assert answer == "PalmMind AI uses FastAPI and Qdrant."
        mock_gemini_client.models.generate_content.assert_called_once()

    def test_generate_response_empty_gemini_text_raises_exception(self):
        """Verify generate_response raises RAGException if Gemini returns empty text."""
        mock_gemini_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_gemini_client.models.generate_content.return_value = mock_response

        service = RAGService(gemini_client=mock_gemini_client)
        with pytest.raises(RAGException) as exc_info:
            service.generate_response(prompt="Test prompt")
        assert exc_info.value.status_code == 502
        assert "empty response" in exc_info.value.message

    def test_generate_response_api_failure_raises_502(self):
        """Verify Gemini connection or quota errors are caught and re-raised as 502."""
        mock_gemini_client = MagicMock()
        mock_gemini_client.models.generate_content.side_effect = RuntimeError("Quota exceeded 429")

        service = RAGService(gemini_client=mock_gemini_client)
        with pytest.raises(RAGException) as exc_info:
            service.generate_response(prompt="Test prompt")
        assert exc_info.value.status_code == 502
        assert "LLM generation failed" in exc_info.value.message

    def test_missing_gemini_api_key_raises_500(self):
        """Verify attempting to generate without API key raises 500 RAGException."""
        mock_settings = MagicMock()
        mock_settings.GEMINI_API_KEY = ""
        service = RAGService(settings=mock_settings)

        with pytest.raises(RAGException) as exc_info:
            service.generate_response(prompt="Hello")
        assert exc_info.value.status_code == 500
        assert "Google Gemini API key is not configured" in exc_info.value.message


# ==============================================================================
# Integration Tests for POST /api/v1/chat Endpoint
# ==============================================================================

class TestChatEndpointIntegration:
    """Integration test suite for the /api/v1/chat endpoint."""

    @pytest.fixture
    def mock_rag_service(self):
        """Create a mock RAGService for API endpoint tests."""
        service = MagicMock(spec=RAGService)
        return service

    @pytest.mark.asyncio
    async def test_chat_success_with_grounded_answer(self, mock_rag_service):
        """Verify successful chat turn returns answer and source chunk citations."""
        mock_rag_service.answer_question.return_value = ChatResponse(
            session_id="session_abc_123",
            response="PalmMind AI implements custom RAG using Qdrant and Gemini.",
            sources_used=[
                {
                    "document_id": 1,
                    "filename": "rag_architecture.pdf",
                    "chunk_index": 0,
                    "score": 0.8921,
                    "text_snippet": "The architecture avoids RetrievalQAChain in favor of custom orchestration...",
                }
            ],
            intent="document_rag",
        )

        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "session_id": "session_abc_123",
                "message": "How does PalmMind AI implement RAG?",
            }
            response = await client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert data["session_id"] == "session_abc_123"
            assert "custom RAG using Qdrant" in data["response"]
            assert len(data["sources_used"]) == 1
            assert data["sources_used"][0]["filename"] == "rag_architecture.pdf"
            assert data["sources_used"][0]["chunk_index"] == 0
            assert data["sources_used"][0]["score"] == 0.8921

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_chat_insufficient_context_handling(self, mock_rag_service):
        """Verify chat endpoint returns clear message when no context matches the inquiry."""
        mock_rag_service.answer_question.return_value = ChatResponse(
            session_id="session_xyz_789",
            response=(
                "Based on the indexed documents, I could not find information regarding your inquiry. "
                "Please verify that the relevant document has been uploaded or rephrase your question."
            ),
            sources_used=[],
            intent="document_rag",
        )

        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "session_id": "session_xyz_789",
                "message": "What is the capital of Mars?",
            }
            response = await client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert data["session_id"] == "session_xyz_789"
            assert "could not find information" in data["response"]
            assert data["sources_used"] == []

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_chat_gemini_failure_returns_502_error_json(self, mock_rag_service):
        """Verify that Gemini upstream failure returns HTTP 502 with structured error JSON."""
        mock_rag_service.answer_question.side_effect = RAGException(
            message="LLM generation failed: Gemini connection timeout.",
            status_code=502,
            details={"model": "gemini-2.5-flash"},
        )

        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "session_id": "session_fail_001",
                "message": "Explain the hiring process.",
            }
            response = await client.post("/api/v1/chat", json=payload)
            assert response.status_code == 502

            data = response.json()
            assert data["error"] == "RAGException"
            assert "Gemini connection timeout" in data["message"]

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_chat_empty_fields_validation_error(self):
        """Verify that missing or empty message/session_id returns HTTP 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Empty message
            res1 = await client.post("/api/v1/chat", json={"session_id": "s1", "message": ""})
            assert res1.status_code == 422

            # Missing session_id
            res2 = await client.post("/api/v1/chat", json={"message": "hello"})
            assert res2.status_code == 422

