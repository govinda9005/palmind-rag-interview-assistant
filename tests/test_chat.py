"""Unit and integration tests for custom Conversational RAG service, Redis memory, and chat endpoint."""

import time
from unittest.mock import MagicMock
import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes.chat import get_memory_service, get_rag_service
from app.api.schemas.chat import ChatResponse
from app.core.config import Settings
from app.main import app
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService, SYSTEM_INSTRUCTION
from app.services.vector_service import ScoredChunk
from app.utils.exceptions import RAGException


# ==============================================================================
# Unit Tests for MemoryService (Redis Session-Isolated Memory)
# ==============================================================================

class TestMemoryServiceUnit:
    """Unit tests for Redis-based multi-turn conversation memory."""

    @pytest.fixture
    def fake_memory_service(self) -> MemoryService:
        """Provide an isolated MemoryService backed by FakeRedis."""
        client = fakeredis.FakeRedis(decode_responses=True)
        settings = Settings(
            REDIS_HISTORY_LIMIT=3,
            REDIS_SESSION_TTL_SECONDS=3600,
        )
        return MemoryService(client=client, settings=settings)

    def test_new_session_empty_history(self, fake_memory_service: MemoryService):
        """Verify that a new session starts with no history."""
        history = fake_memory_service.get_history("brand_new_session")
        assert history == []
        assert isinstance(history, list)

    def test_chronological_ordering_and_add_turn(self, fake_memory_service: MemoryService):
        """Verify messages are stored and retrieved in strict chronological order."""
        session_id = "test_chrono_sess"
        fake_memory_service.add_turn(
            session_id=session_id,
            user_message="Hello",
            assistant_message="Hi there! How can I help you?",
        )
        fake_memory_service.add_turn(
            session_id=session_id,
            user_message="Tell me about PalmMind AI.",
            assistant_message="PalmMind AI is an AI engineering enterprise.",
        )

        history = fake_memory_service.get_history(session_id)
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there! How can I help you?"
        assert history[2]["role"] == "user"
        assert history[2]["content"] == "Tell me about PalmMind AI."
        assert history[3]["role"] == "assistant"
        assert history[3]["content"] == "PalmMind AI is an AI engineering enterprise."

    def test_session_isolation_different_users(self, fake_memory_service: MemoryService):
        """Ensure different session IDs never share or leak conversation history."""
        session_alice = "session_user_alice"
        session_bob = "session_user_bob"

        fake_memory_service.add_turn(
            session_id=session_alice,
            user_message="My name is Alice.",
            assistant_message="Nice to meet you, Alice.",
        )
        fake_memory_service.add_turn(
            session_id=session_bob,
            user_message="My name is Bob.",
            assistant_message="Hello Bob.",
        )

        alice_history = fake_memory_service.get_history(session_alice)
        bob_history = fake_memory_service.get_history(session_bob)

        assert len(alice_history) == 2
        assert len(bob_history) == 2
        assert "Alice" in alice_history[0]["content"]
        assert "Bob" not in str(alice_history)
        assert "Bob" in bob_history[0]["content"]
        assert "Alice" not in str(bob_history)

    def test_duplicate_message_prevention(self, fake_memory_service: MemoryService):
        """Verify duplicate successive submissions are not stored redundantly."""
        session_id = "test_dup_sess"
        fake_memory_service.add_message(session_id, role="user", content="Same question")
        fake_memory_service.add_message(session_id, role="user", content="Same question")

        history = fake_memory_service.get_history(session_id)
        assert len(history) == 1
        assert history[0]["content"] == "Same question"

    def test_history_limit_truncation(self, fake_memory_service: MemoryService):
        """Verify history prunes older turns when exceeding configured REDIS_HISTORY_LIMIT."""
        session_id = "test_limit_sess"
        # Configured REDIS_HISTORY_LIMIT is 3 turns = 6 messages max
        for i in range(1, 6):  # 5 turns = 10 messages
            fake_memory_service.add_turn(
                session_id=session_id,
                user_message=f"Question {i}",
                assistant_message=f"Answer {i}",
            )

        history = fake_memory_service.get_history(session_id)
        assert len(history) == 6  # 3 most recent turns = 6 messages
        assert history[0]["content"] == "Question 3"
        assert history[-1]["content"] == "Answer 5"

    def test_session_expiration_ttl(self, fake_memory_service: MemoryService):
        """Verify TTL expiration is set on the Redis key."""
        session_id = "test_ttl_sess"
        fake_memory_service.add_message(session_id, role="user", content="Test TTL")

        ttl = fake_memory_service.get_session_ttl(session_id)
        assert ttl > 0
        assert ttl <= 3600

    def test_clear_session(self, fake_memory_service: MemoryService):
        """Verify clear_session purges all session keys."""
        session_id = "to_clear"
        fake_memory_service.add_turn(session_id, "hello", "hi")
        assert len(fake_memory_service.get_history(session_id)) == 2

        deleted = fake_memory_service.clear_session(session_id)
        assert deleted is True
        assert fake_memory_service.get_history(session_id) == []


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

    def test_build_prompt_with_conversation_history(self):
        """Verify build_prompt integrates past conversation history."""
        service = RAGService()
        chunks = [
            ScoredChunk(
                document_id=1,
                filename="leave_policy.txt",
                chunk_index=0,
                chunk_text="Employees receive 20 days of annual leave.",
                score=0.92,
            )
        ]
        history = [
            {"role": "user", "content": "What is the company leave policy?"},
            {"role": "assistant", "content": "According to the uploaded document, employees receive 20 days of annual leave."},
        ]
        prompt = service.build_prompt(
            query="How many days is that per year?",
            chunks=chunks,
            conversation_history=history,
        )

        assert "Previous Conversation History:" in prompt
        assert "User: What is the company leave policy?" in prompt
        assert "Assistant: According to the uploaded document" in prompt
        assert "User Question: How many days is that per year?" in prompt

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
# Multi-Turn Conversational RAG Integration Tests
# ==============================================================================

class TestMultiTurnConversationalRAG:
    """Integration test suite verifying multi-turn conversation memory with RAG."""

    @pytest.fixture
    def test_rag_service(self) -> RAGService:
        """Create RAGService equipped with FakeRedis and mocked Gemini."""
        client = fakeredis.FakeRedis(decode_responses=True)
        memory = MemoryService(client=client)

        mock_embedding = MagicMock()
        mock_embedding.generate_embedding.return_value = [0.05] * 384

        mock_vector = MagicMock()
        chunk = ScoredChunk(
            document_id=1,
            filename="leave_policy.txt",
            chunk_index=0,
            chunk_text="According to the company leave policy, employees receive 20 days of annual leave.",
            score=0.91,
        )
        mock_vector.search_similar.return_value = [chunk]

        mock_gemini = MagicMock()

        # Dynamic mock answering based on prompt content
        def mock_generate_content(model, contents, config):
            resp = MagicMock()
            if "How many days is that per year?" in contents:
                resp.text = "It is 20 days per year."
            else:
                resp.text = "According to the uploaded document, employees receive 20 days of annual leave."
            return resp

        mock_gemini.models.generate_content.side_effect = mock_generate_content

        return RAGService(
            embedding_service=mock_embedding,
            vector_service=mock_vector,
            memory_service=memory,
            gemini_client=mock_gemini,
        )

    @pytest.mark.asyncio
    async def test_multi_turn_leave_policy_conversation_scenario(self, test_rag_service: RAGService):
        """Test the required assignment scenario:
        User: What is the company leave policy?
        Assistant: According to the uploaded document, employees receive 20 days of annual leave.
        User: How many days is that per year?
        Assistant: It is 20 days per year.
        The second question must be interpreted using the previous conversation.
        """
        app.dependency_overrides[get_rag_service] = lambda: test_rag_service
        app.dependency_overrides[get_memory_service] = lambda: test_rag_service.memory_service

        session_id = "sess_leave_test_001"
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Turn 1
            turn_1_payload = {
                "session_id": session_id,
                "message": "What is the company leave policy?",
            }
            resp_1 = await client.post("/api/v1/chat", json=turn_1_payload)
            assert resp_1.status_code == 200
            data_1 = resp_1.json()
            assert "20 days of annual leave" in data_1["response"]
            assert len(data_1["sources_used"]) == 1
            assert data_1["sources_used"][0]["filename"] == "leave_policy.txt"

            # Verify Turn 1 is in Redis
            history_after_turn_1 = test_rag_service.memory_service.get_history(session_id)
            assert len(history_after_turn_1) == 2
            assert history_after_turn_1[0]["role"] == "user"
            assert history_after_turn_1[0]["content"] == "What is the company leave policy?"
            assert history_after_turn_1[1]["role"] == "assistant"
            assert "20 days of annual leave" in history_after_turn_1[1]["content"]

            # Turn 2: Follow-up relying on Turn 1 context
            turn_2_payload = {
                "session_id": session_id,
                "message": "How many days is that per year?",
            }
            resp_2 = await client.post("/api/v1/chat", json=turn_2_payload)
            assert resp_2.status_code == 200
            data_2 = resp_2.json()
            assert data_2["response"] == "It is 20 days per year."

            # Verify Redis contains both turns chronologically (4 messages total)
            history_after_turn_2 = test_rag_service.memory_service.get_history(session_id)
            assert len(history_after_turn_2) == 4
            assert history_after_turn_2[2]["role"] == "user"
            assert history_after_turn_2[2]["content"] == "How many days is that per year?"
            assert history_after_turn_2[3]["role"] == "assistant"
            assert history_after_turn_2[3]["content"] == "It is 20 days per year."

            # Test GET /api/v1/chat/history/{session_id}
            get_hist = await client.get(f"/api/v1/chat/history/{session_id}")
            assert get_hist.status_code == 200
            assert len(get_hist.json()) == 4

            # Test DELETE /api/v1/chat/history/{session_id}
            del_hist = await client.delete(f"/api/v1/chat/history/{session_id}")
            assert del_hist.status_code == 200
            assert del_hist.json()["cleared"] is True

            # Verify history is now empty
            get_hist_after_del = await client.get(f"/api/v1/chat/history/{session_id}")
            assert get_hist_after_del.json() == []

        app.dependency_overrides.clear()


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

