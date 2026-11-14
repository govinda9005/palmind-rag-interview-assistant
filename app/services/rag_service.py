"""Custom Retrieval-Augmented Generation (RAG) Service powered by Google Gemini and Qdrant.

Note: Strictly custom implementation. Does NOT use RetrievalQAChain or FAISS/Chroma.
"""

from typing import Any, Dict, List, Optional
from app.api.schemas.chat import ChatResponse, SourceChunk
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.vector_service import ScoredChunk, VectorService, get_vector_service
from app.services.memory_service import MemoryService, get_memory_service
from app.utils.exceptions import RAGException

logger = setup_logging()

SYSTEM_INSTRUCTION = (
    "You are PalmMind AI's expert Document Assistant. Your primary responsibility is "
    "to provide accurate, factual, and helpful answers strictly based on the provided "
    "Context Information extracted from indexed documents.\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. You must NEVER hallucinate or extrapolate facts beyond what is explicitly stated in the context.\n"
    "2. If the context does not contain sufficient details to answer the user's question, you MUST clearly state: "
    "'Based on the indexed documents, I could not find information regarding your inquiry. "
    "Please verify that the relevant document has been uploaded or rephrase your question.'\n"
    "3. Maintain a professional, concise, and helpful tone.\n"
    "4. Cite the source document name and chunk index when providing specific facts from the context."
)


class RAGService:
    """Coordinates custom Retrieval-Augmented Generation without prebuilt chains."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_service: Optional[VectorService] = None,
        memory_service: Optional[MemoryService] = None,
        gemini_client: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_service = vector_service or get_vector_service()
        self.memory_service = memory_service or get_memory_service()
        self._gemini_client = gemini_client

    def _get_gemini_client(self) -> Any:
        """Lazy-load or return configured Google GenAI Client."""
        if self._gemini_client is not None:
            return self._gemini_client

        api_key = self.settings.GEMINI_API_KEY
        if not api_key:
            raise RAGException(
                message="Google Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file.",
                status_code=500,
                details={"reason": "MISSING_GEMINI_API_KEY"},
            )

        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=api_key)
            return self._gemini_client
        except Exception as exc:
            logger.error("Failed to initialize Google GenAI Client: %s", exc)
            raise RAGException(
                message=f"Failed to initialize Gemini client: {str(exc)}",
                status_code=500,
                details={"error": str(exc)},
            ) from exc

    def retrieve_context(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = 0.25,
        document_id: Optional[int] = None,
    ) -> List[ScoredChunk]:
        """Generate a query embedding and retrieve semantically relevant document chunks.

        Args:
            query: The user's question or search prompt.
            top_k: Maximum number of relevant chunks to retrieve.
            score_threshold: Minimum cosine similarity threshold (default: 0.25).
            document_id: Optional filter to restrict retrieval to a specific document.

        Returns:
            List of ScoredChunk instances sorted by descending relevance.
        """
        if not query or not query.strip():
            return []

        try:
            query_vector = self.embedding_service.generate_embedding(query.strip())
            chunks = self.vector_service.search_similar(
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=score_threshold,
                document_id=document_id,
            )
            logger.info(
                "Retrieved %d context chunks for query '%s' (top_k=%d, threshold=%s)",
                len(chunks),
                query[:40],
                top_k,
                score_threshold,
            )
            return chunks
        except Exception as exc:
            logger.error("Error during semantic context retrieval: %s", exc)
            raise RAGException(
                message=f"Context retrieval failed: {str(exc)}",
                status_code=500,
                details={"query": query, "error": str(exc)},
            ) from exc

    def build_prompt(
        self,
        query: str,
        chunks: List[ScoredChunk],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Construct the prompt manually combining context chunks, history, and the user's question.

        Args:
            query: The current user inquiry.
            chunks: Retrieved context chunks from Qdrant.
            conversation_history: Optional list of past conversation turns.

        Returns:
            The complete, formatted prompt string for Gemini.
        """
        if not chunks:
            context_str = "No relevant context found in indexed documents."
        else:
            blocks = []
            for i, chunk in enumerate(chunks, 1):
                block = (
                    f"--- Context Source {i} [File: {chunk.filename}, Chunk: {chunk.chunk_index}, "
                    f"DocID: {chunk.document_id}, Relevance: {chunk.score:.4f}] ---\n"
                    f"{chunk.chunk_text.strip()}\n"
                )
                blocks.append(block)
            context_str = "\n".join(blocks)

        history_str = ""
        if conversation_history:
            history_lines = []
            for turn in conversation_history:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "")
                history_lines.append(f"{role}: {content}")
            history_str = "Previous Conversation History:\n" + "\n".join(history_lines) + "\n\n"

        prompt = (
            f"Retrieved Document Context:\n"
            f"==================================================\n"
            f"{context_str}\n"
            f"==================================================\n\n"
            f"{history_str}"
            f"User Question: {query}\n\n"
            f"Instructions:\n"
            f"1. Answer the question using ONLY the facts present in the 'Retrieved Document Context' above.\n"
            f"2. If the context does not contain the answer or is insufficient, explicitly state: "
            f"'Based on the indexed documents, I could not find information regarding your inquiry. "
            f"Please verify that the relevant document has been uploaded or rephrase your question.'\n"
            f"3. Cite the relevant source file name when answering.\n\n"
            f"Answer:"
        )
        return prompt

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Execute text generation via the Gemini API using the constructed prompt.

        Args:
            prompt: Complete constructed prompt including context.
            system_instruction: Optional system persona prompt.

        Returns:
            The raw text answer generated by Gemini.
        """
        client = self._get_gemini_client()
        sys_inst = system_instruction or SYSTEM_INSTRUCTION

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=0.2,  # Low temperature for strict factual grounding
                system_instruction=sys_inst,
            )

            response = client.models.generate_content(
                model=self.settings.GEMINI_MODEL,
                contents=prompt,
                config=config,
            )

            if not response or not response.text:
                raise RAGException(
                    message="Gemini model returned an empty response.",
                    status_code=502,
                )

            answer = response.text.strip()
            logger.info("Gemini generated %d characters response", len(answer))
            return answer

        except RAGException:
            raise
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise RAGException(
                message=f"LLM generation failed: {str(exc)}",
                status_code=502,
                details={"model": self.settings.GEMINI_MODEL, "error": str(exc)},
            ) from exc

    def answer_question(
        self,
        query: str,
        session_id: str,
        top_k: int = 4,
        score_threshold: Optional[float] = 0.25,
        document_id: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ChatResponse:
        """Execute end-to-end custom RAG answering workflow.

        Steps:
        1. Retrieve top-k semantically relevant chunks from Qdrant.
        2. If no chunks pass the relevance threshold, return a direct insufficient context response.
        3. Construct grounded prompt with retrieved chunks and user question.
        4. Query Gemini with the custom prompt.
        5. Return structured ChatResponse containing the generated answer and source citations.
        """
        # Retrieve prior session conversation history from Redis
        history = (
            conversation_history
            if conversation_history is not None
            else self.memory_service.get_history(session_id=session_id)
        )

        # Step 1: Semantic retrieval
        chunks = self.retrieve_context(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            document_id=document_id,
        )

        # Contextual retrieval fallback for follow-up questions
        if (not chunks or (chunks and chunks[0].score < 0.40)) and history:
            past_user_queries = [turn["content"] for turn in history if turn.get("role") == "user"]
            if past_user_queries:
                last_user_query = past_user_queries[-1]
                contextualized_query = f"{last_user_query} {query}"
                supplemental_chunks = self.retrieve_context(
                    query=contextualized_query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    document_id=document_id,
                )
                seen_points = {c.point_id for c in chunks}
                for sc in supplemental_chunks:
                    if sc.point_id not in seen_points:
                        chunks.append(sc)
                        seen_points.add(sc.point_id)

        # Step 2: Handle case where no relevant context exists
        if not chunks and not history:
            logger.info("No relevant chunks found for query '%s' in session '%s'", query[:40], session_id)
            insufficient_msg = (
                "Based on the indexed documents, I could not find information regarding your inquiry. "
                "Please verify that the relevant document has been uploaded or rephrase your question."
            )
            self.memory_service.add_turn(
                session_id=session_id,
                user_message=query,
                assistant_message=insufficient_msg,
            )
            return ChatResponse(
                session_id=session_id,
                response=insufficient_msg,
                sources_used=[],
                intent="document_rag",
            )

        # Convert ScoredChunks to SourceChunk metadata response models
        sources_used = [
            SourceChunk(
                document_id=c.document_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                score=round(c.score, 4),
                text_snippet=(
                    c.chunk_text[:200] + "..." if len(c.chunk_text) > 200 else c.chunk_text
                ),
            )
            for c in chunks
        ]

        # Step 3: Construct prompt manually with conversation history
        prompt = self.build_prompt(
            query=query,
            chunks=chunks,
            conversation_history=history,
        )

        # Step 4: Generate response via Gemini
        generated_answer = self.generate_response(prompt=prompt)

        # Step 5: Save interaction turn to Redis session memory
        self.memory_service.add_turn(
            session_id=session_id,
            user_message=query,
            assistant_message=generated_answer,
        )

        # Step 6: Return structured response
        return ChatResponse(
            session_id=session_id,
            response=generated_answer,
            sources_used=sources_used,
            intent="document_rag",
        )


# Singleton instance
rag_service = RAGService()


def get_rag_service() -> RAGService:
    """FastAPI dependency provider for RAGService."""
    return rag_service

