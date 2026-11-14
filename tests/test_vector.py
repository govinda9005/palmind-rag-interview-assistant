"""Unit and integration tests for embedding generation and Qdrant vector operations."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.services.chunking_service import TextChunk
from app.services.embedding_service import embedding_service
from app.services.vector_service import VectorService
from app.utils.exceptions import EmbeddingException, VectorStorageException


@pytest.fixture
def mock_qdrant_service() -> VectorService:
    """Fixture providing a VectorService backed by an isolated in-memory Qdrant instance."""
    client = QdrantClient(":memory:", check_compatibility=False)
    service = VectorService(client=client, collection_name="test_palmind_docs")
    return service


class TestEmbeddingGeneration:
    """Test suite for SentenceTransformer embedding generation."""

    def test_embedding_dimensions(self):
        """Verify that single text embedding produces exactly 384 dimensions."""
        vector = embedding_service.generate_embedding("PalmMind AI engineering test")
        assert isinstance(vector, list)
        assert len(vector) == 384
        assert all(isinstance(val, float) for val in vector)

    def test_batch_embedding_dimensions_and_count(self):
        """Verify that batch embedding maintains order, count, and 384 dimensions."""
        texts = [
            "First sentence regarding candidate interviews.",
            "Second sentence about RAG retrieval.",
            "Third sentence regarding vector databases.",
        ]
        vectors = embedding_service.generate_embeddings(texts)
        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == 384

    def test_empty_string_embedding_raises_exception(self):
        """Verify that generating an embedding for empty text raises EmbeddingException."""
        with pytest.raises(EmbeddingException) as exc_info:
            embedding_service.generate_embedding("")
        assert "empty or whitespace-only text" in exc_info.value.message

        with pytest.raises(EmbeddingException):
            embedding_service.generate_embedding("     \n\t   ")


class TestQdrantVectorIntegration:
    """Test suite for Qdrant collection, insertion, payload storage, and search."""

    def test_collection_initialization(self, mock_qdrant_service: VectorService):
        """Verify that Qdrant collection is initialized with 384 dimensions and Cosine metric."""
        service = mock_qdrant_service
        assert service.client.collection_exists(service.collection_name)

        collection_info = service.client.get_collection(service.collection_name)
        vectors_config = collection_info.config.params.vectors
        assert vectors_config.size == 384
        assert vectors_config.distance == Distance.COSINE

    def test_vector_insertion_and_count(self, mock_qdrant_service: VectorService):
        """Verify upserting chunks stores vectors and returns accurate point count."""
        service = mock_qdrant_service
        chunks = [
            TextChunk(
                chunk_index=0,
                content="Govinda Yadav is an AI engineer candidate.",
                character_count=42,
                start_char=0,
                end_char=42,
                strategy="fixed",
            ),
            TextChunk(
                chunk_index=1,
                content="The interview is scheduled for November 20.",
                character_count=43,
                start_char=43,
                end_char=86,
                strategy="fixed",
            ),
        ]
        vectors = embedding_service.generate_embeddings([c.content for c in chunks])

        stored_count = service.upsert_chunks(
            document_id=101,
            filename="candidate_info.txt",
            chunks=chunks,
            embeddings=vectors,
        )
        assert stored_count == 2

    def test_mismatched_chunks_and_embeddings_raises_exception(self, mock_qdrant_service: VectorService):
        """Verify mismatch between chunk count and embedding count raises VectorStorageException."""
        service = mock_qdrant_service
        chunks = [
            TextChunk(
                chunk_index=0,
                content="Sample text.",
                character_count=12,
                start_char=0,
                end_char=12,
                strategy="fixed",
            )
        ]
        mismatched_embeddings = [[0.1] * 384, [0.2] * 384]

        with pytest.raises(VectorStorageException) as exc_info:
            service.upsert_chunks(
                document_id=1,
                filename="test.txt",
                chunks=chunks,
                embeddings=mismatched_embeddings,
            )
        assert "Mismatch between chunks count" in exc_info.value.message

    def test_semantic_similarity_search_and_ranking(self, mock_qdrant_service: VectorService):
        """Verify semantic retrieval ranks the semantically relevant chunk highest."""
        service = mock_qdrant_service
        chunks = [
            TextChunk(
                chunk_index=0,
                content="Candidates can schedule an interview slot through natural conversation.",
                character_count=71,
                start_char=0,
                end_char=71,
                strategy="recursive",
            ),
            TextChunk(
                chunk_index=1,
                content="SQLite database handles persistent relational metadata storage.",
                character_count=63,
                start_char=72,
                end_char=135,
                strategy="recursive",
            ),
            TextChunk(
                chunk_index=2,
                content="Redis memory stores conversation session turns with expiration.",
                character_count=63,
                start_char=136,
                end_char=199,
                strategy="recursive",
            ),
        ]
        vectors = embedding_service.generate_embeddings([c.content for c in chunks])
        service.upsert_chunks(
            document_id=200,
            filename="system_overview.txt",
            chunks=chunks,
            embeddings=vectors,
        )

        # Query specifically related to interview booking
        query = "How do I book an interview appointment?"
        hits = service.search_by_text(query, top_k=2)

        assert len(hits) == 2
        # Top hit must be the interview scheduling chunk
        top_hit = hits[0]
        assert top_hit.chunk_index == 0
        assert "schedule an interview slot" in top_hit.chunk_text
        assert top_hit.score > hits[1].score

    def test_payload_metadata_integrity(self, mock_qdrant_service: VectorService):
        """Verify that all metadata fields are preserved intact in vector payloads."""
        service = mock_qdrant_service
        chunk = TextChunk(
            chunk_index=3,
            content="PalmMind technical evaluation payload content.",
            character_count=46,
            start_char=120,
            end_char=166,
            strategy="recursive",
        )
        vec = embedding_service.generate_embedding(chunk.content)

        service.upsert_chunks(
            document_id=77,
            filename="palmmind_test.pdf",
            chunks=[chunk],
            embeddings=[vec],
        )

        hits = service.search_by_text("technical evaluation payload", top_k=1)
        assert len(hits) == 1
        hit = hits[0]

        assert hit.document_id == 77
        assert hit.filename == "palmmind_test.pdf"
        assert hit.chunk_index == 3
        assert hit.chunk_text == chunk.content
        assert hit.chunking_strategy == "recursive"
        assert hit.character_count == 46
        assert hit.start_char == 120
        assert hit.end_char == 166
        assert isinstance(hit.point_id, str)
        assert hit.score > 0.0

    def test_document_id_filtering(self, mock_qdrant_service: VectorService):
        """Verify that similarity search filters correctly by document_id."""
        service = mock_qdrant_service
        c1 = TextChunk(chunk_index=0, content="Document Alpha candidate.", character_count=26, start_char=0, end_char=26, strategy="fixed")
        c2 = TextChunk(chunk_index=0, content="Document Beta candidate.", character_count=24, start_char=0, end_char=24, strategy="fixed")

        service.upsert_chunks(document_id=1, filename="alpha.txt", chunks=[c1], embeddings=embedding_service.generate_embeddings([c1.content]))
        service.upsert_chunks(document_id=2, filename="beta.txt", chunks=[c2], embeddings=embedding_service.generate_embeddings([c2.content]))

        # Search with filter for document_id=2
        query_vec = embedding_service.generate_embedding("candidate")
        filtered_hits = service.search_similar(query_vector=query_vec, top_k=5, document_id=2)

        assert len(filtered_hits) == 1
        assert filtered_hits[0].document_id == 2
        assert filtered_hits[0].filename == "beta.txt"

    def test_delete_document_vectors(self, mock_qdrant_service: VectorService):
        """Verify that all vector points for a given document_id can be deleted."""
        service = mock_qdrant_service
        c1 = TextChunk(chunk_index=0, content="Doc to delete chunk.", character_count=20, start_char=0, end_char=20, strategy="fixed")
        service.upsert_chunks(document_id=999, filename="delete_me.txt", chunks=[c1], embeddings=embedding_service.generate_embeddings([c1.content]))

        # Verify exists
        hits = service.search_by_text("delete chunk", top_k=1, document_id=999)
        assert len(hits) == 1

        # Delete
        success = service.delete_document_vectors(document_id=999)
        assert success is True

        # Verify no longer present
        hits_after = service.search_by_text("delete chunk", top_k=1, document_id=999)
        assert len(hits_after) == 0

