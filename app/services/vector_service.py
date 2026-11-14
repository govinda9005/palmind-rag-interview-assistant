"""Vector service integrating Qdrant for collection management, storage, and similarity search."""

import logging
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings
from app.services.chunking_service import TextChunk
from app.services.embedding_service import embedding_service
from app.utils.exceptions import VectorStorageException

logger = logging.getLogger("palmind.vector")


class ScoredChunk(BaseModel):
    """Represents a retrieved document chunk with semantic similarity score and metadata."""

    point_id: str = Field(..., description="Unique Qdrant point UUID")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    document_id: int = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original filename of the document")
    chunk_index: int = Field(..., description="Index of this chunk in sequence")
    chunk_text: str = Field(..., description="Text content stored in vector payload")
    chunking_strategy: str = Field(..., description="Strategy used to produce chunk")
    character_count: int = Field(..., description="Length of chunk content")
    start_char: int = Field(..., description="Character offset start in document")
    end_char: int = Field(..., description="Character offset end in document")


class VectorService:
    """Service encapsulating Qdrant vector database operations."""

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._collection_name = collection_name or settings.QDRANT_COLLECTION
        self._dimension = settings.EMBEDDING_DIMENSION

        if client is not None:
            self._client = client
        else:
            self._client = self._init_client(settings)

        # Ensure target collection is ready
        self.ensure_collection()

    def _init_client(self, settings) -> QdrantClient:
        """Initialize Qdrant client connecting to remote host with fallback to local disk storage."""
        try:
            # First attempt connection to remote Qdrant (e.g. Docker container)
            remote_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=1.5,
                check_compatibility=False,
            )
            # Ping remote Qdrant
            remote_client.get_collections()
            logger.info(
                "Connected to remote Qdrant at %s:%d",
                settings.QDRANT_HOST,
                settings.QDRANT_PORT,
            )
            return remote_client
        except Exception as exc:
            logger.warning(
                "Remote Qdrant host '%s:%d' unavailable (%s). "
                "Falling back to local persistent Qdrant storage at './qdrant_storage'.",
                settings.QDRANT_HOST,
                settings.QDRANT_PORT,
                str(exc),
            )
            try:
                # Persistent local real Qdrant storage
                local_client = QdrantClient(path="./qdrant_storage", check_compatibility=False)
                logger.info("Initialized local persistent Qdrant at './qdrant_storage'")
                return local_client
            except Exception as local_exc:
                logger.error("Failed to initialize local Qdrant: %s", local_exc)
                raise VectorStorageException(
                    message=f"Failed to initialize Qdrant client: {str(local_exc)}",
                    details={"error": str(local_exc)},
                )

    @property
    def client(self) -> QdrantClient:
        """Return the underlying QdrantClient."""
        return self._client

    @property
    def collection_name(self) -> str:
        """Return the active collection name."""
        return self._collection_name

    def ensure_collection(self, collection_name: Optional[str] = None) -> bool:
        """Idempotently ensure the target vector collection exists with cosine distance.

        Args:
            collection_name: Optional collection override.

        Returns:
            True if collection exists or was created successfully.
        """
        col = collection_name or self._collection_name
        try:
            if not self._client.collection_exists(col):
                logger.info(
                    "Collection '%s' does not exist. Creating with dim=%d, metric=Cosine",
                    col,
                    self._dimension,
                )
                self._client.create_collection(
                    collection_name=col,
                    vectors_config=VectorParams(
                        size=self._dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Collection '%s' created successfully", col)
            return True
        except Exception as exc:
            logger.error("Failed to ensure collection '%s': %s", col, exc)
            raise VectorStorageException(
                message=f"Failed to create or verify Qdrant collection '{col}': {str(exc)}",
                details={"collection": col, "error": str(exc)},
            )

    def upsert_chunks(
        self,
        document_id: int,
        filename: str,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
        collection_name: Optional[str] = None,
    ) -> int:
        """Store chunk vectors and rich metadata payloads in Qdrant.

        Args:
            document_id: Unique integer ID of parent document from SQLite.
            filename: Original file name of the document.
            chunks: Ordered list of TextChunk objects.
            embeddings: Parallel list of 384-dimensional dense vectors.
            collection_name: Optional collection override.

        Returns:
            Number of points successfully stored in Qdrant.

        Raises:
            VectorStorageException: If inputs are mismatched or Qdrant upsert fails.
        """
        if len(chunks) != len(embeddings):
            raise VectorStorageException(
                message=(
                    f"Mismatch between chunks count ({len(chunks)}) and "
                    f"embeddings count ({len(embeddings)})"
                ),
                details={"chunks_count": len(chunks), "embeddings_count": len(embeddings)},
            )

        if not chunks:
            return 0

        col = collection_name or self._collection_name
        self.ensure_collection(col)

        points: List[PointStruct] = []
        for chunk, vector in zip(chunks, embeddings):
            # Deterministic UUID prevents duplicate points when re-indexing
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{chunk.chunk_index}"))

            payload: Dict[str, Any] = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.content,
                "chunking_strategy": chunk.strategy,
                "character_count": chunk.character_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }

            points.append(
                PointStruct(
                    id=point_uuid,
                    vector=vector,
                    payload=payload,
                )
            )

        try:
            self._client.upsert(
                collection_name=col,
                points=points,
                wait=True,
            )
            logger.info(
                "Successfully stored %d vector points for document %d ('%s') in '%s'",
                len(points),
                document_id,
                filename,
                col,
            )
            return len(points)
        except Exception as exc:
            logger.error("Failed to upsert points for document %d in '%s': %s", document_id, col, exc)
            raise VectorStorageException(
                message=f"Failed to upsert vectors into Qdrant collection '{col}': {str(exc)}",
                details={"document_id": document_id, "filename": filename, "error": str(exc)},
            )

    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        document_id: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> List[ScoredChunk]:
        """Perform semantic similarity search using cosine distance against Qdrant vectors.

        Args:
            query_vector: 384-dimensional query embedding.
            top_k: Maximum number of relevant chunks to retrieve.
            score_threshold: Optional minimum cosine similarity score cutoff.
            document_id: Optional filter to restrict retrieval to a specific document.
            collection_name: Optional collection override.

        Returns:
            List of ScoredChunk models sorted by descending relevance score.
        """
        if len(query_vector) != self._dimension:
            raise VectorStorageException(
                message=f"Query vector dimension mismatch: expected {self._dimension}, got {len(query_vector)}",
                details={"expected": self._dimension, "received": len(query_vector)},
            )

        col = collection_name or self._collection_name
        self.ensure_collection(col)

        # Construct optional payload filter
        query_filter: Optional[Filter] = None
        if document_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        try:
            response = self._client.query_points(
                collection_name=col,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            scored_chunks: List[ScoredChunk] = []
            for hit in response.points:
                payload = hit.payload or {}
                scored_chunks.append(
                    ScoredChunk(
                        point_id=str(hit.id),
                        score=float(hit.score),
                        document_id=int(payload.get("document_id", 0)),
                        filename=str(payload.get("filename", "")),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        chunk_text=str(payload.get("chunk_text", "")),
                        chunking_strategy=str(payload.get("chunking_strategy", "")),
                        character_count=int(payload.get("character_count", 0)),
                        start_char=int(payload.get("start_char", 0)),
                        end_char=int(payload.get("end_char", 0)),
                    )
                )

            logger.info("Semantic search in '%s' retrieved %d hits (top_k=%d)", col, len(scored_chunks), top_k)
            return scored_chunks

        except Exception as exc:
            logger.error("Qdrant similarity search failed in '%s': %s", col, exc)
            raise VectorStorageException(
                message=f"Vector search failed in collection '{col}': {str(exc)}",
                details={"collection": col, "top_k": top_k, "error": str(exc)},
            )

    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        document_id: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> List[ScoredChunk]:
        """Convenience method: embed textual query and retrieve semantically similar chunks."""
        query_vector = embedding_service.generate_embedding(query)
        return self.search_similar(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            document_id=document_id,
            collection_name=collection_name,
        )

    def delete_document_vectors(
        self,
        document_id: int,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Remove all vector points belonging to a specific document ID.

        Args:
            document_id: ID of the document whose chunks should be purged.
            collection_name: Optional collection override.

        Returns:
            True if deletion completed successfully.
        """
        col = collection_name or self._collection_name
        try:
            self._client.delete(
                collection_name=col,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            logger.info("Deleted vector points for document %d in collection '%s'", document_id, col)
            return True
        except Exception as exc:
            logger.error("Failed to delete points for document %d: %s", document_id, exc)
            raise VectorStorageException(
                message=f"Failed to delete document vectors from '{col}': {str(exc)}",
                details={"document_id": document_id, "error": str(exc)},
            )


# Singleton instance for application reuse
vector_service = VectorService()


def get_vector_service() -> VectorService:
    """Return the singleton instance of VectorService."""
    return vector_service

