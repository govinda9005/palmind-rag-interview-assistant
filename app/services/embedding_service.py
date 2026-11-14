"""Embedding service utilizing Sentence Transformers with all-MiniLM-L6-v2."""

import logging
from typing import List, Optional
import numpy as np

from app.core.config import get_settings
from app.utils.exceptions import EmbeddingException

logger = logging.getLogger("palmind.embedding")


class EmbeddingService:
    """Service responsible for generating dense vector embeddings using Sentence Transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._expected_dimension = dimension or settings.EMBEDDING_DIMENSION
        self._model = None

    @property
    def model_name(self) -> str:
        """Return the active model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self._expected_dimension

    def _get_model(self):
        """Lazy-load the SentenceTransformer model to optimize startup time."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading SentenceTransformer model: %s", self._model_name)
                self._model = SentenceTransformer(self._model_name)
                logger.info("SentenceTransformer model '%s' loaded successfully", self._model_name)
            except Exception as exc:
                logger.error("Failed to load embedding model '%s': %s", self._model_name, exc)
                raise EmbeddingException(
                    message=f"Failed to load SentenceTransformer model '{self._model_name}': {str(exc)}",
                    details={"model_name": self._model_name, "error": str(exc)},
                )
        return self._model

    def generate_embedding(self, text: str) -> List[float]:
        """Generate a single 384-dimensional embedding for an input string.

        Args:
            text: Input string to embed.

        Returns:
            List of float values representing the dense embedding vector.

        Raises:
            EmbeddingException: If embedding generation fails or input is empty.
        """
        if not text or not text.strip():
            raise EmbeddingException(
                message="Cannot generate embedding for empty or whitespace-only text.",
                details={"input_length": len(text) if text else 0},
            )

        try:
            model = self._get_model()
            embedding = model.encode(text, normalize_embeddings=True)
            vector = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)

            if len(vector) != self._expected_dimension:
                raise EmbeddingException(
                    message=(
                        f"Expected embedding dimension {self._expected_dimension}, "
                        f"but model produced {len(vector)}."
                    ),
                    details={
                        "expected": self._expected_dimension,
                        "received": len(vector),
                    },
                )
            return vector
        except EmbeddingException:
            raise
        except Exception as exc:
            logger.error("Error generating embedding: %s", exc)
            raise EmbeddingException(
                message=f"Failed to generate embedding: {str(exc)}",
                details={"error": str(exc)},
            )

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Generate embeddings in batches for a list of text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process in parallel per forward pass.

        Returns:
            List of embedding vectors, matching the ordering of input texts.

        Raises:
            EmbeddingException: If generation fails or any text is empty.
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() for t in texts]
        if any(len(t) == 0 for t in cleaned_texts):
            raise EmbeddingException(
                message="Cannot generate embeddings: input list contains empty strings.",
                details={"count": len(texts)},
            )

        try:
            model = self._get_model()
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            results: List[List[float]] = []
            for emb in embeddings:
                vec = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                if len(vec) != self._expected_dimension:
                    raise EmbeddingException(
                        message=f"Embedding dimension mismatch: expected {self._expected_dimension}, got {len(vec)}",
                        details={"expected": self._expected_dimension, "received": len(vec)},
                    )
                results.append(vec)

            logger.info("Generated %d embeddings using %s", len(results), self._model_name)
            return results
        except EmbeddingException:
            raise
        except Exception as exc:
            logger.error("Batch embedding generation failed: %s", exc)
            raise EmbeddingException(
                message=f"Failed to generate batch embeddings: {str(exc)}",
                details={"error": str(exc), "batch_size": batch_size},
            )


# Singleton instance for application reuse
embedding_service = EmbeddingService()

