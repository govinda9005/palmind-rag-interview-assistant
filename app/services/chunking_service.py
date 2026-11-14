"""Document chunking service implementing Fixed-size and Recursive character chunking strategies."""

import logging
import re
from typing import List, Optional
from pydantic import BaseModel, Field

from app.api.schemas.document import ChunkingStrategyEnum
from app.core.config import get_settings

logger = logging.getLogger("palmind.chunking")


class TextChunk(BaseModel):
    """Represents an extracted text chunk with positional and structural metadata."""

    chunk_index: int = Field(..., description="Zero-based index of chunk in document sequence")
    content: str = Field(..., description="Raw text content of the chunk")
    character_count: int = Field(..., description="Total character count in the chunk content")
    start_char: int = Field(..., description="Starting character index in original source text")
    end_char: int = Field(..., description="Ending character index in original source text")
    strategy: str = Field(..., description="Chunking strategy used ('fixed' or 'recursive')")


class ChunkingService:
    """Service providing fixed-size and recursive character chunking strategies."""

    # Default hierarchical separators in descending priority
    DEFAULT_SEPARATORS: List[str] = [
        "\n\n",   # Paragraph boundaries
        "\n",     # Line breaks
        ". ",     # Sentence end with space
        "? ",     # Question end with space
        "! ",     # Exclamation end with space
        " ",      # Word boundaries
        "",       # Character-level fallback
    ]

    def __init__(self) -> None:
        settings = get_settings()
        self.default_fixed_size = settings.DEFAULT_FIXED_CHUNK_SIZE
        self.default_fixed_overlap = settings.DEFAULT_FIXED_CHUNK_OVERLAP
        self.default_recursive_size = settings.DEFAULT_RECURSIVE_CHUNK_SIZE
        self.default_recursive_overlap = settings.DEFAULT_RECURSIVE_CHUNK_OVERLAP

    def chunk_text(
        self,
        text: str,
        strategy: ChunkingStrategyEnum = ChunkingStrategyEnum.RECURSIVE,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[TextChunk]:
        """Split source text into chunks using the selected strategy.

        Args:
            text: The normalized document text to chunk.
            strategy: 'fixed' or 'recursive'.
            chunk_size: Target character size for each chunk.
            chunk_overlap: Number of overlapping characters between adjacent chunks.

        Returns:
            List of TextChunk objects ordered sequentially.
        """
        if not text or not text.strip():
            return []

        if strategy == ChunkingStrategyEnum.FIXED:
            size = chunk_size or self.default_fixed_size
            overlap = chunk_overlap if chunk_overlap is not None else self.default_fixed_overlap
            return self.fixed_size_chunking(text=text, chunk_size=size, chunk_overlap=overlap)

        elif strategy == ChunkingStrategyEnum.RECURSIVE:
            size = chunk_size or self.default_recursive_size
            overlap = chunk_overlap if chunk_overlap is not None else self.default_recursive_overlap
            return self.recursive_character_chunking(
                text=text,
                chunk_size=size,
                chunk_overlap=overlap,
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

    def fixed_size_chunking(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[TextChunk]:
        """Strategy 1: Fixed-size sliding-window chunking.

        Creates sequential chunks of fixed character length with configurable overlap.
        Preserves original text content and eliminates empty chunks.
        """
        self._validate_params(chunk_size, chunk_overlap)

        if not text or not text.strip():
            return []

        chunks: List[TextChunk] = []
        step = chunk_size - chunk_overlap
        text_len = len(text)
        start = 0
        chunk_index = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            content = text[start:end]

            # Only append if content contains non-whitespace characters
            if content.strip():
                chunks.append(
                    TextChunk(
                        chunk_index=chunk_index,
                        content=content,
                        character_count=len(content),
                        start_char=start,
                        end_char=end,
                        strategy=ChunkingStrategyEnum.FIXED.value,
                    )
                )
                chunk_index += 1

            if end >= text_len:
                break

            start += step

        logger.info(
            "Fixed-size chunking produced %d chunks (size=%d, overlap=%d)",
            len(chunks),
            chunk_size,
            chunk_overlap,
        )
        return chunks

    def recursive_character_chunking(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: Optional[List[str]] = None,
    ) -> List[TextChunk]:
        """Strategy 2: Recursive character chunking.

        Recursively splits text using prioritized structural boundaries:
        paragraphs -> newlines -> sentences -> words -> characters.
        Recombines split pieces with an overlap buffer while respecting chunk_size.
        """
        self._validate_params(chunk_size, chunk_overlap)

        if not text or not text.strip():
            return []

        # If text is already smaller than or equal to chunk size, return single chunk
        if len(text) <= chunk_size:
            return [
                TextChunk(
                    chunk_index=0,
                    content=text,
                    character_count=len(text),
                    start_char=0,
                    end_char=len(text),
                    strategy=ChunkingStrategyEnum.RECURSIVE.value,
                )
            ]

        seps = separators or self.DEFAULT_SEPARATORS
        raw_pieces = self._recursive_split(text=text, separators=seps, chunk_size=chunk_size)

        # Merge pieces into chunks respecting chunk_size and chunk_overlap
        chunks = self._merge_pieces_with_overlap(
            pieces=raw_pieces,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_text=text,
        )

        logger.info(
            "Recursive chunking produced %d chunks (size=%d, overlap=%d)",
            len(chunks),
            chunk_size,
            chunk_overlap,
        )
        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        chunk_size: int,
    ) -> List[str]:
        """Split text recursively using the first separator that divides the text."""
        if not text:
            return []

        if len(text) <= chunk_size:
            return [text]

        # Find the first separator present in the text
        chosen_separator: Optional[str] = None
        remaining_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                chosen_separator = ""
                remaining_separators = []
                break
            if sep in text:
                chosen_separator = sep
                remaining_separators = separators[i + 1:]
                break

        # Fallback to atomic character splitting when no separator applies
        if chosen_separator is None or chosen_separator == "":
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Split preserving non-empty parts
        parts = text.split(chosen_separator)
        pieces: List[str] = []

        for idx, part in enumerate(parts):
            if not part:
                continue

            # Re-attach separator to all parts except the last one to preserve text structure
            piece_with_sep = part + chosen_separator if idx < len(parts) - 1 else part

            if len(piece_with_sep) > chunk_size and remaining_separators:
                # Recurse with lower priority separators
                sub_pieces = self._recursive_split(
                    text=piece_with_sep,
                    separators=remaining_separators,
                    chunk_size=chunk_size,
                )
                pieces.extend(sub_pieces)
            else:
                pieces.append(piece_with_sep)

        return pieces

    def _merge_pieces_with_overlap(
        self,
        pieces: List[str],
        chunk_size: int,
        chunk_overlap: int,
        source_text: str,
    ) -> List[TextChunk]:
        """Merge split pieces into final chunks respecting chunk_size and maintaining overlap."""
        final_chunks: List[TextChunk] = []
        current_chunk_pieces: List[str] = []
        current_len = 0
        search_cursor = 0

        for piece in pieces:
            piece_len = len(piece)

            # If adding this piece exceeds chunk_size and we already have accumulated pieces
            if current_chunk_pieces and (current_len + piece_len > chunk_size):
                combined_text = "".join(current_chunk_pieces)
                if combined_text.strip():
                    chunk_obj, search_cursor = self._create_chunk(
                        content=combined_text,
                        chunk_index=len(final_chunks),
                        source_text=source_text,
                        search_start=search_cursor,
                        strategy=ChunkingStrategyEnum.RECURSIVE.value,
                    )
                    final_chunks.append(chunk_obj)

                # Build overlap buffer from the end of current_chunk_pieces
                overlap_pieces: List[str] = []
                overlap_len = 0
                for p in reversed(current_chunk_pieces):
                    if overlap_len + len(p) <= chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break

                current_chunk_pieces = overlap_pieces
                current_len = overlap_len

            # In the rare event a single piece itself exceeds chunk_size (e.g. monolithic token)
            if piece_len > chunk_size:
                if current_chunk_pieces:
                    combined_text = "".join(current_chunk_pieces)
                    if combined_text.strip():
                        chunk_obj, search_cursor = self._create_chunk(
                            content=combined_text,
                            chunk_index=len(final_chunks),
                            source_text=source_text,
                            search_start=search_cursor,
                            strategy=ChunkingStrategyEnum.RECURSIVE.value,
                        )
                        final_chunks.append(chunk_obj)
                    current_chunk_pieces = []
                    current_len = 0

                # Slice large piece into sub-chunks
                sub_start = 0
                while sub_start < piece_len:
                    sub_end = min(sub_start + chunk_size, piece_len)
                    sub_content = piece[sub_start:sub_end]
                    if sub_content.strip():
                        chunk_obj, search_cursor = self._create_chunk(
                            content=sub_content,
                            chunk_index=len(final_chunks),
                            source_text=source_text,
                            search_start=search_cursor,
                            strategy=ChunkingStrategyEnum.RECURSIVE.value,
                        )
                        final_chunks.append(chunk_obj)
                    if sub_end >= piece_len:
                        break
                    sub_start += (chunk_size - chunk_overlap)
                continue

            current_chunk_pieces.append(piece)
            current_len += piece_len

        # Flush remaining pieces
        if current_chunk_pieces:
            combined_text = "".join(current_chunk_pieces)
            if combined_text.strip():
                chunk_obj, _ = self._create_chunk(
                    content=combined_text,
                    chunk_index=len(final_chunks),
                    source_text=source_text,
                    search_start=search_cursor,
                    strategy=ChunkingStrategyEnum.RECURSIVE.value,
                )
                final_chunks.append(chunk_obj)

        return final_chunks

    @staticmethod
    def _create_chunk(
        content: str,
        chunk_index: int,
        source_text: str,
        search_start: int,
        strategy: str,
    ) -> tuple[TextChunk, int]:
        """Locate piece positions in source text and create TextChunk instance."""
        start_char = source_text.find(content, search_start)
        if start_char == -1:
            # Fallback search from beginning if search_start was displaced by overlap
            start_char = source_text.find(content)
        if start_char == -1:
            start_char = 0

        end_char = start_char + len(content)
        # Advance search cursor slightly to avoid matching earlier occurrences
        next_search_cursor = max(0, start_char + 1)

        chunk = TextChunk(
            chunk_index=chunk_index,
            content=content,
            character_count=len(content),
            start_char=start_char,
            end_char=end_char,
            strategy=strategy,
        )
        return chunk, next_search_cursor

    @staticmethod
    def _validate_params(chunk_size: int, chunk_overlap: int) -> None:
        """Validate chunk size and overlap parameters."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )


# Singleton instance for convenience
chunking_service = ChunkingService()

