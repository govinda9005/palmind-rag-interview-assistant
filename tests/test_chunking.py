"""Unit tests for fixed-size and recursive character chunking strategies."""

import pytest
from app.api.schemas.document import ChunkingStrategyEnum
from app.services.chunking_service import chunking_service


SAMPLE_REALISTIC_TEXT = (
    "PalmMind AI is an artificial intelligence engineering company specializing in enterprise conversational "
    "workflows. The company builds reliable multi-turn AI assistants designed for automated candidate screening "
    "and interview scheduling.\n\n"
    "In document-based RAG architectures, chunking strategy fundamentally influences retrieval quality. "
    "Fixed-size chunking blindly splits text at a predetermined character offset regardless of syntax, "
    "often slicing sentences and words mid-phrase.\n\n"
    "In contrast, recursive character chunking respects paragraph and sentence boundaries, preserving full "
    "semantic units. As a result, retrieved vectors provide substantially more coherent context to the LLM."
)


class TestFixedChunking:
    """Test suite for Fixed-size sliding window chunking strategy."""

    def test_fixed_chunking_basic(self):
        """Verify basic fixed-size chunking properties."""
        chunks = chunking_service.chunk_text(
            SAMPLE_REALISTIC_TEXT,
            strategy=ChunkingStrategyEnum.FIXED,
            chunk_size=150,
            chunk_overlap=30,
        )

        assert len(chunks) > 0
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx
            assert chunk.strategy == "fixed"
            assert chunk.character_count == len(chunk.content)
            assert chunk.character_count <= 150
            # Slices must match source text
            assert chunk.content == SAMPLE_REALISTIC_TEXT[chunk.start_char:chunk.end_char]
            assert len(chunk.content.strip()) > 0

    def test_fixed_chunking_overlap_exactness(self):
        """Verify that fixed-size consecutive chunks overlap by exactly the configured offset."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
        chunk_size = 20
        chunk_overlap = 5
        chunks = chunking_service.fixed_size_chunking(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        assert len(chunks) > 1
        for i in range(len(chunks) - 1):
            curr_tail = chunks[i].content[-chunk_overlap:]
            next_head = chunks[i + 1].content[:chunk_overlap]
            assert curr_tail == next_head

    def test_fixed_chunking_no_empty_chunks(self):
        """Verify no empty or whitespace-only chunks are returned."""
        text = "Some words.       \n\n\n\n       More words here."
        chunks = chunking_service.fixed_size_chunking(text, chunk_size=20, chunk_overlap=5)
        for c in chunks:
            assert len(c.content.strip()) > 0


class TestRecursiveChunking:
    """Test suite for Recursive character chunking strategy."""

    def test_recursive_chunking_basic(self):
        """Verify recursive chunking produces non-empty, sequential chunks within size limit."""
        chunks = chunking_service.chunk_text(
            SAMPLE_REALISTIC_TEXT,
            strategy=ChunkingStrategyEnum.RECURSIVE,
            chunk_size=200,
            chunk_overlap=40,
        )

        assert len(chunks) > 0
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx
            assert chunk.strategy == "recursive"
            assert chunk.character_count == len(chunk.content)
            assert chunk.character_count <= 200
            assert len(chunk.content.strip()) > 0

    def test_recursive_chunking_preserves_paragraphs(self):
        """Verify that recursive chunking prioritizes paragraph breaks."""
        p1 = "Paragraph one contains complete context regarding the candidate."
        p2 = "Paragraph two describes the interview schedule and requirements."
        text = f"{p1}\n\n{p2}"

        # Size large enough for each paragraph individually, but smaller than both combined
        chunk_size = len(p1) + 10
        chunks = chunking_service.recursive_character_chunking(
            text,
            chunk_size=chunk_size,
            chunk_overlap=10,
        )

        assert len(chunks) == 2
        assert p1 in chunks[0].content
        assert p2 in chunks[1].content

    def test_recursive_chunking_preserves_sentences(self):
        """Verify that recursive chunking respects sentence boundaries when paragraphs exceed size."""
        s1 = "This is the first clean sentence."
        s2 = "This is the second clean sentence."
        s3 = "This is the third clean sentence."
        text = f"{s1} {s2} {s3}"

        chunk_size = len(s1) + len(s2) + 2
        chunks = chunking_service.recursive_character_chunking(
            text,
            chunk_size=chunk_size,
            chunk_overlap=5,
        )

        assert len(chunks) >= 2
        # First chunk should contain complete sentences without cutting in the middle of a word
        assert s1 in chunks[0].content


class TestChunkingComparisonAndEdgeCases:
    """Compare strategies on realistic text and verify boundary / error conditions."""

    def test_realistic_behavior_difference_demonstration(self):
        """Demonstrate that Fixed chunking cuts mid-phrase while Recursive preserves sentences."""
        chunk_size = 200
        overlap = 30

        fixed_chunks = chunking_service.chunk_text(
            SAMPLE_REALISTIC_TEXT,
            strategy=ChunkingStrategyEnum.FIXED,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

        recursive_chunks = chunking_service.chunk_text(
            SAMPLE_REALISTIC_TEXT,
            strategy=ChunkingStrategyEnum.RECURSIVE,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

        # Fixed chunking slices right through words or sentences:
        fixed_slice_cuts_word = any(
            not chunk.content.endswith((".", "!", "?", "\n", " "))
            for chunk in fixed_chunks[:-1]
        )
        assert fixed_slice_cuts_word, "Fixed chunking should slice arbitrarily at character offsets"

        # Recursive chunking keeps sentences and natural boundaries intact:
        recursive_respects_boundaries = all(
            chunk.content.strip().endswith((".", "!", "?")) or "\n" in chunk.content
            for chunk in recursive_chunks
        )
        assert recursive_respects_boundaries, "Recursive chunking should preserve semantic sentence/paragraph boundaries"

    def test_text_shorter_than_chunk_size(self):
        """Verify that text shorter than chunk_size returns a single chunk."""
        short_text = "Short text under chunk limit."
        fixed = chunking_service.fixed_size_chunking(short_text, chunk_size=100, chunk_overlap=10)
        assert len(fixed) == 1
        assert fixed[0].content == short_text
        assert fixed[0].chunk_index == 0

        recursive = chunking_service.recursive_character_chunking(short_text, chunk_size=100, chunk_overlap=10)
        assert len(recursive) == 1
        assert recursive[0].content == short_text
        assert recursive[0].chunk_index == 0

    def test_empty_and_whitespace_text(self):
        """Verify that empty or whitespace-only text returns empty list."""
        assert chunking_service.chunk_text("", strategy=ChunkingStrategyEnum.FIXED) == []
        assert chunking_service.chunk_text("    \n\n   ", strategy=ChunkingStrategyEnum.RECURSIVE) == []

    def test_invalid_parameters_raise_value_error(self):
        """Verify that invalid chunk sizes and overlaps raise ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunking_service.fixed_size_chunking("Sample", chunk_size=0, chunk_overlap=0)

        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            chunking_service.fixed_size_chunking("Sample", chunk_size=100, chunk_overlap=-5)

        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            chunking_service.fixed_size_chunking("Sample", chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            chunking_service.recursive_character_chunking("Sample", chunk_size=50, chunk_overlap=60)

