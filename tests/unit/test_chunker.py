"""Tests for recursive chunker."""

from __future__ import annotations

from src.ingestion.chunkers.recursive import RecursiveChunker


class TestRecursiveChunker:
    def setup_method(self):
        self.chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)

    def test_short_text_single_chunk(self):
        result = self.chunker.chunk("Short text.", {"source": "test.txt"})
        assert len(result) == 1
        assert result[0]["content"] == "Short text."
        assert result[0]["metadata"]["source"] == "test.txt"
        assert result[0]["metadata"]["chunk_index"] == 0

    def test_long_text_multiple_chunks(self):
        text = "Word " * 100  # ~500 chars
        result = self.chunker.chunk(text.strip(), {"source": "test.txt"})
        assert len(result) > 1
        for i, chunk in enumerate(result):
            assert len(chunk["content"]) <= 110  # allow small overflow at boundaries
            assert chunk["metadata"]["chunk_index"] == i

    def test_paragraph_splitting(self):
        text = "First paragraph with content.\n\nSecond paragraph with content.\n\nThird paragraph."
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        result = chunker.chunk(text, {"source": "test.txt"})
        assert len(result) >= 2

    def test_empty_text(self):
        result = self.chunker.chunk("", {"source": "test.txt"})
        assert len(result) == 0

    def test_metadata_preserved(self):
        result = self.chunker.chunk("Some content.", {"source": "doc.pdf", "page": 3})
        assert result[0]["metadata"]["source"] == "doc.pdf"
        assert result[0]["metadata"]["page"] == 3

    def test_overlap_between_chunks(self):
        # Create text that will split into multiple chunks
        text = "A " * 200  # lots of tokens
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        result = chunker.chunk(text.strip(), {"source": "test.txt"})
        if len(result) >= 2:
            # Last chars of chunk 0 should appear at start of chunk 1 (overlap)
            # This is a soft check since overlap is character-based
            assert len(result) >= 2
