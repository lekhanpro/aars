"""Recursive character text splitter."""

from __future__ import annotations

import copy
from typing import Sequence

import structlog

logger = structlog.get_logger(__name__)

# Default separator hierarchy: prefer splitting on semantic boundaries first.
_DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


class RecursiveChunker:
    """Split text into overlapping chunks using a recursive separator strategy.

    The splitter tries the first separator; for any resulting piece that still
    exceeds ``chunk_size`` it recurses to the next separator, all the way down
    to individual characters (``""``).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Sequence[str] = _DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = tuple(separators)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, metadata: dict) -> list[dict]:
        """Split *text* into chunks and attach *metadata* to each one.

        Each returned dict has:
        - ``content``: the chunk text.
        - ``metadata``: a *copy* of the incoming metadata dict with an added
          ``chunk_index`` (0-based).

        Args:
            text: The source text to split.
            metadata: Base metadata dict to copy onto every chunk.

        Returns:
            Ordered list of chunk dicts.
        """
        if not text:
            return []

        raw_chunks = self._split(text, self._separators)
        merged = self._merge_with_overlap(raw_chunks)

        results: list[dict] = []
        for idx, chunk_text in enumerate(merged):
            chunk_meta = copy.deepcopy(metadata)
            chunk_meta["chunk_index"] = idx
            results.append({"content": chunk_text, "metadata": chunk_meta})

        logger.debug(
            "text_chunked",
            source=metadata.get("source", "unknown"),
            input_length=len(text),
            num_chunks=len(results),
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Recursively split *text* using the first applicable separator."""
        if not text:
            return []

        # Base case: character-level split when no separators remain.
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        # The empty-string separator means split every character.
        if separator == "":
            pieces = list(text)
        else:
            pieces = text.split(separator)

        good_chunks: list[str] = []
        current_piece = ""

        for piece in pieces:
            candidate = (
                piece
                if not current_piece
                else current_piece + separator + piece
            )
            if len(candidate) <= self._chunk_size:
                current_piece = candidate
            else:
                # Flush what we had accumulated so far.
                if current_piece:
                    good_chunks.append(current_piece)
                # If the single piece itself exceeds the limit, recurse
                # with a finer separator.
                if len(piece) > self._chunk_size:
                    good_chunks.extend(
                        self._split(piece, remaining_separators)
                    )
                else:
                    current_piece = piece
                    continue
                current_piece = ""

        if current_piece:
            good_chunks.append(current_piece)

        return good_chunks

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Re-merge small chunks and apply overlap between adjacent ones."""
        if not chunks:
            return []

        merged: list[str] = []
        for chunk in chunks:
            if merged and len(merged[-1]) + len(chunk) + 1 <= self._chunk_size:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        if self._chunk_overlap == 0 or len(merged) <= 1:
            return merged

        # Build overlapping windows.
        overlapped: list[str] = [merged[0]]
        for i in range(1, len(merged)):
            prev = merged[i - 1]
            overlap_text = prev[-self._chunk_overlap:]
            # Find a clean break point (space) within the overlap region to
            # avoid splitting mid-word.
            space_idx = overlap_text.find(" ")
            if space_idx != -1:
                overlap_text = overlap_text[space_idx + 1:]
            combined = overlap_text + " " + merged[i] if overlap_text else merged[i]
            overlapped.append(combined)

        return overlapped
