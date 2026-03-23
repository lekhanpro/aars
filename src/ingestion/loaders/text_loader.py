"""Plain-text document loader."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# Encodings attempted in order of likelihood.
_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


class TextLoader:
    """Load plain-text (and Markdown) files into a single content block."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".log"}

    async def load(self, file_bytes: bytes, filename: str) -> list[dict]:
        """Decode raw bytes and return a single-element content list.

        Multiple encodings are attempted so that common non-UTF-8 files still
        succeed.

        Args:
            file_bytes: Raw bytes of the text file.
            filename: Original filename for metadata attribution.

        Returns:
            Single-element list of ``{content, metadata}`` dicts.

        Raises:
            ValueError: If the file is empty or entirely whitespace.
            RuntimeError: If none of the candidate encodings can decode the
                bytes.
        """
        text: str | None = None
        used_encoding: str | None = None

        for encoding in _ENCODINGS:
            try:
                text = file_bytes.decode(encoding)
                used_encoding = encoding
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if text is None:
            logger.error("text_decode_failed", filename=filename)
            raise RuntimeError(
                f"Failed to decode '{filename}' with any of the supported "
                f"encodings: {', '.join(_ENCODINGS)}"
            )

        stripped = text.strip()
        if not stripped:
            raise ValueError(f"Text file '{filename}' is empty or contains only whitespace.")

        logger.info(
            "text_loaded",
            filename=filename,
            encoding=used_encoding,
            length=len(stripped),
        )
        return [
            {
                "content": stripped,
                "metadata": {"source": filename},
            }
        ]
