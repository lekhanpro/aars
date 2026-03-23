"""PDF document loader using PyMuPDF."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class PDFLoader:
    """Extract text content from PDF files page by page using PyMuPDF (fitz)."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    async def load(self, file_bytes: bytes, filename: str) -> list[dict]:
        """Load a PDF from raw bytes and return per-page content with metadata.

        Args:
            file_bytes: Raw bytes of the PDF file.
            filename: Original filename for metadata attribution.

        Returns:
            List of dicts, each with ``content`` (str) and ``metadata`` (dict)
            keys.  Metadata includes ``source`` and ``page`` (1-indexed).

        Raises:
            ValueError: If the PDF contains no extractable text.
            RuntimeError: If PyMuPDF cannot open or parse the file.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            logger.error("pymupdf_not_installed")
            raise RuntimeError(
                "PyMuPDF is required for PDF loading. "
                "Install it with: pip install pymupdf"
            ) from exc

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            logger.error("pdf_open_failed", filename=filename, error=str(exc))
            raise RuntimeError(
                f"Failed to open PDF '{filename}': {exc}"
            ) from exc

        pages: list[dict] = []
        try:
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                except Exception as exc:
                    logger.warning(
                        "pdf_page_extraction_failed",
                        filename=filename,
                        page=page_num + 1,
                        error=str(exc),
                    )
                    continue

                # Skip pages that are blank or contain only whitespace
                stripped = text.strip()
                if not stripped:
                    logger.debug(
                        "pdf_page_empty",
                        filename=filename,
                        page=page_num + 1,
                    )
                    continue

                pages.append(
                    {
                        "content": stripped,
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1,
                        },
                    }
                )
        finally:
            doc.close()

        if not pages:
            raise ValueError(
                f"PDF '{filename}' contains no extractable text across "
                f"{len(doc)} page(s)."
            )

        logger.info(
            "pdf_loaded",
            filename=filename,
            pages_extracted=len(pages),
        )
        return pages
