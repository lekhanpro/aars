"""Image document loader with optional OCR."""
from __future__ import annotations
import structlog

logger = structlog.get_logger(__name__)


class ImageLoader:
    """Load image files, extracting text via OCR when available."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

    async def load(self, file_bytes: bytes, filename: str) -> list[dict]:
        """Extract text content from an image file.

        Uses pytesseract OCR if available, otherwise generates a metadata-only
        document describing the image properties.
        """
        import io
        metadata = {"source": filename, "modality": "image"}

        # Try to get image dimensions
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(file_bytes))
            metadata["width"] = img.width
            metadata["height"] = img.height
            metadata["format"] = img.format or "unknown"
            metadata["mode"] = img.mode
        except ImportError:
            logger.warning("pillow_not_available", filename=filename)
            metadata["format"] = "unknown"
        except Exception as exc:
            logger.warning("image_open_failed", filename=filename, error=str(exc))
            metadata["format"] = "unknown"

        # Try OCR extraction
        text = ""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(img).strip()
            if text:
                metadata["ocr_extracted"] = True
                logger.info("ocr_text_extracted", filename=filename, length=len(text))
        except ImportError:
            logger.info("pytesseract_not_available", filename=filename)
        except Exception as exc:
            logger.warning("ocr_failed", filename=filename, error=str(exc))

        if not text:
            text = f"[Image: {filename}] Dimensions: {metadata.get('width', '?')}x{metadata.get('height', '?')}, Format: {metadata.get('format', '?')}"
            metadata["ocr_extracted"] = False

        return [{"content": text, "metadata": metadata}]
