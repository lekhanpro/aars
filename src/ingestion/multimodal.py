"""Multimodal content detection and segregation."""
from __future__ import annotations

import os
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)


class ContentModality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


# Extension-to-modality mapping
_MODALITY_MAP: dict[str, ContentModality] = {}

# Text extensions
for ext in (".txt", ".md", ".markdown", ".rst", ".csv", ".log", ".pdf"):
    _MODALITY_MAP[ext] = ContentModality.TEXT

# Image extensions
for ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"):
    _MODALITY_MAP[ext] = ContentModality.IMAGE

# Video extensions
for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"):
    _MODALITY_MAP[ext] = ContentModality.VIDEO


def detect_modality(filename: str) -> ContentModality:
    """Detect content modality from filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    return _MODALITY_MAP.get(ext, ContentModality.UNKNOWN)


class MultimodalSegregator:
    """Segregates mixed-modality uploads into typed processing streams.

    Classifies incoming files by modality (text, image, video) and routes
    them to the appropriate loader. Maintains per-collection modality
    statistics for downstream retrieval strategy selection.
    """

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, int]] = {}

    def classify(self, filename: str) -> ContentModality:
        """Classify a file by its content modality."""
        modality = detect_modality(filename)
        logger.info("modality_detected", filename=filename, modality=modality)
        return modality

    def update_stats(self, collection: str, modality: ContentModality) -> None:
        """Track modality distribution per collection."""
        if collection not in self._stats:
            self._stats[collection] = {"text": 0, "image": 0, "video": 0, "unknown": 0}
        self._stats[collection][modality.value] = self._stats[collection].get(modality.value, 0) + 1

    def get_stats(self, collection: str) -> dict[str, int]:
        """Return modality counts for a collection."""
        return self._stats.get(collection, {"text": 0, "image": 0, "video": 0, "unknown": 0})

    def is_multimodal(self, collection: str) -> bool:
        """Check if a collection contains multiple modalities."""
        stats = self.get_stats(collection)
        active = sum(1 for count in stats.values() if count > 0)
        return active > 1
