"""Video document loader with keyframe and audio extraction."""
from __future__ import annotations
import structlog

logger = structlog.get_logger(__name__)


class VideoLoader:
    """Load video files, extracting keyframe descriptions and audio transcript."""

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

    async def load(self, file_bytes: bytes, filename: str) -> list[dict]:
        """Extract content from a video file.

        Attempts to extract keyframes via OpenCV and audio transcript via
        speech_recognition. Falls back to metadata-only if dependencies are
        unavailable.
        """
        import io
        import tempfile
        import os

        metadata = {"source": filename, "modality": "video"}
        segments: list[dict] = []

        # Write to temp file for processing
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # Try video metadata extraction with OpenCV
            keyframe_texts = []
            try:
                import cv2
                cap = cv2.VideoCapture(tmp_path)
                if cap.isOpened():
                    metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
                    metadata["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = metadata["frame_count"] / max(metadata["fps"], 1)
                    metadata["duration_seconds"] = round(duration, 2)

                    # Extract keyframes at regular intervals (max 10)
                    num_keyframes = min(10, max(1, int(duration / 10)))
                    interval = int(metadata["frame_count"] / num_keyframes)

                    for i in range(num_keyframes):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
                        ret, frame = cap.read()
                        if ret:
                            timestamp = round((i * interval) / max(metadata["fps"], 1), 2)
                            keyframe_texts.append(f"[Keyframe at {timestamp}s]")
                    cap.release()
                    logger.info("video_metadata_extracted", filename=filename, duration=metadata.get("duration_seconds"))
            except ImportError:
                logger.info("opencv_not_available", filename=filename)
            except Exception as exc:
                logger.warning("video_extraction_failed", filename=filename, error=str(exc))

            # Try audio transcript extraction
            transcript = ""
            try:
                import subprocess
                audio_path = tmp_path + ".wav"
                subprocess.run(
                    ["ffmpeg", "-i", tmp_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
                    capture_output=True, timeout=60
                )
                if os.path.exists(audio_path):
                    import speech_recognition as sr
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(audio_path) as source:
                        audio = recognizer.record(source, duration=300)
                    transcript = recognizer.recognize_google(audio)
                    os.unlink(audio_path)
                    logger.info("audio_transcript_extracted", filename=filename, length=len(transcript))
            except (ImportError, FileNotFoundError):
                logger.info("audio_extraction_dependencies_unavailable", filename=filename)
            except Exception as exc:
                logger.warning("audio_extraction_failed", filename=filename, error=str(exc))

            # Build content from available extractions
            parts = []
            if keyframe_texts:
                parts.append(f"Video keyframes: {', '.join(keyframe_texts)}")
            if transcript:
                parts.append(f"Audio transcript: {transcript}")
                metadata["has_transcript"] = True

            if parts:
                content = f"[Video: {filename}] " + " | ".join(parts)
            else:
                content = (
                    f"[Video: {filename}] Duration: {metadata.get('duration_seconds', '?')}s, "
                    f"Resolution: {metadata.get('width', '?')}x{metadata.get('height', '?')}, "
                    f"FPS: {metadata.get('fps', '?')}"
                )
                metadata["has_transcript"] = False

            segments.append({"content": content, "metadata": metadata})

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not segments:
            segments.append({
                "content": f"[Video: {filename}] No content could be extracted.",
                "metadata": metadata,
            })

        return segments
