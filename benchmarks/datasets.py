"""Load and prepare benchmark datasets from HuggingFace.

Each loader normalises the dataset-specific schema into a consistent
dictionary format suitable for evaluation, capping the number of samples
to keep benchmarking tractable.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Unified interface for loading standard QA / IR benchmark datasets."""

    # ------------------------------------------------------------------
    # HotpotQA
    # ------------------------------------------------------------------

    @staticmethod
    def load_hotpotqa(
        split: str = "validation",
        max_samples: int = 500,
    ) -> list[dict[str, Any]]:
        """Load HotpotQA multi-hop question-answering dataset.

        Returns a list of dicts, each containing:
            question         – the natural-language question
            answer           – gold answer string
            supporting_facts – dict with ``title`` and ``sent_id`` lists
            type             – question type (bridge / comparison)
            level            – difficulty level (easy / medium / hard)
        """
        logger.info("Loading HotpotQA split=%s max_samples=%d", split, max_samples)
        try:
            ds = hf_load_dataset("hotpot_qa", "fullwiki", split=split)
        except Exception:
            # Some mirrors host it under a slightly different config.
            ds = hf_load_dataset("hotpot_qa", "distractor", split=split)

        samples: list[dict[str, Any]] = []
        for row in ds:
            if len(samples) >= max_samples:
                break
            samples.append({
                "question": row["question"],
                "answer": row["answer"],
                "supporting_facts": row.get("supporting_facts", {}),
                "type": row.get("type", ""),
                "level": row.get("level", ""),
                "context": row.get("context", {}),
            })

        logger.info("HotpotQA loaded: %d samples", len(samples))
        return samples

    # ------------------------------------------------------------------
    # Natural Questions
    # ------------------------------------------------------------------

    @staticmethod
    def load_natural_questions(
        split: str = "validation",
        max_samples: int = 500,
    ) -> list[dict[str, Any]]:
        """Load Google Natural Questions dataset.

        Returns a list of dicts, each containing:
            question  – the natural-language question
            answer    – list of short-answer strings (may be empty)
            document  – the associated Wikipedia document text
        """
        logger.info("Loading NQ split=%s max_samples=%d", split, max_samples)
        ds = hf_load_dataset("google-research-datasets/natural_questions", split=split)

        samples: list[dict[str, Any]] = []
        for row in ds:
            if len(samples) >= max_samples:
                break

            # Extract short answers from the annotations.
            annotations = row.get("annotations", {})
            short_answers_raw = annotations.get("short_answers", [])
            answers: list[str] = []
            document_text: str = row.get("document", {}).get("tokens", {}).get("token", "")
            if isinstance(document_text, list):
                document_text = " ".join(document_text)

            # Each annotation entry may have multiple short answer spans.
            for sa_list in short_answers_raw:
                if isinstance(sa_list, dict):
                    start_tokens = sa_list.get("start_token", [])
                    end_tokens = sa_list.get("end_token", [])
                    if isinstance(start_tokens, int):
                        start_tokens = [start_tokens]
                        end_tokens = [end_tokens]
                    for s, e in zip(start_tokens, end_tokens):
                        tokens = row.get("document", {}).get("tokens", {}).get("token", [])
                        if isinstance(tokens, list) and s < len(tokens):
                            answers.append(" ".join(tokens[s:e]))
                elif isinstance(sa_list, list):
                    for sa in sa_list:
                        if isinstance(sa, dict):
                            text = sa.get("text", "")
                            if text:
                                answers.append(text)

            # Fallback: use yes/no answer if no short answer was extracted.
            if not answers:
                yes_no = annotations.get("yes_no_answer", [])
                if isinstance(yes_no, list):
                    for yn in yes_no:
                        if yn and yn not in ("NONE", -1):
                            answers.append(str(yn))
                elif yes_no and yes_no not in ("NONE", -1):
                    answers.append(str(yes_no))

            samples.append({
                "question": row.get("question", {}).get("text", ""),
                "answer": answers,
                "document": document_text,
            })

        logger.info("NQ loaded: %d samples", len(samples))
        return samples

    # ------------------------------------------------------------------
    # TriviaQA
    # ------------------------------------------------------------------

    @staticmethod
    def load_triviaqa(
        split: str = "validation",
        max_samples: int = 500,
    ) -> list[dict[str, Any]]:
        """Load TriviaQA dataset.

        Returns a list of dicts, each containing:
            question     – the natural-language question
            answer       – dict with ``value`` (canonical) and ``aliases``
            entity_pages – associated entity page information
        """
        logger.info("Loading TriviaQA split=%s max_samples=%d", split, max_samples)
        ds = hf_load_dataset("trivia_qa", "rc", split=split)

        samples: list[dict[str, Any]] = []
        for row in ds:
            if len(samples) >= max_samples:
                break
            answer_obj = row.get("answer", {})
            samples.append({
                "question": row["question"],
                "answer": {
                    "value": answer_obj.get("value", ""),
                    "aliases": answer_obj.get("aliases", []),
                },
                "entity_pages": row.get("entity_pages", {}),
            })

        logger.info("TriviaQA loaded: %d samples", len(samples))
        return samples

    # ------------------------------------------------------------------
    # MS MARCO
    # ------------------------------------------------------------------

    @staticmethod
    def load_msmarco(
        split: str = "validation",
        max_samples: int = 500,
    ) -> list[dict[str, Any]]:
        """Load MS MARCO passage-ranking dataset.

        Returns a list of dicts, each containing:
            query    – the search query
            passages – dict with ``is_selected`` and ``passage_text`` lists
            answers  – list of human-written answer strings
        """
        logger.info("Loading MS MARCO split=%s max_samples=%d", split, max_samples)
        ds = hf_load_dataset("microsoft/ms_marco", "v2.1", split=split)

        samples: list[dict[str, Any]] = []
        for row in ds:
            if len(samples) >= max_samples:
                break

            answers_raw = row.get("answers", [])
            # Filter out empty or "No Answer Present." entries.
            answers = [
                a for a in answers_raw
                if a and a.strip().lower() != "no answer present."
            ]

            samples.append({
                "query": row.get("query", ""),
                "passages": row.get("passages", {}),
                "answers": answers,
                "query_type": row.get("query_type", ""),
            })

        logger.info("MS MARCO loaded: %d samples", len(samples))
        return samples

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    @staticmethod
    def load_dataset_by_name(name: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Load a benchmark dataset by its short name.

        Parameters
        ----------
        name:
            One of ``"hotpotqa"``, ``"nq"`` / ``"natural_questions"``,
            ``"triviaqa"``, or ``"msmarco"`` / ``"ms_marco"``.
        **kwargs:
            Forwarded to the underlying loader (e.g. ``split``, ``max_samples``).

        Returns
        -------
        list[dict]
            Loaded and normalised samples.

        Raises
        ------
        ValueError
            If *name* is not recognised.
        """
        registry: dict[str, Any] = {
            "hotpotqa": DatasetLoader.load_hotpotqa,
            "nq": DatasetLoader.load_natural_questions,
            "natural_questions": DatasetLoader.load_natural_questions,
            "triviaqa": DatasetLoader.load_triviaqa,
            "msmarco": DatasetLoader.load_msmarco,
            "ms_marco": DatasetLoader.load_msmarco,
        }

        loader = registry.get(name.lower())
        if loader is None:
            available = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown dataset '{name}'. Available datasets: {available}"
            )

        return loader(**kwargs)
