"""Anthropic SDK wrapper with structured output support."""

from __future__ import annotations

import json
from typing import Any, TypeVar

import anthropic
import structlog
from pydantic import BaseModel

from config.settings import LLMSettings

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Wrapper around Anthropic SDK for structured and free-form generation."""

    def __init__(self, api_key: str, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings()
        self.client = anthropic.Anthropic(api_key=api_key)
        self.total_tokens = 0
        self.total_calls = 0

    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a free-form text response."""
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        self.total_calls += 1
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens

        logger.debug(
            "llm_generate",
            model=self.settings.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.content[0].text

    async def structured_output(
        self,
        prompt: str,
        output_model: type[T],
        system: str = "",
        max_tokens: int | None = None,
    ) -> T:
        """Generate a structured response parsed into a Pydantic model."""
        schema_hint = json.dumps(output_model.model_json_schema(), indent=2)
        augmented_prompt = (
            f"{prompt}\n\n"
            f"Respond with ONLY a valid JSON object matching this schema:\n"
            f"```json\n{schema_hint}\n```\n"
            f"Do not include any text before or after the JSON."
        )

        raw = await self.generate(augmented_prompt, system=system, max_tokens=max_tokens)

        # Extract JSON from response (handle markdown code blocks)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (``` markers)
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        parsed = json.loads(text)
        return output_model.model_validate(parsed)

    def reset_counters(self) -> None:
        self.total_tokens = 0
        self.total_calls = 0
