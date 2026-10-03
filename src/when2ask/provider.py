from __future__ import annotations

import json
import os
from typing import Any, Mapping


class OpenAIJsonGenerator:
    """Small OpenAI-compatible adapter used only by the reconstruction harness."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIJsonGenerator requires the optional llm dependencies"
            ) from exc

        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    @staticmethod
    def _parse_json(text: str) -> Mapping[str, Any]:
        payload = json.loads(text.strip())
        if not isinstance(payload, dict):
            raise ValueError("model response must be a JSON object")
        return payload

    def generate_json(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> Mapping[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON only. Do not add markdown or prose.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        return self._parse_json(text)
