"""OpenAI-compatible chat client that returns validated TestPlan objects."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from drivetest_agent.domain.models import TestPlan
from drivetest_agent.llm.exceptions import LLMFormatError, LLMResponseError, LLMServiceError
from drivetest_agent.llm.protocol import LLMGeneration

_FORMAT_RETRY_HINT = (
    "Your previous response was not valid JSON matching the TestPlan schema. "
    "Return only a JSON object with test_cases and pytest_code."
)


class OpenAICompatibleClient:
    """Calls an OpenAI-compatible chat completions API and parses TestPlan JSON."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        client: Any | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        timeout_env = os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS")
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(timeout_env) if timeout_env else 60.0
        )
        self._clock = clock or __import__("time").monotonic
        self._client = client
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )

    @classmethod
    def from_env(cls, *, client: Any | None = None) -> OpenAICompatibleClient:
        return cls(client=client)

    def generate(self, prompt: str) -> LLMGeneration:
        total_tokens = 0
        start = self._clock()
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        for attempt in range(2):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                raise LLMServiceError(f"LLM API request failed: {exc}") from exc

            total_tokens += _extract_tokens(completion)
            content = _extract_content(completion)
            try:
                test_plan = _parse_test_plan(content)
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
                if attempt == 0:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": _FORMAT_RETRY_HINT})
                    continue
                raise LLMFormatError(
                    "Model response was not valid TestPlan JSON after one retry."
                ) from None

            duration_seconds = max(0.0, self._clock() - start)
            return LLMGeneration(
                test_plan=test_plan,
                tokens=total_tokens,
                duration_seconds=duration_seconds,
            )


def _parse_test_plan(content: str) -> TestPlan:
    payload = json.loads(content)
    return TestPlan.model_validate(payload)


def _extract_content(completion: Any) -> str:
    try:
        choices = getattr(completion, "choices", None)
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise LLMResponseError("LLM response structure error: invalid choices.") from exc
    if not isinstance(choices, (list, tuple)) or not choices:
        raise LLMResponseError("LLM response structure error: choices must be non-empty.")

    try:
        message = getattr(choices[0], "message", None)
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise LLMResponseError("LLM response structure error: invalid message.") from exc
    if message is None:
        raise LLMResponseError("LLM response structure error: choice is missing message.")

    try:
        content = getattr(message, "content", None)
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise LLMResponseError("LLM response structure error: invalid content.") from exc
    if not isinstance(content, str) or not content:
        raise LLMResponseError("LLM response structure error: message is missing content.")
    return content


def _extract_tokens(completion: Any) -> int:
    try:
        usage = getattr(completion, "usage", None)
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise LLMResponseError("LLM response structure error: invalid usage.") from exc
    if usage is None:
        return 0

    try:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is not None or completion_tokens is not None:
            prompt_count = _token_count(prompt_tokens, "prompt_tokens")
            completion_count = _token_count(completion_tokens, "completion_tokens")
            return prompt_count + completion_count
        if total_tokens is not None:
            return _token_count(total_tokens, "total_tokens")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise LLMResponseError("LLM response structure error: invalid usage token counts.") from exc

    return 0


def _token_count(value: Any, field_name: str) -> int:
    count = int(value or 0)
    if count < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return count
