"""In-memory LLM client for deterministic integration tests."""

from __future__ import annotations

from collections import deque

from drivetest_agent.llm.protocol import LLMGeneration


class FakeLLMClient:
    """Returns queued generations or exceptions and records prompts."""

    def __init__(self) -> None:
        self._queue: deque[LLMGeneration | Exception] = deque()
        self.prompts: list[str] = []

    def enqueue(self, item: LLMGeneration | Exception) -> None:
        self._queue.append(item)

    def generate(self, prompt: str) -> LLMGeneration:
        self.prompts.append(prompt)
        if not self._queue:
            raise RuntimeError("no queued LLM responses remaining")
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item
