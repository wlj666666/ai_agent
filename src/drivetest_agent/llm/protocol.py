"""LLM generation contract shared by real and fake clients."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from drivetest_agent.domain.models import TestPlan


class LLMGeneration(BaseModel):
    test_plan: TestPlan
    tokens: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)


class LLMClient(Protocol):
    def generate(self, prompt: str) -> LLMGeneration: ...
