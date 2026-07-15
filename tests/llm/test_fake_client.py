"""Tests for FakeLLMClient."""

from __future__ import annotations

import pytest

from drivetest_agent.domain.models import TestPlan
from drivetest_agent.llm.fake_client import FakeLLMClient
from drivetest_agent.llm.protocol import LLMGeneration


class TestFakeLLMClient:
    def test_returns_queued_generations_in_order(self, sample_test_plan: TestPlan) -> None:
        plan_a = sample_test_plan
        plan_b = sample_test_plan.model_copy(
            update={"pytest_code": "def test_other():\n    assert True\n"}
        )
        client = FakeLLMClient()
        client.enqueue(LLMGeneration(test_plan=plan_a, tokens=100, duration_seconds=0.5))
        client.enqueue(LLMGeneration(test_plan=plan_b, tokens=50, duration_seconds=0.2))

        first = client.generate("prompt-a")
        second = client.generate("prompt-b")

        assert first.test_plan.pytest_code == plan_a.pytest_code
        assert second.test_plan.pytest_code == plan_b.pytest_code
        assert first.tokens == 100
        assert second.tokens == 50

    def test_records_prompts_in_order(self, sample_test_plan: TestPlan) -> None:
        client = FakeLLMClient()
        generation = LLMGeneration(test_plan=sample_test_plan, tokens=10, duration_seconds=0.1)
        client.enqueue(generation)
        client.enqueue(generation)

        client.generate("first prompt")
        client.generate("second prompt")

        assert client.prompts == ["first prompt", "second prompt"]

    def test_raises_queued_exception(self) -> None:
        client = FakeLLMClient()
        client.enqueue(RuntimeError("simulated failure"))

        with pytest.raises(RuntimeError, match="simulated failure"):
            client.generate("any prompt")

    def test_raises_when_queue_empty(self) -> None:
        client = FakeLLMClient()

        with pytest.raises(RuntimeError, match="no queued"):
            client.generate("unexpected prompt")
