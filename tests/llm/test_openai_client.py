"""Tests for OpenAICompatibleClient without real network access."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from drivetest_agent.config import ConfigError
from drivetest_agent.domain.models import TestPlan
from drivetest_agent.llm.exceptions import LLMFormatError, LLMResponseError, LLMServiceError
from drivetest_agent.llm.openai_client import OpenAICompatibleClient
from drivetest_agent.llm.protocol import LLMGeneration


class FakeClock:
    def __init__(self) -> None:
        self._now = 0.0
        self._on_tick: Callable[[], None] | None = None

    def monotonic(self) -> float:
        if self._on_tick is not None:
            self._on_tick()
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class FakeChatCompletions:
    def __init__(
        self,
        responses: list[Any],
        *,
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._on_call = on_call
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        if self._on_call is not None:
            self._on_call()
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("no fake responses left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeOpenAIClient:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


def _completion(content: str, *, prompt_tokens: int = 10, completion_tokens: int = 20) -> Any:
    return type(
        "Completion",
        (),
        {
            "choices": [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()],
            "usage": type(
                "Usage",
                (),
                {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            )(),
        },
    )()


def _response(**attributes: Any) -> Any:
    return type("Response", (), attributes)()


def _client(
    responses: list[Any],
    *,
    clock: FakeClock | None = None,
    on_call: Callable[[], None] | None = None,
) -> OpenAICompatibleClient:
    fake_clock = clock or FakeClock()
    completions = FakeChatCompletions(responses, on_call=on_call)
    return OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        timeout_seconds=30.0,
        client=FakeOpenAIClient(completions),
        clock=fake_clock.monotonic,
    )


class TestOpenAICompatibleClient:
    def test_parses_valid_json_into_test_plan(self, sample_test_plan_json: str) -> None:
        client = _client([_completion(sample_test_plan_json)])

        result = client.generate("build tests")

        assert isinstance(result, LLMGeneration)
        assert isinstance(result.test_plan, TestPlan)
        assert result.test_plan.test_cases[0].name == "test_aeb_triggers_at_threshold"
        assert result.tokens == 30
        assert result.duration_seconds >= 0.0

    def test_retries_once_on_invalid_format_then_succeeds(
        self, sample_test_plan_json: str
    ) -> None:
        clock = FakeClock()
        completions = FakeChatCompletions(
            [
                _completion("not valid json"),
                _completion(sample_test_plan_json, prompt_tokens=5, completion_tokens=7),
            ],
            on_call=lambda: clock.advance(0.5),
        )
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
            timeout_seconds=30.0,
            client=FakeOpenAIClient(completions),
            clock=clock.monotonic,
        )

        result = client.generate("build tests")

        assert len(completions.calls) == 2
        assert result.tokens == 42
        assert result.duration_seconds == pytest.approx(1.0)

    def test_raises_after_two_invalid_responses(self) -> None:
        client = _client([_completion("{bad"), _completion("still bad")])

        with pytest.raises(LLMFormatError, match="valid TestPlan"):
            client.generate("build tests")

    def test_raises_llm_service_error_on_api_failure(self) -> None:
        client = _client([ConnectionError("network down")])

        with pytest.raises(LLMServiceError, match="network"):
            client.generate("build tests")

    def test_accumulates_tokens_and_duration_across_retry(
        self, sample_test_plan_json: str
    ) -> None:
        clock = FakeClock()
        completions = FakeChatCompletions(
            [
                _completion("{}", prompt_tokens=100, completion_tokens=50),
                _completion(
                    sample_test_plan_json,
                    prompt_tokens=200,
                    completion_tokens=80,
                ),
            ],
            on_call=lambda: clock.advance(1.75),
        )
        client = OpenAICompatibleClient(
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
            timeout_seconds=30.0,
            client=FakeOpenAIClient(completions),
            clock=clock.monotonic,
        )

        result = client.generate("build tests")

        assert result.tokens == 430
        assert result.duration_seconds == pytest.approx(3.5)

    def test_reads_configuration_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
        monkeypatch.setenv("OPENAI_MODEL", "env-model")
        monkeypatch.setenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "45")

        fake_client = FakeOpenAIClient(FakeChatCompletions([_completion("{}")]))
        client = OpenAICompatibleClient.from_env(client=fake_client)

        assert client.api_key == "env-key"
        assert client.base_url == "https://env.example/v1"
        assert client.model == "env-model"
        assert client.timeout_seconds == 45.0

    def test_direct_construction_rejects_invalid_explicit_timeout(self) -> None:
        fake_client = FakeOpenAIClient(FakeChatCompletions([]))

        with pytest.raises(ConfigError, match="OPENAI_REQUEST_TIMEOUT_SECONDS"):
            OpenAICompatibleClient(timeout_seconds=0, client=fake_client)

    def test_from_env_rejects_non_finite_timeout_with_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "nan")
        fake_client = FakeOpenAIClient(FakeChatCompletions([]))

        with pytest.raises(ConfigError, match="OPENAI_REQUEST_TIMEOUT_SECONDS"):
            OpenAICompatibleClient.from_env(client=fake_client)

    def test_rejects_schema_invalid_json(self) -> None:
        invalid = json.dumps({"test_cases": [], "pytest_code": "x"})
        client = _client([_completion(invalid), _completion(invalid)])

        with pytest.raises(LLMFormatError):
            client.generate("build tests")

    @pytest.mark.parametrize(
        ("response", "expected_detail"),
        [
            (_response(choices=[], usage=None), "choices"),
            (_response(choices=[_response()], usage=None), "message"),
            (
                _response(choices=[_response(message=_response())], usage=None),
                "content",
            ),
            (
                _response(
                    choices=[_response(message=_response(content=None))],
                    usage=None,
                ),
                "content",
            ),
        ],
    )
    def test_malformed_response_structure_raises_classified_error(
        self, response: Any, expected_detail: str
    ) -> None:
        client = _client([response])

        with pytest.raises(LLMResponseError, match=expected_detail):
            client.generate("build tests")

    def test_malformed_usage_raises_classified_error(
        self, sample_test_plan_json: str
    ) -> None:
        response = _completion(sample_test_plan_json)
        response.usage = _response(prompt_tokens="not-a-number", completion_tokens=2)
        client = _client([response])

        with pytest.raises(LLMResponseError, match="usage"):
            client.generate("build tests")

    def test_negative_usage_raises_classified_error(
        self, sample_test_plan_json: str
    ) -> None:
        response = _completion(sample_test_plan_json)
        response.usage = _response(prompt_tokens=-1, completion_tokens=2)
        client = _client([response])

        with pytest.raises(LLMResponseError, match="usage"):
            client.generate("build tests")

    def test_total_tokens_is_used_when_split_counts_are_absent(
        self, sample_test_plan_json: str
    ) -> None:
        response = _completion(sample_test_plan_json)
        response.usage = _response(total_tokens=47)
        client = _client([response])

        result = client.generate("build tests")

        assert result.tokens == 47

    def test_response_structure_error_is_not_retried(self) -> None:
        completions = FakeChatCompletions(
            [
                _response(choices=[], usage=None),
                _completion("unused valid response"),
            ]
        )
        client = OpenAICompatibleClient(
            api_key="test-key",
            model="test-model",
            client=FakeOpenAIClient(completions),
        )

        with pytest.raises(LLMResponseError):
            client.generate("build tests")

        assert len(completions.calls) == 1
