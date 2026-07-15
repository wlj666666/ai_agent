"""Tests for .env loading and RETRIEVAL_MIN_RELEVANCE parsing.

None of these tests touch the real project ``.env`` file: every dotenv test
builds its own file under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drivetest_agent.config import (
    DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RETRIEVAL_MIN_RELEVANCE,
    ConfigError,
    load_dotenv_if_present,
    parse_openai_request_timeout_seconds,
    parse_retrieval_min_relevance,
)


class TestLoadDotenvIfPresent:
    def test_returns_false_and_does_nothing_when_file_is_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DRIVETEST_TEST_VAR", raising=False)
        missing_path = tmp_path / ".env"

        loaded = load_dotenv_if_present(missing_path)

        assert loaded is False
        assert "DRIVETEST_TEST_VAR" not in __import__("os").environ

    def test_sets_variables_from_file_when_absent_from_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DRIVETEST_TEST_VAR", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("DRIVETEST_TEST_VAR=from-file\n", encoding="utf-8")

        loaded = load_dotenv_if_present(env_file)

        import os

        assert loaded is True
        assert os.environ["DRIVETEST_TEST_VAR"] == "from-file"

    def test_does_not_override_an_existing_environment_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DRIVETEST_TEST_VAR", "from-real-environment")
        env_file = tmp_path / ".env"
        env_file.write_text("DRIVETEST_TEST_VAR=from-file\n", encoding="utf-8")

        load_dotenv_if_present(env_file)

        import os

        assert os.environ["DRIVETEST_TEST_VAR"] == "from-real-environment"

    def test_accepts_string_path_as_well_as_path_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DRIVETEST_TEST_VAR", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("DRIVETEST_TEST_VAR=from-string-path\n", encoding="utf-8")

        loaded = load_dotenv_if_present(str(env_file))

        import os

        assert loaded is True
        assert os.environ["DRIVETEST_TEST_VAR"] == "from-string-path"


class TestParseRetrievalMinRelevance:
    def test_returns_default_when_raw_is_none(self) -> None:
        assert parse_retrieval_min_relevance(None) == DEFAULT_RETRIEVAL_MIN_RELEVANCE

    def test_returns_default_when_raw_is_blank(self) -> None:
        assert parse_retrieval_min_relevance("   ") == DEFAULT_RETRIEVAL_MIN_RELEVANCE

    def test_returns_custom_default_when_provided(self) -> None:
        assert parse_retrieval_min_relevance(None, default=0.3) == 0.3

    @pytest.mark.parametrize("raw,expected", [("0", 0.0), ("1", 1.0), ("0.15", 0.15), ("0.5", 0.5)])
    def test_parses_valid_values_within_range(self, raw: str, expected: float) -> None:
        assert parse_retrieval_min_relevance(raw) == pytest.approx(expected)

    def test_boundary_zero_is_valid(self) -> None:
        assert parse_retrieval_min_relevance("0.0") == 0.0

    def test_boundary_one_is_valid(self) -> None:
        assert parse_retrieval_min_relevance("1.0") == 1.0

    def test_raises_config_error_on_non_numeric_string(self) -> None:
        with pytest.raises(ConfigError, match="RETRIEVAL_MIN_RELEVANCE"):
            parse_retrieval_min_relevance("not-a-number")

    def test_raises_config_error_when_below_zero(self) -> None:
        with pytest.raises(ConfigError, match="RETRIEVAL_MIN_RELEVANCE"):
            parse_retrieval_min_relevance("-0.01")

    def test_raises_config_error_when_above_one(self) -> None:
        with pytest.raises(ConfigError, match="RETRIEVAL_MIN_RELEVANCE"):
            parse_retrieval_min_relevance("1.01")

    def test_config_error_is_a_value_error(self) -> None:
        assert issubclass(ConfigError, ValueError)


class TestParseOpenAIRequestTimeoutSeconds:
    def test_returns_default_when_raw_is_none(self) -> None:
        assert (
            parse_openai_request_timeout_seconds(None)
            == DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS
        )

    def test_returns_default_when_raw_is_blank(self) -> None:
        assert (
            parse_openai_request_timeout_seconds("   ")
            == DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS
        )

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("0.1", 0.1), ("30", 30.0), ("60.5", 60.5)],
    )
    def test_parses_finite_positive_values(self, raw: str, expected: float) -> None:
        assert parse_openai_request_timeout_seconds(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "raw",
        ["not-a-number", "0", "-0.1", "nan", "inf", "-inf"],
    )
    def test_rejects_non_numeric_non_positive_or_non_finite_values(self, raw: str) -> None:
        with pytest.raises(ConfigError, match="OPENAI_REQUEST_TIMEOUT_SECONDS"):
            parse_openai_request_timeout_seconds(raw)
