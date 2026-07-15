"""Environment-variable configuration loading and validation.

Loading the project's ``.env`` file and parsing tuning variables never makes
a network call and never raises for anything other than a genuinely invalid
value, so callers can surface a safe, specific error message without ever
touching the LLM API.
"""

from __future__ import annotations

import math
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_RETRIEVAL_MIN_RELEVANCE = 0.15
_OPENAI_REQUEST_TIMEOUT_SECONDS_ENV_VAR = "OPENAI_REQUEST_TIMEOUT_SECONDS"
_RETRIEVAL_MIN_RELEVANCE_ENV_VAR = "RETRIEVAL_MIN_RELEVANCE"


class ConfigError(ValueError):
    """Raised when an environment-variable configuration value is invalid."""


def load_dotenv_if_present(env_path: Path | str) -> bool:
    """Load ``KEY=VALUE`` pairs from *env_path* into ``os.environ``.

    Existing environment variables are never overridden: real process/shell
    environment variables (e.g. those set by a deployment platform or CI)
    always take precedence over the ``.env`` file. Returns whether the file
    existed and was loaded; does nothing (and returns ``False``) if it does
    not exist.
    """
    path = Path(env_path)
    if not path.is_file():
        return False
    load_dotenv(dotenv_path=path, override=False)
    return True


def parse_retrieval_min_relevance(
    raw: str | None, *, default: float = DEFAULT_RETRIEVAL_MIN_RELEVANCE
) -> float:
    """Parse the ``RETRIEVAL_MIN_RELEVANCE`` tuning variable.

    Returns *default* when *raw* is ``None`` or blank. Raises
    :class:`ConfigError` with a clear, user-facing message when *raw* is not
    a number or falls outside the inclusive ``[0, 1]`` range expected by
    :class:`~drivetest_agent.retrieval.retriever.KnowledgeRetriever`.
    """
    if raw is None or not raw.strip():
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{_RETRIEVAL_MIN_RELEVANCE_ENV_VAR} 配置无效：{raw!r} 不是合法数字，"
            "请设置为 0 到 1 之间的小数（例如 0.15）。"
        ) from exc

    if not (0.0 <= value <= 1.0):
        raise ConfigError(
            f"{_RETRIEVAL_MIN_RELEVANCE_ENV_VAR} 配置超出范围：{value}，"
            "必须在 0 到 1 之间（包含 0 和 1）。"
        )
    return value


def parse_openai_request_timeout_seconds(
    raw: str | float | None,
    *,
    default: float = DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS,
) -> float:
    """Parse ``OPENAI_REQUEST_TIMEOUT_SECONDS`` as a finite positive number."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default

    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{_OPENAI_REQUEST_TIMEOUT_SECONDS_ENV_VAR} 配置无效：{raw!r} 不是合法数字，"
            "请设置为有限正数（例如 60）。"
        ) from exc

    if not math.isfinite(value) or value <= 0.0:
        raise ConfigError(
            f"{_OPENAI_REQUEST_TIMEOUT_SECONDS_ENV_VAR} 配置无效：{raw!r}，"
            "必须是有限正数（大于 0）。"
        )
    return value
