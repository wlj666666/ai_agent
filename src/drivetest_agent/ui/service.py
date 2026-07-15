"""Business wiring between the Streamlit page and the real agent stack.

Constructing an :class:`~drivetest_agent.agent.orchestrator.DriveTestAgent`
via :func:`build_agent` never performs a network call by itself (the
OpenAI SDK client only talks to the network when a request is issued).
Callers should still check :func:`missing_llm_config_message` before
running the agent so the UI can show a clear message instead of waiting on
a request that is certain to fail.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

from drivetest_agent.config import (
    load_dotenv_if_present,
    parse_openai_request_timeout_seconds,
    parse_retrieval_min_relevance,
)
from drivetest_agent.domain.models import AgentReport, Requirement
from drivetest_agent.llm.openai_client import OpenAICompatibleClient
from drivetest_agent.retrieval.retriever import KnowledgeRetriever
from drivetest_agent.tools.pytest_runner import run_pytest
from drivetest_agent.ui.paths import KNOWLEDGE_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

_ENV_PATH = PROJECT_ROOT / ".env"

_MISSING_API_KEY_MESSAGE = (
    "未检测到 OPENAI_API_KEY 环境变量，请在 .env 中配置模型密钥后重启应用。"
)
_UNEXPECTED_ERROR_MESSAGE = "运行测试生成流程时发生未预期错误，请检查日志后重试。"


class AgentLike(Protocol):
    """Structural contract for anything with a ``run(requirement)`` method."""

    def run(self, requirement: Requirement) -> AgentReport: ...


@dataclass(frozen=True)
class AgentRunOutcome:
    """Result of attempting to run the agent for one requirement.

    Exactly one of ``report`` / ``error_message`` is populated. ``report``
    being set does not guarantee ``final_status == "success"``: the agent's
    own error/failed/insufficient_info reports are still reports.
    """

    report: AgentReport | None
    error_message: str | None


def missing_llm_config_message() -> str | None:
    """Return a user-facing error if the LLM API is not configured, else None.

    Loads the project's ``.env`` file first (without overriding any
    already-set environment variable) so a locally configured key is picked
    up the same way it would be for :func:`build_agent`.
    """
    load_dotenv_if_present(_ENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _MISSING_API_KEY_MESSAGE
    return None


def build_agent() -> AgentLike:
    """Wire up the real retriever, LLM client, and restricted pytest runner.

    Loads the project's ``.env`` file (without overriding already-set
    environment variables) and validates ``RETRIEVAL_MIN_RELEVANCE`` and
    ``OPENAI_REQUEST_TIMEOUT_SECONDS`` before constructing anything. An invalid value raises
    :class:`~drivetest_agent.config.ConfigError` before any client or
    network object is created, so a bad config value can never trigger an
    API call.
    """
    load_dotenv_if_present(_ENV_PATH)
    min_relevance = parse_retrieval_min_relevance(os.environ.get("RETRIEVAL_MIN_RELEVANCE"))
    timeout_seconds = parse_openai_request_timeout_seconds(
        os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS")
    )

    retriever = KnowledgeRetriever(KNOWLEDGE_DIR, low_confidence_threshold=min_relevance)
    llm_client = OpenAICompatibleClient(timeout_seconds=timeout_seconds)

    from drivetest_agent.agent.orchestrator import DriveTestAgent

    return DriveTestAgent(retriever=retriever, llm_client=llm_client, test_runner=run_pytest)


def run_requirement(agent: AgentLike, requirement: Requirement) -> AgentRunOutcome:
    """Run the agent for *requirement*, never raising and never mutating it.

    If the LLM API is not configured, the agent is not invoked at all and a
    safe, translated error message is returned instead.
    """
    config_error = missing_llm_config_message()
    if config_error is not None:
        return AgentRunOutcome(report=None, error_message=config_error)

    try:
        report = agent.run(requirement)
    except Exception as exc:
        logger.exception("Agent run failed (%s): %s", type(exc).__name__, exc)
        return AgentRunOutcome(report=None, error_message=_UNEXPECTED_ERROR_MESSAGE)

    return AgentRunOutcome(report=report, error_message=None)
