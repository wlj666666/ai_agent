"""Top-level Streamlit page: session-state wiring and layout.

Importing this module must never perform an LLM API call. Real service
objects (retriever, LLM client, agent) are built lazily inside
``_handle_run_click``/``_cached_agent``, which only run once the user
interacts with the page.
"""

from __future__ import annotations

import logging

import streamlit as st

from drivetest_agent.config import ConfigError
from drivetest_agent.domain.models import Requirement
from drivetest_agent.ui.examples import ExampleCase, load_example_cases
from drivetest_agent.ui.render import (
    render_error,
    render_example_picker,
    render_header,
    render_report,
    render_requirement_inputs,
)
from drivetest_agent.ui.service import (
    AgentLike,
    build_agent,
    missing_llm_config_message,
    run_requirement,
)
from drivetest_agent.ui.view_model import build_report_view_model

logger = logging.getLogger(__name__)

_SESSION_DEFAULTS: dict[str, object] = {
    "requirement_text": "",
    "component_description": "",
    "last_report": None,
    "last_run_error": None,
}

_EXAMPLES_LOAD_ERROR_MESSAGE = (
    "加载固定演示案例失败，请检查 examples/cases.json 是否存在且格式正确。"
)
_AGENT_INIT_ERROR_MESSAGE = "初始化检索或模型组件失败，请检查配置后重试。"
_EMPTY_REQUIREMENT_MESSAGE = "请先填写需求文本，再点击“运行”。"


@st.cache_resource(show_spinner=False)
def _cached_agent() -> AgentLike:
    return build_agent()


def _init_session_state() -> None:
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _load_examples_safely() -> list[ExampleCase]:
    try:
        return load_example_cases()
    except Exception:
        logger.exception("Failed to load example cases.")
        return []


def _handle_run_click() -> None:
    text = str(st.session_state.get("requirement_text", "")).strip()
    if not text:
        st.session_state["last_run_error"] = _EMPTY_REQUIREMENT_MESSAGE
        st.session_state["last_report"] = None
        return

    config_error = missing_llm_config_message()
    if config_error is not None:
        st.session_state["last_run_error"] = config_error
        st.session_state["last_report"] = None
        return

    component_description = str(st.session_state.get("component_description", "")).strip() or None
    requirement = Requirement(text=text, component_description=component_description)

    try:
        agent = _cached_agent()
    except ConfigError as exc:
        logger.exception("Invalid configuration while building the agent stack.")
        st.session_state["last_run_error"] = str(exc)
        st.session_state["last_report"] = None
        return
    except Exception:
        logger.exception("Failed to build the agent stack.")
        st.session_state["last_run_error"] = _AGENT_INIT_ERROR_MESSAGE
        st.session_state["last_report"] = None
        return

    outcome = run_requirement(agent, requirement)
    if outcome.error_message is not None:
        st.session_state["last_run_error"] = outcome.error_message
        st.session_state["last_report"] = None
    else:
        st.session_state["last_run_error"] = None
        st.session_state["last_report"] = outcome.report


def render_page() -> None:
    """Render the single-page Streamlit demo. Call only from ``streamlit run``."""
    st.set_page_config(page_title="DriveTest Agent", layout="wide")
    _init_session_state()
    render_header()

    examples = _load_examples_safely()
    if not examples:
        render_error(_EXAMPLES_LOAD_ERROR_MESSAGE)
    else:
        render_example_picker(examples)

    render_requirement_inputs()
    st.button("运行", type="primary", on_click=_handle_run_click, key="run_button")

    error_message = st.session_state.get("last_run_error")
    report = st.session_state.get("last_report")
    if error_message:
        render_error(str(error_message))
    elif report is not None:
        view = build_report_view_model(report)
        render_report(view)
    else:
        st.info("尚未运行。选择案例或填写需求后点击“运行”。")
