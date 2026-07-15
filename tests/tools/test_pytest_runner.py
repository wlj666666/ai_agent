"""Tests for restricted pytest runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from drivetest_agent.tools import pytest_runner
from drivetest_agent.tools.pytest_runner import TMP_TESTS_DIR, _truncate, run_pytest

PASSING_TEST = """
from drivetest_agent.domain.aeb import should_trigger_aeb

def test_aeb_passes():
    assert should_trigger_aeb(ttc=1.0, relative_speed=2.0, sensor_valid=True)
"""

FAILING_TEST = """
from drivetest_agent.domain.aeb import should_trigger_aeb

def test_aeb_fails():
    assert should_trigger_aeb(ttc=5.0, relative_speed=2.0, sensor_valid=True)
"""

SLOW_TEST = """
def test_slow():
    total = 0
    while total < 10**9:
        total += 1
    assert True
"""


class TestRunPytest:
    def test_passing_test_returns_success_counts(self) -> None:
        result = run_pytest(PASSING_TEST)

        assert result.exit_code == 0
        assert result.passed == 1
        assert result.failed == 0
        assert result.timed_out is False
        assert result.duration_seconds >= 0.0

    def test_failing_test_returns_failure_counts_and_summary(self) -> None:
        result = run_pytest(FAILING_TEST)

        assert result.exit_code != 0
        assert result.passed == 0
        assert result.failed == 1
        assert result.error_summary
        assert result.timed_out is False

    def test_timeout_terminates_and_flags_timed_out(self) -> None:
        result = run_pytest(SLOW_TEST, timeout_seconds=0.5)

        assert result.timed_out is True
        assert "timed out" in (result.error_summary or "").lower()

    def test_rejects_disallowed_import_os(self) -> None:
        code = "import os\n\ndef test_bad():\n    assert True\n"
        result = run_pytest(code)

        assert result.exit_code != 0
        assert result.passed == 0
        assert result.failed == 0
        assert result.error_summary
        assert "os" in result.error_summary.lower()

    def test_rejects_disallowed_import_subprocess(self) -> None:
        code = "import subprocess\n\ndef test_bad():\n    assert True\n"
        result = run_pytest(code)

        assert result.exit_code != 0
        assert "subprocess" in (result.error_summary or "").lower()

    def test_rejects_dangerous_eval_call(self) -> None:
        code = "def test_bad():\n    eval('1+1')\n"
        result = run_pytest(code)

        assert result.exit_code != 0
        assert "eval" in (result.error_summary or "").lower()

    def test_rejects_dangerous_open_call(self) -> None:
        code = "def test_bad():\n    open('x.txt')\n"
        result = run_pytest(code)

        assert result.exit_code != 0
        assert "open" in (result.error_summary or "").lower()

    def test_truncates_long_error_summary(self) -> None:
        long_assert = "assert " + "x" * 10_000
        code = f"def test_long():\n    {long_assert}\n"
        result = run_pytest(code, max_output_chars=500)

        assert result.error_summary is not None
        assert len(result.error_summary) <= 500

    def test_truncation_preserves_pytest_output_tail_and_marks_truncation(self) -> None:
        pytest_output = "HEAD_MARKER\n" + ("middle\n" * 500) + "TAIL_MARKER\n1 failed"

        summary = _truncate(pytest_output, 200)

        assert len(summary) <= 200
        assert "HEAD_MARKER" not in summary
        assert "TAIL_MARKER" in summary
        assert "1 failed" in summary
        assert "[truncated" in summary.lower()

    @pytest.mark.parametrize("name", ["getattr", "vars", "globals", "locals"])
    def test_rejects_indirect_builtin_escape_calls(self, name: str) -> None:
        code = f"def test_bad():\n    {name}(__builtins__, 'eval')\n"

        result = run_pytest(code)

        assert result.exit_code != 0
        assert result.failed == 0
        assert name in (result.error_summary or "").lower()

    def test_rejects_getattr_eval_escape(self) -> None:
        code = "def test_bad():\n    getattr(__builtins__, 'eval')('1 + 1')\n"

        result = run_pytest(code)

        assert result.exit_code != 0
        assert result.failed == 0
        assert "getattr" in (result.error_summary or "").lower()

    def test_rejects_dunder_dict_subscript_escape(self) -> None:
        code = "def test_bad():\n    assert object.__dict__['__class__']\n"

        result = run_pytest(code)

        assert result.exit_code != 0
        assert result.failed == 0
        assert "__dict__" in (result.error_summary or "").lower()

    @pytest.mark.parametrize(
        "attribute",
        [
            "__builtins__",
            "__globals__",
            "__dict__",
            "__class__",
            "__bases__",
            "__subclasses__",
        ],
    )
    def test_rejects_dangerous_dunder_attribute_access(self, attribute: str) -> None:
        code = f"def test_bad():\n    assert object.{attribute}\n"

        result = run_pytest(code)

        assert result.exit_code != 0
        assert result.failed == 0
        assert attribute in (result.error_summary or "").lower()

    def test_cleans_up_temp_directory(self) -> None:
        before = set(Path(TMP_TESTS_DIR).glob("*")) if Path(TMP_TESTS_DIR).exists() else set()

        run_pytest(PASSING_TEST)

        after = set(Path(TMP_TESTS_DIR).glob("*")) if Path(TMP_TESTS_DIR).exists() else set()
        new_entries = after - before
        assert new_entries == set()

    def test_uses_project_temp_root_when_started_outside_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_root = Path(pytest_runner.__file__).resolve().parents[3]
        expected_temp_root = project_root / ".tmp_tests"
        before = set(expected_temp_root.glob("*"))
        external_cwd = tmp_path / "outside"
        external_cwd.mkdir()
        observed: dict[str, object] = {}
        real_run = pytest_runner.subprocess.run

        def recording_run(*args: object, **kwargs: object) -> object:
            observed["command"] = args[0]
            observed["cwd"] = kwargs["cwd"]
            return real_run(*args, **kwargs)

        monkeypatch.setattr(pytest_runner.subprocess, "run", recording_run)
        monkeypatch.chdir(external_cwd)

        result = run_pytest(PASSING_TEST)

        command = observed["command"]
        assert isinstance(command, list)
        generated_test = Path(command[-1])
        assert result.exit_code == 0
        assert Path(TMP_TESTS_DIR) == expected_temp_root
        assert observed["cwd"] == project_root
        assert generated_test.parent.parent == expected_temp_root
        assert not (external_cwd / ".tmp_tests").exists()
        assert set(expected_temp_root.glob("*")) == before

    def test_allows_whitelisted_imports(self) -> None:
        code = """
import math
import pytest
from drivetest_agent.domain.aeb import should_trigger_aeb

def test_whitelisted():
    assert math.isfinite(1.0)
    assert should_trigger_aeb(ttc=1.5, relative_speed=1.0, sensor_valid=True)
"""
        result = run_pytest(code)
        assert result.exit_code == 0
        assert result.passed == 1
