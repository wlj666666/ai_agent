"""Restricted pytest runner for generated test code."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from drivetest_agent.domain.models import TestExecutionResult

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
TMP_TESTS_DIR = _PROJECT_ROOT / ".tmp_tests"

_ALLOWED_TOP_LEVEL_MODULES = frozenset({"pytest", "math", "drivetest_agent"})
_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {"os", "subprocess", "pathlib", "socket", "sys", "shutil", "builtins"}
)
_FORBIDDEN_CALLS = frozenset(
    {"eval", "exec", "compile", "open", "__import__", "getattr", "vars", "globals", "locals"}
)
_FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "__builtins__",
        "__globals__",
        "__dict__",
        "__class__",
        "__bases__",
        "__subclasses__",
    }
)


class _ImportValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.error: str | None = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _FORBIDDEN_IMPORT_ROOTS or root not in _ALLOWED_TOP_LEVEL_MODULES:
                self.error = f"Disallowed import: {alias.name}"
                return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            self.error = "Disallowed relative import"
            return
        root = node.module.split(".")[0]
        if root in _FORBIDDEN_IMPORT_ROOTS or root not in _ALLOWED_TOP_LEVEL_MODULES:
            self.error = f"Disallowed import: {node.module}"
            return
        self.generic_visit(node)


class _DangerousCallValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.error: str | None = None

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in _FORBIDDEN_CALLS:
            self.error = f"Disallowed call: {name}"
            return
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _FORBIDDEN_ATTRIBUTES:
            self.error = f"Disallowed attribute access: {node.attr}"
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_ATTRIBUTES:
            self.error = f"Disallowed name access: {node.id}"
            return
        self.generic_visit(node)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _validate_test_code(test_code: str) -> str | None:
    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        return f"Invalid Python syntax: {exc.msg}"

    import_validator = _ImportValidator()
    import_validator.visit(tree)
    if import_validator.error:
        return import_validator.error

    call_validator = _DangerousCallValidator()
    call_validator.visit(tree)
    if call_validator.error:
        return call_validator.error

    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "... [truncated; showing output tail] ...\n"
    if max_chars <= len(marker):
        return marker[:max_chars]
    return marker + text[-(max_chars - len(marker)) :]


def _parse_pytest_summary(output: str) -> tuple[int, int]:
    passed = 0
    failed = 0
    match = re.search(r"(\d+)\s+passed", output)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+)\s+failed", output)
    if match:
        failed = int(match.group(1))
    return passed, failed


def run_pytest(
    test_code: str,
    *,
    timeout_seconds: float = 5.0,
    max_output_chars: int = 4000,
) -> TestExecutionResult:
    """Validate, execute, and summarize generated pytest code in a temp directory."""
    validation_error = _validate_test_code(test_code)
    if validation_error:
        return TestExecutionResult(
            exit_code=1,
            passed=0,
            failed=0,
            duration_seconds=0.0,
            error_summary=validation_error,
            timed_out=False,
        )

    project_root = _PROJECT_ROOT
    temp_root = TMP_TESTS_DIR
    temp_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"run_{uuid.uuid4().hex}_", dir=temp_root))
    test_file = run_dir / "test_generated.py"

    start = time.monotonic()
    try:
        test_file.write_text(test_code, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(test_file)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        duration_seconds = time.monotonic() - start
        combined_output = completed.stdout + "\n" + completed.stderr
        passed, failed = _parse_pytest_summary(combined_output)
        error_summary = None
        if completed.returncode != 0:
            error_summary = _truncate(combined_output.strip() or "pytest failed", max_output_chars)

        return TestExecutionResult(
            exit_code=completed.returncode,
            passed=passed,
            failed=failed,
            duration_seconds=duration_seconds,
            error_summary=error_summary,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration_seconds = time.monotonic() - start
        output = ""
        if exc.stdout:
            output += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8")
        if exc.stderr:
            output += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8")
        summary = _truncate(
            (output.strip() + "\nExecution timed out.").strip(),
            max_output_chars,
        )
        return TestExecutionResult(
            exit_code=-1,
            passed=0,
            failed=0,
            duration_seconds=duration_seconds,
            error_summary=summary,
            timed_out=True,
        )
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
