"""Loader for the fixed demo cases used by the Streamlit page.

The cases file (``examples/cases.json``) is a plain JSON list. Each entry
must have a non-empty, unique ``id``, a non-empty ``title`` and a non-empty
``requirement``. ``component_description`` is optional.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from drivetest_agent.ui.paths import EXAMPLES_PATH


class ExampleCase(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    component_description: str | None = None

    @field_validator("id", "title", "requirement")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


def load_example_cases(path: Path | str | None = None) -> list[ExampleCase]:
    """Load and validate the fixed demo cases from *path* (default: EXAMPLES_PATH)."""
    cases_path = Path(path) if path is not None else EXAMPLES_PATH
    if not cases_path.is_file():
        raise FileNotFoundError(f"Example cases file not found: {cases_path}")

    raw_text = cases_path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    if not isinstance(payload, list):
        raise ValueError("Example cases file must contain a JSON list of case objects.")

    cases = [_parse_case(entry) for entry in payload]
    _validate_unique_ids(cases)
    return cases


def _parse_case(entry: object) -> ExampleCase:
    if not isinstance(entry, dict):
        raise ValueError(f"Each example case must be a JSON object, got: {entry!r}")
    try:
        return ExampleCase.model_validate(entry)
    except Exception as exc:
        raise ValueError(f"Invalid example case {entry!r}: {exc}") from exc


def _validate_unique_ids(cases: list[ExampleCase]) -> None:
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"Duplicate example case id: {case.id!r}")
        seen.add(case.id)
