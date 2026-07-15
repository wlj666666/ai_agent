"""Shared fixtures for LLM module tests."""

from __future__ import annotations

import pytest

from drivetest_agent.domain.models import TestCasePlan, TestPlan


@pytest.fixture
def sample_test_plan() -> TestPlan:
    return TestPlan(
        test_cases=[
            TestCasePlan(
                name="test_aeb_triggers_at_threshold",
                description="Verify braking at TTC exactly 1.5 seconds.",
                expected_outcome="Braking is triggered.",
            )
        ],
        pytest_code=(
            "from drivetest_agent.domain.aeb import should_trigger_aeb\n\n"
            "def test_aeb_triggers_at_threshold():\n"
            "    assert should_trigger_aeb(ttc=1.5, relative_speed=1.0, sensor_valid=True)\n"
        ),
    )


@pytest.fixture
def sample_test_plan_json(sample_test_plan: TestPlan) -> str:
    return sample_test_plan.model_dump_json()
