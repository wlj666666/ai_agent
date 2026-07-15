"""Unit tests for the deterministic AEB braking judgment function."""

import math

import pytest

from drivetest_agent.domain.aeb import should_trigger_aeb


class TestShouldTriggerAebTriggers:
    """Cases where braking should trigger."""

    def test_triggers_at_threshold_ttc_with_positive_relative_speed(self) -> None:
        assert should_trigger_aeb(ttc=1.5, relative_speed=10.0, sensor_valid=True) is True

    def test_triggers_below_threshold(self) -> None:
        assert should_trigger_aeb(ttc=1.0, relative_speed=5.0, sensor_valid=True) is True

    def test_triggers_at_zero_ttc_with_positive_relative_speed(self) -> None:
        assert should_trigger_aeb(ttc=0.0, relative_speed=5.0, sensor_valid=True) is True

    def test_does_not_trigger_above_threshold(self) -> None:
        assert should_trigger_aeb(ttc=1.51, relative_speed=10.0, sensor_valid=True) is False

    def test_does_not_trigger_when_relative_speed_is_zero(self) -> None:
        assert should_trigger_aeb(ttc=1.0, relative_speed=0.0, sensor_valid=True) is False

    def test_does_not_trigger_when_relative_speed_is_negative(self) -> None:
        assert should_trigger_aeb(ttc=1.0, relative_speed=-3.0, sensor_valid=True) is False


class TestShouldTriggerAebInvalidSensor:
    """Invalid sensor must raise an explicit error."""

    def test_raises_when_sensor_invalid(self) -> None:
        with pytest.raises(ValueError, match="sensor"):
            should_trigger_aeb(ttc=1.0, relative_speed=10.0, sensor_valid=False)


class TestShouldTriggerAebInvalidTtc:
    """TTC must be a finite non-negative number."""

    @pytest.mark.parametrize("ttc", [-0.1, -1.0])
    def test_raises_for_negative_ttc(self, ttc: float) -> None:
        with pytest.raises(ValueError, match="ttc"):
            should_trigger_aeb(ttc=ttc, relative_speed=10.0, sensor_valid=True)

    @pytest.mark.parametrize("ttc", [math.nan, math.inf, -math.inf])
    def test_raises_for_non_finite_ttc(self, ttc: float) -> None:
        with pytest.raises(ValueError, match="ttc"):
            should_trigger_aeb(ttc=ttc, relative_speed=10.0, sensor_valid=True)


class TestShouldTriggerAebInvalidRelativeSpeed:
    """Relative speed must be a finite number."""

    @pytest.mark.parametrize("relative_speed", [math.nan, math.inf, -math.inf])
    def test_raises_for_non_finite_relative_speed(self, relative_speed: float) -> None:
        with pytest.raises(ValueError, match="relative_speed"):
            should_trigger_aeb(ttc=1.0, relative_speed=relative_speed, sensor_valid=True)
