"""Deterministic simulated AEB braking judgment."""

from __future__ import annotations

import math

TTC_THRESHOLD_SECONDS = 1.5


def _validate_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")


def should_trigger_aeb(*, ttc: float, relative_speed: float, sensor_valid: bool) -> bool:
    """Return whether AEB braking should trigger for the given inputs.

    Braking triggers if and only if TTC <= 1.5 seconds, relative speed is
    strictly positive, and the sensor is valid. Invalid sensor state raises
    ``ValueError`` instead of returning a boolean result.
    """
    if not sensor_valid:
        raise ValueError("sensor must be valid")

    _validate_finite(ttc, "ttc")
    _validate_finite(relative_speed, "relative_speed")

    if ttc < 0:
        raise ValueError("ttc must be non-negative")

    return ttc <= TTC_THRESHOLD_SECONDS and relative_speed > 0
