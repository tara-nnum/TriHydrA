"""Explicit hydrological unit conversion helpers."""

from __future__ import annotations

import pandas as pd

SECONDS_PER_DAY_PER_MM_KM2 = 86.4


def runoff_mm_day_to_discharge_m3_s(
    series: pd.Series, area_km2: float
) -> pd.Series:
    """Convert mm/day to m³/s while preserving timestamps and NaNs."""
    if area_km2 <= 0:
        raise ValueError("area_km2 must be positive.")
    result = series.copy(deep=True) * float(area_km2) / SECONDS_PER_DAY_PER_MM_KM2
    result.name = series.name
    return result


def discharge_m3_s_to_runoff_mm_day(
    series: pd.Series, area_km2: float
) -> pd.Series:
    """Convert m³/s to mm/day while preserving timestamps and NaNs."""
    if area_km2 <= 0:
        raise ValueError("area_km2 must be positive.")
    result = series.copy(deep=True) * SECONDS_PER_DAY_PER_MM_KM2 / float(area_km2)
    result.name = series.name
    return result


__all__ = [
    "runoff_mm_day_to_discharge_m3_s",
    "discharge_m3_s_to_runoff_mm_day",
]
