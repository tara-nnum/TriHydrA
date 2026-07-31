"""Rising limb signature calculation for TriHydrA Layer 2.

This module owns this complete single-series signature calculation. It performs
no OBS–ML comparison and writes no files. Temporary imputation and provenance
are handled before the calculator is called.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.trihydra.layer2.signature_result import SignatureResult
from src.trihydra.layer2.signature_utils import *

def calculate_rising_limb_signatures(
    series: pd.Series,
    tolerance: float = 0.0,
    minimum_limb_length: int = 1,
) -> SignatureResult:
    """
    Calculate rising-limb signatures.

    Includes:
    - fraction of valid change steps that are rising,
    - mean and median positive dQ/dt,
    - P90 rising rate,
    - maximum rising rate,
    - normalised rising rates,
    - number of rising limbs,
    - rising-limb density,
    - rising-limb durations.

    Rising-limb density is calculated as:

        number of rising limbs / total rising timesteps

    This is equivalent to the inverse of mean rising-limb length.
    """
    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    differences = discharge.diff()

    valid_change = (
        discharge.notna()
        & discharge.shift(1).notna()
    )

    rising_condition = (
        differences > tolerance
    ) & valid_change

    rising_rates = differences.loc[rising_condition].dropna()

    limbs = _segment_directional_limbs(
        series=discharge,
        direction="rising",
        tolerance=tolerance,
        minimum_length=minimum_limb_length,
    )

    valid_change_count = int(valid_change.sum())
    rising_step_count = int(rising_condition.sum())
    rising_limb_count = int(len(limbs))

    if rising_step_count == 0:
        rising_limb_density = np.nan
    else:
        rising_limb_density = (
            rising_limb_count / rising_step_count
        )

    median_flow = valid.median() if not valid.empty else np.nan

    normalised_rates = (
        rising_rates / median_flow
        if pd.notna(median_flow) and median_flow != 0
        else pd.Series(dtype=float)
    )

    metrics = {
        "rising_tolerance": float(tolerance),
        "minimum_limb_length": int(minimum_limb_length),
        "valid_change_step_count": valid_change_count,
        "rising_step_count": rising_step_count,
        "rising_day_fraction": _safe_divide(
            rising_step_count,
            valid_change_count,
        ),
        "mean_rising_rate": (
            float(rising_rates.mean())
            if not rising_rates.empty
            else np.nan
        ),
        "median_rising_rate": (
            float(rising_rates.median())
            if not rising_rates.empty
            else np.nan
        ),
        "p90_rising_rate": (
            float(rising_rates.quantile(0.90))
            if not rising_rates.empty
            else np.nan
        ),
        "maximum_rising_rate": (
            float(rising_rates.max())
            if not rising_rates.empty
            else np.nan
        ),
        "mean_normalised_rising_rate": (
            float(normalised_rates.mean())
            if not normalised_rates.empty
            else np.nan
        ),
        "median_normalised_rising_rate": (
            float(normalised_rates.median())
            if not normalised_rates.empty
            else np.nan
        ),
        "rising_limb_count": rising_limb_count,
        "rising_limb_density": float(rising_limb_density),
        "mean_rising_limb_duration": (
            float(limbs["duration_steps"].mean())
            if not limbs.empty
            else np.nan
        ),
        "median_rising_limb_duration": (
            float(limbs["duration_steps"].median())
            if not limbs.empty
            else np.nan
        ),
        "maximum_rising_limb_duration": (
            float(limbs["duration_steps"].max())
            if not limbs.empty
            else np.nan
        ),
        "median_rising_limb_total_change": (
            float(limbs["total_change"].median())
            if not limbs.empty
            else np.nan
        ),
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=2,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "rising_limbs": limbs,
            "rising_rates": rising_rates.rename(
                "rising_rate"
            ).to_frame(),
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
