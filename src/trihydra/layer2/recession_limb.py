"""Recession limb signature calculation for TriHydrA Layer 2.

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

def calculate_recession_limb_signatures(
    series: pd.Series,
    tolerance: float = 0.0,
    minimum_limb_length: int = 1,
) -> SignatureResult:
    """
    Calculate recession-limb signatures.

    Includes ordinary discharge-decrease rates and log-flow recession
    rates.

    Log recession rate is calculated for positive consecutive flows as:

        log(Q[t]) - log(Q[t-1])

    Negative values indicate recession.
    """
    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    differences = discharge.diff()

    valid_change = (
        discharge.notna()
        & discharge.shift(1).notna()
    )

    recession_condition = (
        differences < -tolerance
    ) & valid_change

    recession_rates = differences.loc[
        recession_condition
    ].dropna()

    positive_pair = (
        (discharge > 0)
        & (discharge.shift(1) > 0)
        & valid_change
    )

    log_difference = (
        np.log(discharge)
        - np.log(discharge.shift(1))
    )

    log_recession_rates = log_difference.loc[
        recession_condition & positive_pair
    ].dropna()

    limbs = _segment_directional_limbs(
        series=discharge,
        direction="falling",
        tolerance=tolerance,
        minimum_length=minimum_limb_length,
    )

    valid_change_count = int(valid_change.sum())
    recession_step_count = int(recession_condition.sum())

    median_flow = valid.median() if not valid.empty else np.nan

    normalised_rates = (
        recession_rates / median_flow
        if pd.notna(median_flow) and median_flow != 0
        else pd.Series(dtype=float)
    )

    metrics = {
        "recession_tolerance": float(tolerance),
        "minimum_limb_length": int(minimum_limb_length),
        "valid_change_step_count": valid_change_count,
        "recession_step_count": recession_step_count,
        "recession_day_fraction": _safe_divide(
            recession_step_count,
            valid_change_count,
        ),
        "mean_recession_rate": (
            float(recession_rates.mean())
            if not recession_rates.empty
            else np.nan
        ),
        "median_recession_rate": (
            float(recession_rates.median())
            if not recession_rates.empty
            else np.nan
        ),
        "p10_recession_rate": (
            float(recession_rates.quantile(0.10))
            if not recession_rates.empty
            else np.nan
        ),
        "minimum_recession_rate": (
            float(recession_rates.min())
            if not recession_rates.empty
            else np.nan
        ),
        "mean_normalised_recession_rate": (
            float(normalised_rates.mean())
            if not normalised_rates.empty
            else np.nan
        ),
        "median_normalised_recession_rate": (
            float(normalised_rates.median())
            if not normalised_rates.empty
            else np.nan
        ),
        "mean_log_recession_rate": (
            float(log_recession_rates.mean())
            if not log_recession_rates.empty
            else np.nan
        ),
        "median_log_recession_rate": (
            float(log_recession_rates.median())
            if not log_recession_rates.empty
            else np.nan
        ),
        "recession_limb_count": int(len(limbs)),
        "mean_recession_limb_duration": (
            float(limbs["duration_steps"].mean())
            if not limbs.empty
            else np.nan
        ),
        "median_recession_limb_duration": (
            float(limbs["duration_steps"].median())
            if not limbs.empty
            else np.nan
        ),
        "maximum_recession_limb_duration": (
            float(limbs["duration_steps"].max())
            if not limbs.empty
            else np.nan
        ),
        "median_recession_limb_total_change": (
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
            "recession_limbs": limbs,
            "recession_rates": recession_rates.rename(
                "recession_rate"
            ).to_frame(),
            "log_recession_rates": log_recession_rates.rename(
                "log_recession_rate"
            ).to_frame(),
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
