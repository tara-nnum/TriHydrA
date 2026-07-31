"""Zero flow signature calculation for TriHydrA Layer 2.

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

def calculate_zero_flow_signatures(
    series: pd.Series,
    zero_threshold: float = 1e-6,
) -> SignatureResult:
    """
    Calculate zero-flow frequency and persistence.

    A zero-flow day is defined as Q <= zero_threshold.

    Parameters
    ----------
    series
        Discharge time series.

    zero_threshold
        Numerical threshold used to identify zero or near-zero flow.
    """
    if zero_threshold < 0:
        raise ValueError("zero_threshold cannot be negative.")

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    if valid.empty:
        warnings.append("No valid discharge values are available.")

    zero_condition = (
        (discharge <= zero_threshold)
        & discharge.notna()
    )

    events = _identify_consecutive_events(
        condition=zero_condition,
        original_values=discharge,
    )

    duration_metrics = _event_duration_metrics(events)

    zero_days = int(zero_condition.sum())
    valid_days = int(discharge.notna().sum())

    annual_rows: list[dict[str, Any]] = []

    for year, yearly_values in discharge.groupby(discharge.index.year):
        yearly_zero = zero_condition.reindex(yearly_values.index)

        n_valid = int(yearly_values.notna().sum())
        n_zero = int(yearly_zero.sum())

        annual_rows.append(
            {
                "year": int(year),
                "valid_day_count": n_valid,
                "zero_flow_days": n_zero,
                "zero_flow_frequency": _safe_divide(
                    n_zero,
                    n_valid,
                ),
            }
        )

    monthly_rows: list[dict[str, Any]] = []

    grouped_monthly = discharge.groupby(
        [
            discharge.index.year,
            discharge.index.month,
        ]
    )

    for (year, month), monthly_values in grouped_monthly:
        monthly_zero = zero_condition.reindex(
            monthly_values.index
        )

        n_valid = int(monthly_values.notna().sum())
        n_zero = int(monthly_zero.sum())

        monthly_rows.append(
            {
                "year": int(year),
                "month": int(month),
                "valid_day_count": n_valid,
                "zero_flow_days": n_zero,
                "zero_flow_frequency": _safe_divide(
                    n_zero,
                    n_valid,
                ),
            }
        )

    annual_frequency = pd.DataFrame(annual_rows)
    monthly_frequency = pd.DataFrame(monthly_rows)

    if monthly_frequency.empty:
        monthly_climatology = pd.DataFrame(
            columns=[
                "month",
                "median_zero_flow_frequency",
                "mean_zero_flow_frequency",
                "maximum_zero_flow_frequency",
            ]
        )
    else:
        monthly_climatology = (
            monthly_frequency
            .groupby("month", as_index=False)
            .agg(
                median_zero_flow_frequency=(
                    "zero_flow_frequency",
                    "median",
                ),
                mean_zero_flow_frequency=(
                    "zero_flow_frequency",
                    "mean",
                ),
                maximum_zero_flow_frequency=(
                    "zero_flow_frequency",
                    "max",
                ),
            )
        )

    metrics = {
        "zero_threshold": float(zero_threshold),
        "zero_flow_days": zero_days,
        "zero_flow_ratio": _safe_divide(
            zero_days,
            valid_days,
        ),
        "zero_flow_event_count": duration_metrics["event_count"],
        "mean_zero_flow_duration": duration_metrics[
            "mean_event_duration"
        ],
        "median_zero_flow_duration": duration_metrics[
            "median_event_duration"
        ],
        "maximum_zero_flow_duration": duration_metrics[
            "maximum_event_duration"
        ],
        "minimum_zero_flow_duration": duration_metrics[
            "minimum_event_duration"
        ],
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=1,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "zero_flow_events": events,
            "annual_zero_flow_frequency": annual_frequency,
            "monthly_zero_flow_frequency": monthly_frequency,
            "monthly_zero_flow_climatology": monthly_climatology,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
