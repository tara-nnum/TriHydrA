"""High flow signature calculation for TriHydrA Layer 2.

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

def calculate_high_flow_signatures(
    series: pd.Series,
    threshold: Optional[float] = None,
    percentile: float = 0.95,
    include_equal: bool = True,
) -> SignatureResult:
    """
    Calculate high-flow magnitude, frequency, and duration.

    By default, the high-flow threshold is the 95th percentile of the
    valid discharge values.

    Parameters
    ----------
    series
        Discharge time series.

    threshold
        Optional externally supplied threshold.

    percentile
        Quantile used when threshold is not supplied. Default is 0.95.

    include_equal
        If True, high flow is Q >= threshold.
        If False, high flow is Q > threshold.
    """
    if not 0 < percentile < 1:
        raise ValueError("percentile must be between 0 and 1.")

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    if valid.empty:
        warnings.append("No valid discharge values are available.")
        calculated_threshold = np.nan
    elif threshold is None:
        calculated_threshold = float(valid.quantile(percentile))
    else:
        calculated_threshold = float(threshold)

    if pd.isna(calculated_threshold):
        high_condition = pd.Series(
            False,
            index=discharge.index,
            dtype=bool,
        )
    elif include_equal:
        high_condition = discharge >= calculated_threshold
    else:
        high_condition = discharge > calculated_threshold

    high_condition = high_condition & discharge.notna()

    events = _identify_consecutive_events(
        condition=high_condition,
        original_values=discharge,
    )

    duration_metrics = _event_duration_metrics(events)

    high_flow_days = int(high_condition.sum())
    valid_days = int(discharge.notna().sum())

    annual_rows: list[dict[str, Any]] = []

    for year, yearly_values in discharge.groupby(discharge.index.year):
        yearly_condition = high_condition.reindex(yearly_values.index)

        year_valid_days = int(yearly_values.notna().sum())
        year_high_days = int(yearly_condition.sum())

        annual_rows.append(
            {
                "year": int(year),
                "valid_day_count": year_valid_days,
                "high_flow_days": year_high_days,
                "high_flow_frequency": _safe_divide(
                    year_high_days,
                    year_valid_days,
                ),
            }
        )

    annual_frequency = pd.DataFrame(annual_rows)

    metrics = {
        "high_flow_threshold": calculated_threshold,
        "threshold_percentile": float(percentile),
        "high_flow_days": high_flow_days,
        "high_flow_frequency": _safe_divide(
            high_flow_days,
            valid_days,
        ),
        "high_flow_event_count": duration_metrics["event_count"],
        "mean_high_flow_duration": duration_metrics[
            "mean_event_duration"
        ],
        "median_high_flow_duration": duration_metrics[
            "median_event_duration"
        ],
        "maximum_high_flow_duration": duration_metrics[
            "maximum_event_duration"
        ],
        "minimum_high_flow_duration": duration_metrics[
            "minimum_event_duration"
        ],
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=10,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "high_flow_events": events,
            "annual_high_flow_frequency": annual_frequency,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
