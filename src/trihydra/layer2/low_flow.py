"""Low flow signature calculation for TriHydrA Layer 2.

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

def calculate_low_flow_signatures(
    series: pd.Series,
    threshold: Optional[float] = None,
    percentile: float = 0.05,
    include_equal: bool = True,
) -> SignatureResult:
    """
    Calculate low-flow magnitude, frequency, and duration.

    By default, the low-flow threshold is the fifth percentile of the
    valid discharge values.

    Parameters
    ----------
    series
        Discharge time series.

    threshold
        Optional externally supplied threshold. During later OBS-ML
        comparison, this can be the OBS-derived threshold.

    percentile
        Quantile used when threshold is not supplied. Default is 0.05.

    include_equal
        If True, low flow is Q <= threshold.
        If False, low flow is Q < threshold.

    Returns
    -------
    SignatureResult
        Includes scalar metrics and a low-flow event table.
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
        low_condition = pd.Series(
            False,
            index=discharge.index,
            dtype=bool,
        )
    elif include_equal:
        low_condition = discharge <= calculated_threshold
    else:
        low_condition = discharge < calculated_threshold

    low_condition = low_condition & discharge.notna()

    events = _identify_consecutive_events(
        condition=low_condition,
        original_values=discharge,
    )

    duration_metrics = _event_duration_metrics(events)

    low_flow_days = int(low_condition.sum())
    valid_days = int(discharge.notna().sum())

    annual_rows: list[dict[str, Any]] = []

    for year, yearly_values in discharge.groupby(discharge.index.year):
        yearly_condition = low_condition.reindex(yearly_values.index)

        year_valid_days = int(yearly_values.notna().sum())
        year_low_days = int(yearly_condition.sum())

        annual_rows.append(
            {
                "year": int(year),
                "valid_day_count": year_valid_days,
                "low_flow_days": year_low_days,
                "low_flow_frequency": _safe_divide(
                    year_low_days,
                    year_valid_days,
                ),
            }
        )

    annual_frequency = pd.DataFrame(annual_rows)

    metrics = {
        "low_flow_threshold": calculated_threshold,
        "threshold_percentile": float(percentile),
        "low_flow_days": low_flow_days,
        "low_flow_frequency": _safe_divide(
            low_flow_days,
            valid_days,
        ),
        "low_flow_event_count": duration_metrics["event_count"],
        "mean_low_flow_duration": duration_metrics[
            "mean_event_duration"
        ],
        "median_low_flow_duration": duration_metrics[
            "median_event_duration"
        ],
        "maximum_low_flow_duration": duration_metrics[
            "maximum_event_duration"
        ],
        "minimum_low_flow_duration": duration_metrics[
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
            "low_flow_events": events,
            "annual_low_flow_frequency": annual_frequency,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
