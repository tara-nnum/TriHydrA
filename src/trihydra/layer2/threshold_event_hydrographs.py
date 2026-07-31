"""Threshold event hydrographs signature calculation for TriHydrA Layer 2.

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

def calculate_threshold_event_hydrographs(
    series: pd.Series,
    threshold: Optional[float] = None,
    percentile: float = 0.95,
    include_equal: bool = True,
) -> SignatureResult:
    """
    Calculate the event-hydrograph metrics used in the Layer 2 notebook.

    An event is one consecutive spell above a fixed threshold, Q95 by default.
    The event start/end are therefore threshold crossings, not complete
    catchment-response boundaries.
    """
    if not 0 < percentile < 1:
        raise ValueError("percentile must be between 0 and 1.")

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()
    warnings: list[str] = []

    if valid.empty:
        calculated_threshold = np.nan
        warnings.append("No valid discharge values are available.")
    elif threshold is None:
        calculated_threshold = float(valid.quantile(percentile))
    else:
        calculated_threshold = float(threshold)

    if pd.isna(calculated_threshold):
        condition = pd.Series(False, index=discharge.index, dtype=bool)
    elif include_equal:
        condition = discharge.ge(calculated_threshold) & discharge.notna()
    else:
        condition = discharge.gt(calculated_threshold) & discharge.notna()

    event_start = condition & ~condition.shift(1, fill_value=False)
    event_ids = event_start.cumsum()
    rows: list[dict[str, Any]] = []

    for current_event_id in event_ids[condition].unique():
        dates = discharge.index[(event_ids == current_event_id) & condition]
        if len(dates) == 0:
            continue

        start_date = dates.min()
        end_date = dates.max()
        event_flow = discharge.loc[start_date:end_date].dropna()
        if event_flow.empty:
            continue

        peak_date = event_flow.idxmax()
        peak_flow = float(event_flow.max())

        rising_flow = discharge.loc[start_date:peak_date].dropna()
        rising_changes = rising_flow.diff().dropna()
        positive_rises = rising_changes[rising_changes > 0]

        time_to_peak_days = float(
            (peak_date - start_date) / pd.Timedelta(days=1)
        )
        overall_rising_slope = (
            (peak_flow - float(rising_flow.iloc[0])) / time_to_peak_days
            if time_to_peak_days > 0
            else np.nan
        )

        recession_flow = discharge.loc[peak_date:end_date].dropna()
        recession_changes = recession_flow.diff().dropna()
        negative_recessions = recession_changes[recession_changes < 0]

        recession_duration_days = float(
            (end_date - peak_date) / pd.Timedelta(days=1)
        )
        overall_recession_slope = (
            (float(recession_flow.iloc[-1]) - peak_flow)
            / recession_duration_days
            if recession_duration_days > 0
            else np.nan
        )

        positive_recession_flow = recession_flow[recession_flow > 0]
        if len(positive_recession_flow) >= 2:
            elapsed_days = (
                positive_recession_flow.index
                - positive_recession_flow.index[0]
            ) / pd.Timedelta(days=1)
            log_recession_slope = float(
                np.polyfit(
                    elapsed_days,
                    np.log(positive_recession_flow.to_numpy()),
                    1,
                )[0]
            )
        else:
            log_recession_slope = np.nan

        rows.append(
            {
                "event_id": int(current_event_id),
                "event_start": start_date,
                "peak_date": peak_date,
                "event_end": end_date,
                "event_duration_days": int((end_date - start_date).days + 1),
                "start_flow": float(event_flow.iloc[0]),
                "peak_flow": peak_flow,
                "end_flow": float(event_flow.iloc[-1]),
                "time_to_peak_days": time_to_peak_days,
                "mean_positive_rising_rate": (
                    float(positive_rises.mean())
                    if not positive_rises.empty else np.nan
                ),
                "median_positive_rising_rate": (
                    float(positive_rises.median())
                    if not positive_rises.empty else np.nan
                ),
                "maximum_positive_rising_rate": (
                    float(positive_rises.max())
                    if not positive_rises.empty else np.nan
                ),
                "overall_rising_slope": float(overall_rising_slope)
                if pd.notna(overall_rising_slope) else np.nan,
                "mean_recession_slope": (
                    float(negative_recessions.mean())
                    if not negative_recessions.empty else np.nan
                ),
                "median_recession_slope": (
                    float(negative_recessions.median())
                    if not negative_recessions.empty else np.nan
                ),
                "steepest_recession_slope": (
                    float(negative_recessions.min())
                    if not negative_recessions.empty else np.nan
                ),
                "overall_recession_slope": float(overall_recession_slope)
                if pd.notna(overall_recession_slope) else np.nan,
                "log_recession_slope": log_recession_slope,
            }
        )

    events = pd.DataFrame(rows)

    def _metric(column: str, operation: str = "median") -> float:
        if events.empty or column not in events:
            return np.nan
        values = pd.to_numeric(events[column], errors="coerce").dropna()
        if values.empty:
            return np.nan
        return float(getattr(values, operation)())

    duration_years = _basic_metadata(discharge)["duration_years"]

    metrics = {
        "event_threshold": calculated_threshold,
        "threshold_percentile": float(percentile),
        "event_count": int(len(events)),
        "event_frequency_per_year": _safe_divide(len(events), duration_years),
        "median_event_duration_days": _metric("event_duration_days"),
        "median_peak_flow": _metric("peak_flow"),
        "median_rising_limb_rate": _metric("median_positive_rising_rate"),
        "median_overall_rising_slope": _metric("overall_rising_slope"),
        "median_recession_limb_slope": _metric("median_recession_slope"),
        "median_overall_recession_slope": _metric("overall_recession_slope"),
        "median_log_recession_slope": _metric("log_recession_slope"),
        "median_time_to_peak_days": _metric("time_to_peak_days"),
        "maximum_time_to_peak_days": _metric("time_to_peak_days", "max"),
    }

    if events.empty:
        warnings.append("No threshold-exceedance events were detected.")

    return SignatureResult(
        status=_result_status(
            valid_count=len(valid),
            minimum_required=3,
            warnings=warnings,
        ),
        metrics=metrics,
        tables={"events": events},
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
