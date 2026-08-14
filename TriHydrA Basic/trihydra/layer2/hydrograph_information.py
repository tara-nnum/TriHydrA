"""Observed high-flow events and representative hydrograph information."""

from __future__ import annotations

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "event_id", "event_start", "peak_date", "event_end",
    "trigger_threshold", "boundary_threshold", "start_flow",
    "peak_flow", "end_flow", "time_to_peak_days", "recession_days",
    "event_duration_days", "rising_slope", "recession_slope",
]


def calculate_high_flow_events(
    raw_series: pd.Series,
    *,
    trigger_threshold: float,
    boundary_threshold: float,
) -> pd.DataFrame:
    """Find contiguous high-flow periods that reach the trigger threshold."""
    if not np.isfinite(trigger_threshold) or not np.isfinite(boundary_threshold):
        return pd.DataFrame(columns=EVENT_COLUMNS)
    if boundary_threshold > trigger_threshold:
        raise ValueError("boundary_threshold cannot exceed trigger_threshold.")
    series = pd.to_numeric(raw_series, errors="coerce").sort_index()
    valid = series.dropna()
    if len(valid) < 3:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    daily = series.index.to_series().diff().eq(pd.Timedelta(days=1)).to_numpy()
    above = series.ge(boundary_threshold) & series.notna()
    starts = above & ~(above.shift(1, fill_value=False) & daily)
    event_ids = starts.cumsum()
    rows = []
    for _, values in series[above].groupby(event_ids[above]):
        if values.empty or float(values.max()) < trigger_threshold:
            continue
        start, end, peak_date = values.index.min(), values.index.max(), values.idxmax()
        peak_flow = float(values.max())
        rise_days = float((peak_date - start) / pd.Timedelta(days=1))
        recession_days = float((end - peak_date) / pd.Timedelta(days=1))
        rows.append({
            "event_id": len(rows) + 1,
            "event_start": start, "peak_date": peak_date, "event_end": end,
            "trigger_threshold": float(trigger_threshold),
            "boundary_threshold": float(boundary_threshold),
            "start_flow": float(values.iloc[0]), "peak_flow": peak_flow,
            "end_flow": float(values.iloc[-1]), "time_to_peak_days": rise_days,
            "recession_days": recession_days,
            "event_duration_days": float((end - start) / pd.Timedelta(days=1) + 1),
            "rising_slope": (peak_flow - float(values.iloc[0])) / rise_days if rise_days > 0 else np.nan,
            "recession_slope": (float(values.iloc[-1]) - peak_flow) / recession_days if recession_days > 0 else np.nan,
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def select_representative_event(events: pd.DataFrame) -> pd.Series | None:
    """Return the real event closest to the multimetric robust median."""
    if events.empty:
        return None
    eligible = events
    if "representative_eligible" in events:
        eligible = events[events["representative_eligible"].fillna(False)]
    if eligible.empty:
        return None
    columns = [
        "peak_flow", "event_duration_days", "time_to_peak_days",
        "rising_slope", "recession_slope",
    ]
    values = eligible[columns].apply(pd.to_numeric, errors="coerce")
    medians = values.median()
    scale = (values - medians).abs().median().replace(0, np.nan)
    scale = scale.fillna(values.std(ddof=0).replace(0, np.nan)).fillna(1.0)
    available = values.notna()
    distance = ((((values - medians) / scale) ** 2).where(available).sum(axis=1)
                / available.sum(axis=1).clip(lower=1))
    return eligible.loc[distance.idxmin()].copy()
