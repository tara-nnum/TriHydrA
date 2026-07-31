"""
Preparation of discharge series for TriHydrA Layer 2.

Layer 1 must inspect the raw record, including every missing value and long
gap. Layer 2, by contrast, needs a continuous *temporary analysis copy* for
several hydrological-signature calculations. This module keeps those two
ideas separate:

* the caller's raw pandas Series is never edited;
* an OBS-only run uses the full valid OBS record;
* an OBS/model run is restricted to one identical common calendar;
* missing values are filled only in temporary copies;
* every filled timestamp is recorded in an auditable table.

The default seasonal-climatology fill uses observations from the same
day-of-year neighbourhood across other years. If too few seasonal samples
exist, time interpolation is used inside the record, followed by nearest
valid values at the two record edges. The fallback method is recorded for
each timestamp, so temporary filling is never mistaken for observed data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PreparedLayer2Inputs:
    """Raw-preserving, calculation-ready inputs and their provenance."""

    mode: str
    obs_raw: pd.Series
    model_raw: Optional[pd.Series]
    obs_aligned: pd.Series
    model_aligned: Optional[pd.Series]
    obs_analysis: pd.Series
    model_analysis: Optional[pd.Series]
    imputation_log: pd.DataFrame
    coverage: dict


def _normalise_copy(series: pd.Series, name: str) -> pd.Series:
    """Return a numeric, sorted copy without removing internal NaN values."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    result = series.copy(deep=True)
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result.loc[~result.index.isna()].sort_index()
    if result.index.has_duplicates:
        result = result.groupby(level=0).median()
    result = pd.to_numeric(result, errors="coerce").astype(float)
    result.name = name
    return result


def _record_bounds(series: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return first and last valid timestamps or raise for an empty record."""
    start, end = series.first_valid_index(), series.last_valid_index()
    if start is None or end is None:
        raise ValueError(f"{series.name} contains no valid discharge values.")
    return pd.Timestamp(start), pd.Timestamp(end)


def _temporary_fill(
    series: pd.Series,
    series_name: str,
    method: str,
    window_days: int,
    min_samples: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Fill a calculation copy and describe every value that was inserted."""
    if method not in {"seasonal_climatology", "interpolate", "ffill", "none"}:
        raise ValueError(
            "fill_method must be 'seasonal_climatology', 'interpolate', "
            "'ffill', or 'none'."
        )
    if window_days < 0 or min_samples < 1:
        raise ValueError("window_days must be non-negative and min_samples >= 1.")

    analysis = series.copy(deep=True)
    originally_missing = analysis.isna()
    methods = pd.Series(index=analysis.index, dtype="object")

    if method == "seasonal_climatology":
        valid = analysis.dropna()
        valid_doy = valid.index.dayofyear.to_numpy()
        valid_values = valid.to_numpy(dtype=float)
        for timestamp in analysis.index[originally_missing]:
            target = int(timestamp.dayofyear)
            distance = np.abs(valid_doy - target)
            circular = np.minimum(distance, 366 - distance)
            candidates = valid_values[circular <= window_days]
            if len(candidates) >= min_samples:
                analysis.loc[timestamp] = float(np.median(candidates))
                methods.loc[timestamp] = "seasonal_climatology"

        remaining = analysis.isna()
        interpolated = analysis.interpolate(method="time", limit_area="inside")
        inserted = remaining & interpolated.notna()
        analysis.loc[inserted] = interpolated.loc[inserted]
        methods.loc[inserted] = "time_interpolation_fallback"

        remaining = analysis.isna()
        edge_filled = analysis.ffill().bfill()
        inserted = remaining & edge_filled.notna()
        analysis.loc[inserted] = edge_filled.loc[inserted]
        methods.loc[inserted] = "nearest_value_edge_fallback"
    elif method == "interpolate":
        analysis = analysis.interpolate(method="time").ffill().bfill()
        methods.loc[originally_missing & analysis.notna()] = "time_interpolation"
    elif method == "ffill":
        analysis = analysis.ffill().bfill()
        methods.loc[originally_missing & analysis.notna()] = "forward_or_back_fill"

    rows = []
    for timestamp in analysis.index[originally_missing]:
        rows.append(
            {
                "series": series_name,
                "timestamp": timestamp,
                "temporary_value": analysis.loc[timestamp],
                "method": methods.loc[timestamp]
                if pd.notna(methods.loc[timestamp])
                else "unfilled",
                "temporary_only": True,
            }
        )
    return analysis, pd.DataFrame(
        rows,
        columns=[
            "series", "timestamp", "temporary_value", "method", "temporary_only"
        ],
    )


def prepare_layer2_inputs(
    obs_series: pd.Series,
    model_series: Optional[pd.Series] = None,
    fill_method: str = "seasonal_climatology",
    window_days: int = 15,
    min_samples: int = 5,
) -> PreparedLayer2Inputs:
    """Create OBS-only or common-period OBS/model Layer 2 analysis copies."""
    obs_raw = _normalise_copy(obs_series, "obs")
    obs_start, obs_end = _record_bounds(obs_raw)

    model_raw = (
        None if model_series is None else _normalise_copy(model_series, "model")
    )
    if model_raw is None:
        start, end, mode = obs_start, obs_end, "obs_only"
    else:
        model_start, model_end = _record_bounds(model_raw)
        start, end = max(obs_start, model_start), min(obs_end, model_end)
        if start > end:
            raise ValueError("OBS and model records have no overlapping period.")
        mode = "obs_model_comparison"

    calendar = pd.date_range(start, end, freq="D")
    obs_aligned = obs_raw.reindex(calendar).rename("obs")
    model_aligned = (
        None if model_raw is None else model_raw.reindex(calendar).rename("model")
    )

    obs_analysis, obs_log = _temporary_fill(
        obs_aligned, "obs", fill_method, window_days, min_samples
    )
    if model_aligned is None:
        model_analysis, model_log = None, pd.DataFrame(columns=obs_log.columns)
    else:
        model_analysis, model_log = _temporary_fill(
            model_aligned, "model", fill_method, window_days, min_samples
        )

    log = pd.concat([obs_log, model_log], ignore_index=True)
    coverage = {
        "mode": mode,
        "common_start": start,
        "common_end": end,
        "calendar_days": int(len(calendar)),
        "obs_original_missing": int(obs_aligned.isna().sum()),
        "obs_temporarily_filled": int(
            (obs_aligned.isna() & obs_analysis.notna()).sum()
        ),
        "model_original_missing": (
            None if model_aligned is None else int(model_aligned.isna().sum())
        ),
        "model_temporarily_filled": (
            None
            if model_aligned is None
            else int((model_aligned.isna() & model_analysis.notna()).sum())
        ),
        "fill_method": fill_method,
        "raw_data_modified": False,
    }
    return PreparedLayer2Inputs(
        mode=mode,
        obs_raw=obs_raw,
        model_raw=model_raw,
        obs_aligned=obs_aligned,
        model_aligned=model_aligned,
        obs_analysis=obs_analysis,
        model_analysis=model_analysis,
        imputation_log=log,
        coverage=coverage,
    )


__all__ = ["PreparedLayer2Inputs", "prepare_layer2_inputs"]
