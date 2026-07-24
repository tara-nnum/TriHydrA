"""
discharge_comparison.py

Metrics comparing a target gauge's discharge against a context
candidate's discharge: does the nearby gauge's behaviour corroborate
what's happening at the target, or not.

Written generically against two already-loaded pandas Series -- this
file never touches a CSV or a NetCDF directly, and doesn't care which
one a series came from. That separation is deliberate: if candidate
data ever comes from somewhere other than the network NetCDF later,
nothing here needs to change.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


def robust_normalise(series: pd.Series) -> pd.Series:
    """(x - median) / IQR -- robust to the outliers/floods that a
    plain z-score would be distorted by, and puts two gauges with very
    different catchment areas on a comparable scale for plotting."""
    series = pd.to_numeric(series, errors="coerce")
    median = series.median()
    iqr = series.quantile(0.75) - series.quantile(0.25)

    if pd.isna(iqr) or iqr == 0:
        return series * np.nan

    return (series - median) / iqr


def flashiness_index(series: pd.Series) -> float:
    """Richards-Baker flashiness: sum of |day-to-day change| / total flow."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2 or values.sum() == 0:
        return np.nan
    return float(values.diff().abs().sum() / values.sum())


def direction_agreement(target: pd.Series, neighbour: pd.Series) -> float:
    """
    Fraction of overlapping days where target and neighbour both rose,
    or both fell, after a light 3-day smoothing (raw daily noise would
    otherwise dominate a day-to-day sign comparison).
    """
    frame = pd.concat(
        [target.rename("target"), neighbour.rename("neighbour")], axis=1
    ).dropna()

    if len(frame) < 10:
        return np.nan

    smoothed = frame.rolling(3, center=True, min_periods=1).median()
    target_change = np.sign(smoothed["target"].diff())
    neighbour_change = np.sign(smoothed["neighbour"].diff())

    valid = target_change.notna() & neighbour_change.notna()
    if not valid.any():
        return np.nan

    return float((target_change[valid] == neighbour_change[valid]).mean())


def high_flow_agreement(target: pd.Series, neighbour: pd.Series, lag_days: int = 7) -> float:
    """
    Of the target's own high-flow days (>= its own 95th percentile),
    what fraction have the neighbour also showing elevated flow
    (>= the neighbour's own 95th percentile) within +/- lag_days.
    Each gauge's threshold is its own -- this is about co-occurrence,
    not requiring both gauges to share the same absolute scale.
    """
    frame = pd.concat(
        [target.rename("target"), neighbour.rename("neighbour")], axis=1
    ).dropna()

    if len(frame) < 30:
        return np.nan

    target_high = frame["target"] >= frame["target"].quantile(0.95)
    neighbour_high = frame["neighbour"] >= frame["neighbour"].quantile(0.95)

    neighbour_near_high = (
        neighbour_high.rolling(2 * lag_days + 1, center=True, min_periods=1)
        .max()
        .astype(bool)
    )

    if target_high.sum() == 0:
        return np.nan

    return float(neighbour_near_high[target_high].mean())


def compare_target_with_candidates(
    target_series: pd.Series,
    candidates: pd.DataFrame,
    candidate_series_loader: Callable[[str], pd.Series],
    lag_days: int = 7,
) -> pd.DataFrame:
    """
    Run every comparison metric between `target_series` and each
    candidate gauge in `candidates` (as returned by
    gauge_network.find_context_candidates).

    `candidate_series_loader(gauge_id) -> pd.Series` is a callback, not
    a fixed data source -- layer3.py supplies one backed by
    nc_loader.py today, so this function stays agnostic about where
    candidate discharge actually comes from.

    Raw discharge magnitudes are not directly comparable across gauges
    with different catchment areas -- correlation and event agreement
    matter more here than the raw numbers do.
    """
    rows = []

    for _, candidate in candidates.iterrows():
        candidate_id = candidate["gauge_id"]
        candidate_series = candidate_series_loader(candidate_id)

        aligned = pd.concat(
            [target_series.rename("target"), candidate_series.rename("candidate")],
            axis=1,
        ).dropna()

        if aligned.empty:
            continue

        norm_target = robust_normalise(aligned["target"])
        norm_candidate = robust_normalise(aligned["candidate"])

        rows.append({
            "candidate_gauge_id": candidate_id,
            "candidate_station_name": candidate.get("StationName"),
            "same_river": candidate["same_river"],
            "distance_km": candidate["distance_km"],
            "candidate_area_km2": candidate["area_km2"],
            "overlap_start": aligned.index.min(),
            "overlap_end": aligned.index.max(),
            "overlap_days": len(aligned),
            "candidate_mean": aligned["candidate"].mean(),
            "candidate_median": aligned["candidate"].median(),
            "candidate_p05": aligned["candidate"].quantile(0.05),
            "candidate_p95": aligned["candidate"].quantile(0.95),
            "candidate_flashiness": flashiness_index(aligned["candidate"]),
            "candidate_lag1_autocorrelation": aligned["candidate"].autocorr(lag=1),
            "normalised_spearman_correlation": pd.concat(
                [norm_target.rename("target"), norm_candidate.rename("candidate")], axis=1
            ).corr(method="spearman").iloc[0, 1],
            "direction_agreement_fraction": direction_agreement(
                aligned["target"], aligned["candidate"]
            ),
            "target_high_flow_supported_fraction": high_flow_agreement(
                aligned["target"], aligned["candidate"], lag_days=lag_days
            ),
        })

    return pd.DataFrame(rows)
