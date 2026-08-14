"""Five-year epoch drift screening for arbitrary discharge series.

The check detects statistical evolution rather than a hydrological signature.
It builds seasonally adjusted annual flow levels and fits robust slopes to
consecutive multi-year evidence epochs.

No station identifiers, metadata, or imputed values are required. A dated
``pandas.Series`` is required; the same public function accepts observations
and model output.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from trihydra.layer1.check_result import make_result
from trihydra.layer1.step_shift import (
    _monthly_observed_statistics,
    _station_log_offset,
)
from trihydra.layer1.timeseries_validity import get_valid_record


def _annual_level_evidence(
    series: pd.Series,
    minimum_valid_days_per_month: int,
    minimum_valid_months_per_year: int,
) -> tuple[pd.DataFrame, float, float]:
    """Return month-of-year-adjusted annual median levels in log space."""
    monthly = _monthly_observed_statistics(
        series, minimum_valid_days_per_month
    )
    offset = _station_log_offset(series)
    monthly["level_log"] = np.log(monthly["median"] + offset)
    climatology = monthly["level_log"].groupby(monthly.index.month).median()
    monthly["level_anomaly"] = (
        monthly["level_log"]
        - monthly.index.month.map(climatology).to_numpy(dtype=float)
    )
    reference_log = float(monthly["level_log"].median())

    grouped = monthly.groupby(monthly.index.year)
    annual = pd.DataFrame({
        "valid_months": grouped["level_anomaly"].count(),
        "level_anomaly": grouped["level_anomaly"].median(),
    })
    annual.index.name = "year"
    annual.loc[
        annual["valid_months"] < minimum_valid_months_per_year,
        "level_anomaly",
    ] = np.nan
    annual["display_level"] = (
        np.exp(reference_log + annual["level_anomaly"]) - offset
    )
    return annual, offset, reference_log


def _continuous_annual_runs(annual: pd.DataFrame) -> list[pd.DataFrame]:
    valid = annual.dropna(subset=["level_anomaly"]).copy()
    if valid.empty:
        return []
    valid["run"] = valid.index.to_series().diff().ne(1).cumsum().to_numpy()
    return [part.drop(columns="run") for _, part in valid.groupby("run")]


def _annual_noise_scale(runs: list[pd.DataFrame], floor: float) -> float:
    differences: list[float] = []
    for run in runs:
        differences.extend(run["level_anomaly"].diff().abs().dropna().tolist())
    if not differences:
        return floor
    values = np.asarray(differences, dtype=float)
    ordinary = values[values <= np.median(values)]
    centre = float(np.median(ordinary)) if len(ordinary) else 0.0
    mad = (
        float(np.median(np.abs(ordinary - centre)))
        if len(ordinary)
        else 0.0
    )
    return max(centre + 1.4826 * mad, floor)


def _sen_fit(frame: pd.DataFrame) -> tuple[float, float]:
    """Return the robust Theil-Sen slope and intercept for annual evidence."""
    x = frame.index.to_numpy(dtype=float)
    y = frame["level_anomaly"].to_numpy(dtype=float)
    slope, intercept, _, _ = theilslopes(
        y, x, alpha=0.95, method="separate"
    )
    return float(slope), float(intercept)


def _balanced_epochs(
    run: pd.DataFrame,
    epoch_years: int,
) -> list[pd.DataFrame]:
    """Use the maximum number of balanced epochs, each at least epoch_years."""
    epoch_count = len(run) // epoch_years
    if epoch_count < 1:
        return []
    positions = np.array_split(np.arange(len(run)), epoch_count)
    return [
        run.iloc[position].copy()
        for position in positions
        if len(position) >= epoch_years
    ]


def _epoch_points(
    runs: list[pd.DataFrame],
    reference_log: float,
    offset: float,
    noise_scale: float,
    epoch_years: int,
    meaningful_epoch_change_score: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    epoch_number = 0
    for run_number, run in enumerate(runs, start=1):
        for epoch in _balanced_epochs(run, epoch_years):
            epoch_number += 1
            slope, intercept = _sen_fit(epoch)
            years = epoch.index.to_numpy(dtype=float)
            span = max(float(years.max() - years.min()), 1.0)
            change_score = abs(slope) * span / noise_scale
            state = (
                "rising"
                if change_score >= meaningful_epoch_change_score and slope > 0
                else "falling"
                if change_score >= meaningful_epoch_change_score
                else "stable"
            )
            for year in epoch.index:
                fitted_anomaly = float(intercept + slope * float(year))
                rows.append({
                    "continuous_run": run_number,
                    "epoch": epoch_number,
                    "year": int(year),
                    "fitted_level": float(
                        np.exp(reference_log + fitted_anomaly) - offset
                    ),
                    "sen_slope_percent_per_year": float(
                        100.0 * np.expm1(slope)
                    ),
                    "epoch_change_score": float(change_score),
                    "state": state,
                })
    return pd.DataFrame(rows)


def _state_for_slope(
    slope: float,
    span: float,
    noise_scale: float,
    meaningful_epoch_change_score: float,
) -> str:
    score = abs(slope) * max(span, 1.0) / noise_scale
    if score < meaningful_epoch_change_score:
        return "stable"
    return "rising" if slope > 0 else "falling"


def _overview_points(
    runs: list[pd.DataFrame],
    epoch_points: pd.DataFrame,
    reference_log: float,
    offset: float,
    noise_scale: float,
    meaningful_epoch_change_score: float,
    overview_epochs_per_segment: int,
    maximum_overview_slopes: int,
) -> pd.DataFrame:
    if epoch_points.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    overview_number = 0
    for run_number, run in enumerate(runs, start=1):
        epoch_ids = (
            epoch_points.loc[
                epoch_points["continuous_run"].eq(run_number), "epoch"
            ]
            .drop_duplicates()
            .to_numpy(dtype=int)
        )
        if not len(epoch_ids):
            continue
        overview_count = max(
            1, int(np.ceil(len(epoch_ids) / overview_epochs_per_segment))
        )
        overview_count = min(overview_count, maximum_overview_slopes)
        best_groups: list[np.ndarray] | None = None
        best_error = np.inf
        for cuts in combinations(range(1, len(epoch_ids)), overview_count - 1):
            bounds = (0, *cuts, len(epoch_ids))
            groups = [
                epoch_ids[bounds[i]:bounds[i + 1]]
                for i in range(len(bounds) - 1)
            ]
            total_error = 0.0
            for group in groups:
                years = epoch_points.loc[
                    epoch_points["epoch"].isin(group), "year"
                ].unique()
                candidate = run.loc[run.index.isin(years)].copy()
                slope, intercept = _sen_fit(candidate)
                fitted = intercept + slope * candidate.index.to_numpy(dtype=float)
                total_error += float(np.square(
                    candidate["level_anomaly"].to_numpy(dtype=float) - fitted
                ).sum())
            if total_error < best_error:
                best_error = total_error
                best_groups = groups
        if best_groups is None:
            best_groups = [epoch_ids]

        merged: list[tuple[np.ndarray, str]] = []
        for group in best_groups:
            years = epoch_points.loc[
                epoch_points["epoch"].isin(group), "year"
            ].unique()
            candidate = run.loc[run.index.isin(years)].copy()
            slope, _ = _sen_fit(candidate)
            state = _state_for_slope(
                slope,
                float(candidate.index.max() - candidate.index.min()),
                noise_scale,
                meaningful_epoch_change_score,
            )
            if merged and merged[-1][1] == state:
                previous, _ = merged[-1]
                merged[-1] = (np.concatenate([previous, group]), state)
            else:
                merged.append((group, state))

        for group, _ in merged:
            overview_number += 1
            years = epoch_points.loc[
                epoch_points["epoch"].isin(group), "year"
            ].unique()
            evidence = run.loc[run.index.isin(years)].copy()
            slope, intercept = _sen_fit(evidence)
            span = float(evidence.index.max() - evidence.index.min())
            state = _state_for_slope(
                slope, span, noise_scale, meaningful_epoch_change_score
            )
            for year in evidence.index:
                fitted_anomaly = float(intercept + slope * float(year))
                rows.append({
                    "continuous_run": run_number,
                    "overview": overview_number,
                    "year": int(year),
                    "fitted_level": float(
                        np.exp(reference_log + fitted_anomaly) - offset
                    ),
                    "sen_slope_percent_per_year": float(
                        100.0 * np.expm1(slope)
                    ),
                    "state": state,
                })
    return pd.DataFrame(rows)


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    output = frame.reset_index(drop=frame.index.name is None).to_dict("records")
    for row in output:
        for key, value in list(row.items()):
            if isinstance(value, np.generic):
                row[key] = value.item()
    return output


def check_epoch_drift(
    series: pd.Series,
    series_type: str = "unknown",
    minimum_valid_days_per_month: int = 10,
    minimum_valid_months_per_year: int = 8,
    epoch_years: int = 5,
    minimum_valid_annual_levels: int = 5,
    annual_noise_floor_log: float = 0.03,
    meaningful_epoch_change_score: float = 1.0,
    tier_3_minimum_stable_fraction: float = 0.75,
    tier_2_minimum_stable_fraction: float = 0.50,
    overview_epochs_per_segment: int = 4,
    maximum_overview_slopes: int = 4,
) -> dict:
    """Diagnose stable or drifting epoch-scale flow-level evolution."""
    record = get_valid_record(series)
    if record.empty:
        return make_result(
            check="epoch_drift", flag=False, value=None,
            series_type=series_type, status="skipped",
            reason_skipped="No valid data found.",
            message="Epoch drift not calculated: no valid data found.",
            diagnosis="Not assessed", tier="Not assessed",
            annual_level_evidence=[], five_year_epochs=[],
            consolidated_overview_slopes=[], overview_diagnosis=[],
        )
    if epoch_years < 2 or minimum_valid_annual_levels < epoch_years:
        raise ValueError(
            "minimum_valid_annual_levels must be at least epoch_years, and "
            "epoch_years must be at least 2."
        )
    if overview_epochs_per_segment < 1:
        raise ValueError("overview_epochs_per_segment must be at least 1.")
    if maximum_overview_slopes < 1:
        raise ValueError("maximum_overview_slopes must be at least 1.")
    if not (
        0 <= tier_2_minimum_stable_fraction
        < tier_3_minimum_stable_fraction <= 1
    ):
        raise ValueError(
            "Epoch-drift stable-fraction cutoffs must satisfy 0 <= Tier 2 "
            "minimum < Tier 3 minimum <= 1."
        )

    annual, offset, reference_log = _annual_level_evidence(
        record,
        minimum_valid_days_per_month,
        minimum_valid_months_per_year,
    )
    runs = _continuous_annual_runs(annual)
    analysis_runs = [
        run for run in runs if len(run) >= minimum_valid_annual_levels
    ]
    noise_scale = _annual_noise_scale(analysis_runs, annual_noise_floor_log)
    points = _epoch_points(
        analysis_runs, reference_log, offset, noise_scale, epoch_years,
        meaningful_epoch_change_score,
    )
    valid_years = int(annual["level_anomaly"].notna().sum())
    qualifying_runs = len(analysis_runs)
    if not qualifying_runs or points.empty:
        reason = (
            f"No continuous run contained {minimum_valid_annual_levels} "
            "valid annual levels."
        )
        return make_result(
            check="epoch_drift", flag=False, value=None,
            series_type=series_type, status="skipped",
            reason_skipped=reason,
            message=f"Epoch drift not calculated: {reason}",
            diagnosis="Not assessed", tier="Not assessed",
            valid_annual_levels=valid_years,
            annual_level_evidence=_records(annual),
            five_year_epochs=[], consolidated_overview_slopes=[],
            overview_diagnosis=[],
        )
    stable_year_fraction = float(points["state"].eq("stable").mean())
    if stable_year_fraction >= tier_3_minimum_stable_fraction:
        tier = "Tier 3"
    elif stable_year_fraction >= tier_2_minimum_stable_fraction:
        tier = "Tier 2"
    else:
        tier = "Tier 1"
    diagnosis = "Drifting" if tier in {"Tier 1", "Tier 2"} else "Stable"
    description = f"stable for {100 * stable_year_fraction:.1f}% of assessed years"

    directional = set(points.loc[points["state"].ne("stable"), "state"])
    dominant = (
        "stable" if not directional
        else next(iter(directional)) if len(directional) == 1
        else "mixed"
    )
    overview = _overview_points(
        analysis_runs, points, reference_log, offset, noise_scale,
        meaningful_epoch_change_score, overview_epochs_per_segment,
        maximum_overview_slopes,
    )
    overview_rows: list[dict] = []
    if not overview.empty:
        for number, period in overview.groupby("overview", sort=False):
            overview_rows.append({
                "overview": int(number),
                "period": f"{int(period['year'].min())}-{int(period['year'].max())}",
                "behaviour": str(period["state"].iloc[0]),
            })
    flag = tier in {"Tier 1", "Tier 2"}
    return make_result(
        check="epoch_drift",
        flag=flag,
        value=round(stable_year_fraction, 3),
        series_type=series_type,
        finding_status="candidate_detected" if flag else "passed",
        message=(
            f"Epoch drift diagnosis: {diagnosis} ({tier}); {description}. "
            f"Calculated {int(points['epoch'].nunique())} base epoch(s) and "
            f"{int(overview['overview'].nunique()) if not overview.empty else 0} "
            "consolidated overview slope(s)."
        ),
        diagnosis=diagnosis,
        tier=tier,
        dominant_behaviour=dominant,
        evidence_description=description,
        valid_annual_levels=valid_years,
        base_epoch_count=int(points["epoch"].nunique()),
        overview_slope_count=(
            int(overview["overview"].nunique()) if not overview.empty else 0
        ),
        stable_year_fraction=stable_year_fraction,
        annual_noise_scale_log=noise_scale,
        annual_level_evidence=_records(annual),
        five_year_epochs=_records(points),
        consolidated_overview_slopes=_records(overview),
        overview_diagnosis=overview_rows,
        resolved_settings={
            "minimum_valid_days_per_month": minimum_valid_days_per_month,
            "minimum_valid_months_per_year": minimum_valid_months_per_year,
            "epoch_years": epoch_years,
            "minimum_valid_annual_levels": minimum_valid_annual_levels,
            "annual_noise_floor_log": annual_noise_floor_log,
            "meaningful_epoch_change_score": meaningful_epoch_change_score,
            "tier_3_minimum_stable_fraction": tier_3_minimum_stable_fraction,
            "tier_2_minimum_stable_fraction": tier_2_minimum_stable_fraction,
            "overview_epochs_per_segment": overview_epochs_per_segment,
            "maximum_overview_slopes": maximum_overview_slopes,
        },
    )


__all__ = ["check_epoch_drift"]
