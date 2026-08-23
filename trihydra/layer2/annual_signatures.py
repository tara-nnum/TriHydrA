"""Annual hydrological signatures for one discharge series.

Source values are copied, never filled, and never modified. Calculations
that require adjacent days use genuinely consecutive valid calendar days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ANNUAL_COLUMNS = [
    "year", "mean_flow", "median_flow", "minimum_flow", "maximum_flow",
    "raw_minimum_flow", "raw_maximum_flow",
    "extrema_excluded_candidate_count", "flashiness_index",
    "baseflow_index", "seasonality_index", "lag1_autocorrelation",
    "wettest_month", "driest_month", "valid_days", "usable_months",
]

MONTHLY_COLUMNS = [
    "year", "month", "monthly_mean", "monthly_median", "valid_days",
]


def _daily_pairs(values: pd.Series) -> pd.Series:
    daily = values.index.to_series().diff().eq(pd.Timedelta(days=1)).to_numpy()
    return values.notna() & values.shift(1).notna() & daily


def _flashiness(values: pd.Series) -> float:
    denominator = float(values.dropna().sum())
    pairs = _daily_pairs(values)
    if denominator <= 0 or not pairs.any():
        return np.nan
    return float(values.diff().abs().where(pairs).sum() / denominator)


def _lag1(values: pd.Series) -> float:
    pairs = _daily_pairs(values)
    if int(pairs.sum()) < 3:
        return np.nan
    return float(values[pairs].corr(values.shift(1)[pairs]))


def _lyne_hollick(values: np.ndarray, alpha: float, passes: int) -> np.ndarray:
    current = values.astype(float)
    for pass_number in range(passes):
        forward = pass_number % 2 == 0
        quickflow = np.zeros(len(current), dtype=float)
        factor = (1.0 + alpha) / 2.0
        order = range(1, len(current)) if forward else range(len(current) - 2, -1, -1)
        step = -1 if forward else 1
        for index in order:
            previous = index + step
            quickflow[index] = max(
                alpha * quickflow[previous]
                + factor * (current[index] - current[previous]),
                0.0,
            )
        current = np.clip(current - quickflow, 0.0, current)
    return current


def _baseflow_index(
    values: pd.Series,
    alpha: float,
    passes: int,
    minimum_segment_days: int,
) -> float:
    valid = values.notna()
    daily = values.index.to_series().diff().eq(pd.Timedelta(days=1)).to_numpy()
    starts = valid & ~(valid.shift(1, fill_value=False) & daily)
    groups = starts.cumsum()
    baseflow_sum = 0.0
    discharge_sum = 0.0
    for _, segment in values[valid].groupby(groups[valid]):
        if len(segment) < minimum_segment_days or (segment < 0).any():
            continue
        array = segment.to_numpy(dtype=float)
        baseflow_sum += float(_lyne_hollick(array, alpha, passes).sum())
        discharge_sum += float(array.sum())
    return baseflow_sum / discharge_sum if discharge_sum > 0 else np.nan


def _monthly_profile(values: pd.Series, minimum_valid_days: int) -> pd.DataFrame:
    rows = []
    for month, month_values in values.groupby(values.index.month):
        valid = month_values.dropna()
        if len(valid) >= minimum_valid_days:
            rows.append({
                "month": int(month),
                "monthly_mean": float(valid.mean()),
                "monthly_median": float(valid.median()),
                "valid_days": int(len(valid)),
            })
    return pd.DataFrame(rows)


def _walsh_lawler(months: pd.DataFrame) -> float:
    if len(months) != 12:
        return np.nan
    values = months.sort_values("month")["monthly_mean"].to_numpy(dtype=float)
    total = float(values.sum())
    return float(np.abs(values - total / 12.0).sum() / total) if total > 0 else np.nan


def calculate_annual_signatures(
    raw_series: pd.Series,
    *,
    excluded_extrema_timestamps: pd.DatetimeIndex | None = None,
    minimum_valid_days_per_year: int = 30,
    minimum_valid_days_per_month: int = 10,
    baseflow_alpha: float = 0.925,
    baseflow_passes: int = 3,
    minimum_baseflow_segment_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Calculate annual signatures without modifying the source record."""
    series = pd.to_numeric(raw_series.copy(deep=True), errors="coerce").sort_index()
    series = series.loc[~series.index.duplicated(keep="first")]
    valid_record = series.dropna()
    if valid_record.empty:
        raise ValueError("No valid observations are available for Layer 2.")
    annual_rows: list[dict] = []
    monthly_frames: list[pd.DataFrame] = []
    excluded = pd.DatetimeIndex([])
    if excluded_extrema_timestamps is not None:
        excluded = pd.DatetimeIndex(excluded_extrema_timestamps)
    for year, year_values in series.groupby(series.index.year):
        valid = year_values.dropna()
        if len(valid) < minimum_valid_days_per_year:
            continue
        months = _monthly_profile(year_values, minimum_valid_days_per_month)
        if not months.empty:
            months.insert(0, "year", int(year))
            monthly_frames.append(months)
        complete = len(months) == 12
        excluded_in_year = valid.index.intersection(excluded)
        extrema_values = valid.drop(index=excluded_in_year)
        annual_rows.append({
            "year": int(year),
            "mean_flow": float(valid.mean()),
            "median_flow": float(valid.median()),
            "minimum_flow": (
                float(extrema_values.min()) if not extrema_values.empty else np.nan
            ),
            "maximum_flow": (
                float(extrema_values.max()) if not extrema_values.empty else np.nan
            ),
            "raw_minimum_flow": float(valid.min()),
            "raw_maximum_flow": float(valid.max()),
            "extrema_excluded_candidate_count": int(len(excluded_in_year)),
            "flashiness_index": _flashiness(year_values),
            "baseflow_index": _baseflow_index(
                year_values, baseflow_alpha, baseflow_passes,
                minimum_baseflow_segment_days,
            ),
            "seasonality_index": _walsh_lawler(months),
            "lag1_autocorrelation": _lag1(year_values),
            "wettest_month": int(months.loc[months["monthly_median"].idxmax(), "month"]) if complete else np.nan,
            "driest_month": int(months.loc[months["monthly_median"].idxmin(), "month"]) if complete else np.nan,
            "valid_days": int(len(valid)),
            "usable_months": int(len(months)),
        })
    annual = pd.DataFrame(annual_rows, columns=ANNUAL_COLUMNS)
    monthly = (
        pd.concat(monthly_frames, ignore_index=True)
        if monthly_frames else pd.DataFrame(columns=MONTHLY_COLUMNS)
    )
    references = {
        "q05_percentile_low_flow_fdc_q95": float(valid_record.quantile(0.05)),
        "q95_percentile_high_flow_fdc_q05": float(valid_record.quantile(0.95)),
        "start_date": valid_record.index.min(),
        "end_date": valid_record.index.max(),
        "raw_missing_count": int(series.isna().sum()),
        "imputation_used": False,
        "extrema_screening_used": bool(len(excluded)),
        "extrema_excluded_candidate_count": int(
            valid_record.index.isin(excluded).sum()
        ),
    }
    return annual, monthly, references


def build_seasonality_profile(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame(columns=["month", "median", "year_count"])
    return (
        monthly.groupby("month")["monthly_median"]
        .agg(median="median", year_count="count")
        .reindex(range(1, 13)).rename_axis("month").reset_index()
    )


def build_diagnostic_summary(annual: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specifications = [
        (annual, "flashiness_index", "Median annual Richards-Baker flashiness"),
        (annual, "baseflow_index", "Median annual Lyne-Hollick baseflow index"),
        (annual, "seasonality_index", "Median annual Walsh-Lawler seasonality index"),
        (annual, "lag1_autocorrelation", "Median annual lag-1 autocorrelation"),
        (events, "time_to_peak_days", "Median event time to peak (days)"),
        (events, "rising_slope", "Median event rising slope"),
        (events, "recession_slope", "Median event recession slope"),
        (events, "event_duration_days", "Median event duration (days)"),
        (events, "peak_flow", "Median event peak flow"),
    ]
    for frame, column, label in specifications:
        values = frame[column].dropna() if not frame.empty and column in frame else pd.Series(dtype=float)
        rows.append({
            "diagnostic": label,
            "median": values.median() if not values.empty else np.nan,
            "p10": values.quantile(0.10) if not values.empty else np.nan,
            "p90": values.quantile(0.90) if not values.empty else np.nan,
            "sample_count": int(len(values)),
        })
    return pd.DataFrame(rows)
