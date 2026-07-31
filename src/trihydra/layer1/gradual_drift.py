"""Segment-aware gradual-drift analysis for Layer 1.

Long gaps are never interpolated across. Monthly values are split into
continuous analyzable segments after interpolation of at most two missing
months. Every sufficiently long segment receives its own Seasonal
Mann–Kendall test, Sen slope, effect estimate, STL trend, and plotting series.
A skipped calculation uses ``None`` rather than the misleading numeric zero.
"""

import numpy as np
import pandas as pd
import pymannkendall as mk
from statsmodels.tsa.seasonal import STL

from src.trihydra.layer1.check_result import make_result
from src.trihydra.layer1.timeseries_validity import get_valid_record


def _monthly_segments(monthly: pd.Series) -> tuple[list[pd.Series], list[dict]]:
    """Split a monthly series at every unresolved run of missing months."""
    valid = monthly.notna()
    groups = (valid != valid.shift(fill_value=False)).cumsum()
    segments, gaps = [], []
    for _, mask in valid.groupby(groups):
        index = mask.index
        if bool(mask.iloc[0]):
            segments.append(monthly.loc[index])
        else:
            gaps.append({
                "start": str(index[0]),
                "end": str(index[-1]),
                "missing_month_count": int(len(index)),
            })
    return segments, gaps


def _analyse_segment(
    segment: pd.Series,
    significance_level: float,
    min_record_years: float,
) -> dict:
    """Calculate trend evidence and plot-ready values for one segment."""
    record_months = len(segment)
    record_years = record_months / 12
    base = {
        "start": str(segment.index[0]),
        "end": str(segment.index[-1]),
        "month_count": int(record_months),
        "record_years": round(record_years, 2),
        "monthly_dates": [str(x) for x in segment.index],
        "monthly_values": segment.astype(float).tolist(),
    }
    if record_years < min_record_years or record_months < 24:
        return {
            **base,
            "execution_status": "skipped",
            "reason_skipped": (
                f"Segment has {record_years:.2f} years; "
                f"{min_record_years:.2f} required."
            ),
            "trend_values": [],
        }

    result = mk.seasonal_test(
        segment.to_numpy(dtype=float),
        period=12,
        alpha=significance_level,
    )
    slope = float(result.slope)
    total_change = slope * max(record_years - 1, 0)
    reference = float(segment.median())
    relative_change = (
        total_change / reference * 100 if reference != 0 else np.nan
    )
    magnitude_threshold = 2.0 if record_years <= 5 else 5.0 if record_years <= 10 else 8.0
    meaningful = bool(
        np.isfinite(relative_change)
        and abs(relative_change) >= magnitude_threshold
    )
    significant = bool(result.h)
    direction_estimate = (
        "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
    )
    interpretation = (
        f"supported {direction_estimate} trend"
        if significant
        else f"inconclusive {direction_estimate} estimate"
    )

    stl = STL(segment, period=12, robust=True).fit()
    trend = pd.Series(stl.trend, index=segment.index)
    residual = pd.Series(stl.resid, index=segment.index)
    seasonal = pd.Series(stl.seasonal, index=segment.index)
    seasonal_denominator = np.var(residual + seasonal)
    trend_denominator = np.var(residual + trend)
    elapsed_years = np.arange(record_months, dtype=float) / 12
    centred_years = elapsed_years - np.median(elapsed_years)
    sen_line = reference + slope * centred_years

    return {
        **base,
        "execution_status": "completed",
        "reason_skipped": None,
        "direction": result.trend,
        "direction_estimate": direction_estimate,
        "interpretation": interpretation,
        "p_value": float(result.p),
        "z_statistic": float(result.z),
        "kendall_tau": float(result.Tau),
        "significant": significant,
        "sen_slope_per_year": slope,
        "estimated_total_change": float(total_change),
        "reference_flow": reference,
        "estimated_relative_change_percent": (
            float(relative_change) if np.isfinite(relative_change) else None
        ),
        "magnitude_threshold_percent": magnitude_threshold,
        "meaningful_magnitude": meaningful,
        "drift_detected": bool(significant and meaningful),
        "seasonal_strength": (
            float(max(0.0, 1 - np.var(residual) / seasonal_denominator))
            if seasonal_denominator > 0 else None
        ),
        "trend_strength": (
            float(max(0.0, 1 - np.var(residual) / trend_denominator))
            if trend_denominator > 0 else None
        ),
        "trend_dates": [str(x) for x in trend.index],
        "trend_values": trend.astype(float).tolist(),
        "sen_line_dates": [str(x) for x in segment.index],
        "sen_line_values": sen_line.astype(float).tolist(),
    }


def check_gradual_drift(
    series: pd.Series,
    series_type: str = "unknown",
    significance_level: float = 0.05,
    min_daily_values_per_month: int = 20,
    min_record_years: float = 3.0,
    max_interpolated_gap_months: int = 2,
) -> dict:
    """Assess gradual drift separately within continuous record periods.

    Daily values are never altered. Duplicate timestamps are reduced only in
    the private monthly-analysis copy. Months require the configured number of
    valid days. At most ``max_interpolated_gap_months`` consecutive missing
    months are interpolated for the trend calculation; unresolved longer gaps
    divide the record into independent segments and are exposed in the result.
    """
    record = get_valid_record(series)
    if record.empty:
        return make_result(
            check="gradual_drift",
            flag=False,
            value=None,
            series_type=series_type,
            status="skipped",
            reason_skipped="No valid data found.",
            message="Gradual drift not calculated: no valid data found.",
            drift_segments=[],
            unresolved_gaps=[],
        )

    daily = record.astype(float).groupby(record.index).median().sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar)
    monthly = daily.resample("MS").median()
    valid_counts = daily.resample("MS").count()
    monthly = monthly.where(valid_counts >= min_daily_values_per_month)
    monthly = monthly.loc[monthly.first_valid_index():monthly.last_valid_index()]
    short_filled = monthly.interpolate(
        method="time",
        limit=max_interpolated_gap_months,
        limit_area="inside",
    )
    segments, unresolved_gaps = _monthly_segments(short_filled)
    analysed = [
        _analyse_segment(segment, significance_level, min_record_years)
        for segment in segments
    ]
    completed = [x for x in analysed if x["execution_status"] == "completed"]

    if not completed:
        reason = "No continuous record period is long enough for trend analysis."
        return make_result(
            check="gradual_drift",
            flag=False,
            value=None,
            series_type=series_type,
            status="skipped",
            reason_skipped=reason,
            message=f"Gradual drift not calculated: {reason}",
            drift_segments=analysed,
            unresolved_gaps=unresolved_gaps,
        )

    detected = [x for x in completed if x["drift_detected"]]
    observed = series_type.casefold() in {"obs", "observed", "observation"}
    flag = bool(detected and not observed)
    status = "descriptor" if detected and observed else "soft_flag" if flag else "completed"
    changes = [
        abs(x["estimated_relative_change_percent"])
        for x in detected
        if x["estimated_relative_change_percent"] is not None
    ]
    principal_value = max(changes) if changes else 0.0
    finding = (
        "descriptor" if detected and observed
        else "candidate_detected" if detected
        else "passed"
    )

    return make_result(
        check="gradual_drift",
        flag=flag,
        value=round(principal_value, 3),
        series_type=series_type,
        status=status,
        finding_status=finding,
        message=(
            f"Analysed {len(completed)} continuous record period(s); "
            f"{len(detected)} showed statistically significant drift with "
            f"meaningful magnitude. Unresolved long monthly gaps = "
            f"{len(unresolved_gaps)}. Trends are not fitted across those gaps."
        ),
        drift_segments=analysed,
        unresolved_gaps=unresolved_gaps,
        missing_months_before_short_fill=int(monthly.isna().sum()),
    )
