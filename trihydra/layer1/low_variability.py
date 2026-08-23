"""Detect persistent runs of repeated, non-zero discharge values."""

import pandas as pd

from trihydra.layer1.check_result import make_result
from trihydra.layer1.timeseries_validity import (
    get_valid_record,
    timestamps_to_strings,
)


def _repeated_nonzero_runs(record: pd.Series, decimals: int) -> list[dict]:
    """Return consecutive daily runs of one rounded non-zero value."""
    frame = pd.DataFrame({
        "date": record.index,
        "value": pd.to_numeric(record, errors="coerce").round(decimals).to_numpy(),
        "position": range(len(record)),
    })
    previous_value = frame["value"].shift(1)
    previous_date = frame["date"].shift(1)
    continues = (
        frame["value"].notna()
        & previous_value.notna()
        & frame["value"].ne(0)
        & frame["value"].eq(previous_value)
        & frame["date"].sub(previous_date).eq(pd.Timedelta(days=1))
    )
    run_id = (~continues).cumsum()
    runs = []
    for _, group in frame.groupby(run_id):
        values = group["value"]
        if len(group) < 2 or values.isna().any() or float(values.iloc[0]) == 0:
            continue
        dates = pd.DatetimeIndex(group["date"])
        runs.append({
            "start": dates[0],
            "end": dates[-1],
            "value": float(values.iloc[0]),
            "observation_count": int(len(group)),
            "calendar_duration_days": int((dates[-1] - dates[0]).days + 1),
            "dates": dates,
            "start_position": int(group["position"].iloc[0]),
            "end_position": int(group["position"].iloc[-1]),
        })
    return runs


def check_low_variability(
    series: pd.Series,
    series_type: str = "unknown",
    minimum_plateau_days: int = 15,
    decimals: int = 3,
) -> dict:
    """Flag daily non-zero plateaus lasting at least the configured duration."""
    if minimum_plateau_days < 2:
        raise ValueError("minimum_plateau_days must be at least 2")
    if decimals < 0:
        raise ValueError("decimals must be zero or greater")

    record = get_valid_record(series)
    if record.empty:
        return make_result(
            check="low_variability", flag=False, value=None,
            flagged_timestamps=[], series_type=series_type, status="skipped",
            reason_skipped="No valid data found.",
            message="Non-zero plateau check not calculated: no valid data found.",
            plateau_periods=[],
        )

    runs = _repeated_nonzero_runs(record, decimals=decimals)
    threshold = int(minimum_plateau_days)
    periods = []
    flagged_dates = []
    for run in runs:
        if run["observation_count"] < threshold:
            continue
        start_pos = run["start_position"]
        end_pos = run["end_position"]
        before = record.iloc[start_pos - 1] if start_pos > 0 else None
        after = record.iloc[end_pos + 1] if end_pos + 1 < len(record) else None
        periods.append({
            "start": str(run["start"]), "end": str(run["end"]),
            "plateau_value": run["value"],
            "observation_count": run["observation_count"],
            "calendar_duration_days": run["calendar_duration_days"],
            "value_before": float(before) if pd.notna(before) else None,
            "value_after": float(after) if pd.notna(after) else None,
        })
        flagged_dates.extend(run["dates"])

    message = (
        f"Non-zero plateau candidates = {len(periods)}; flagged observations = "
        f"{len(flagged_dates)}; minimum duration = {threshold} days; "
        f"values compared after rounding to {decimals} decimals."
    )
    return make_result(
        check="low_variability", flag=bool(periods), value=len(flagged_dates),
        flagged_timestamps=timestamps_to_strings(flagged_dates),
        series_type=series_type, status="completed",
        finding_status="candidate_detected" if periods else "passed",
        message=message, plateau_periods=periods,
        plateau_count=len(periods),
        longest_plateau_days=max(
            (period["calendar_duration_days"] for period in periods), default=0
        ),
        minimum_plateau_days=threshold,
        rounding_decimals=int(decimals),
    )
