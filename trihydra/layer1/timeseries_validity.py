"""Shared, non-mutating preparation of Layer 1 time series."""

import numpy as np
import pandas as pd


def to_datetime_sorted(series: pd.Series) -> pd.Series:
    """Return a datetime-indexed, chronologically sorted copy."""
    result = series.copy()
    result.index = pd.to_datetime(result.index)
    return result.sort_index(kind="stable")


def _valid_bounds(series: pd.Series) -> tuple[int, int] | None:
    """Return positional bounds of the first-to-last valid observation."""
    valid_positions = np.flatnonzero(series.notna().to_numpy())
    if not len(valid_positions):
        return None
    return int(valid_positions[0]), int(valid_positions[-1])


def get_valid_record(series: pd.Series) -> pd.Series:
    """Exclude leading and trailing missing rows from a sorted copy."""
    result = to_datetime_sorted(series)
    bounds = _valid_bounds(result)
    if bounds is None:
        return result.iloc[0:0]
    first, last = bounds
    return result.iloc[first : last + 1]


def timestamps_to_strings(index) -> list[str]:
    """Convert timestamps to strings used by result tables and files."""
    return [str(timestamp) for timestamp in index]


def _missing_intervals_from_sorted(
    series: pd.Series,
    internal_only: bool,
) -> list[dict]:
    """Describe missing runs in an already sorted series."""
    if series.empty:
        return []

    bounds = _valid_bounds(series)
    if bounds is None:
        if internal_only:
            return []
        start, end = series.index[0], series.index[-1]
        return [{
            "start": str(start),
            "end": str(end),
            "missing_count": int(len(series)),
            "calendar_duration_days": int(
                (end.normalize() - start.normalize()).days + 1
            ),
            "position": "all",
        }]

    first_valid, last_valid = bounds
    first = first_valid if internal_only else 0
    last = last_valid if internal_only else len(series) - 1
    missing = series.isna().to_numpy()
    intervals = []
    position = first

    while position <= last:
        if not missing[position]:
            position += 1
            continue
        run_start = position
        while position <= last and missing[position]:
            position += 1
        run_end = position - 1
        start, end = series.index[run_start], series.index[run_end]
        location = (
            "leading" if run_end < first_valid
            else "trailing" if run_start > last_valid
            else "internal"
        )
        intervals.append({
            "start": str(start),
            "end": str(end),
            "missing_count": int(run_end - run_start + 1),
            "calendar_duration_days": int(
                (end.normalize() - start.normalize()).days + 1
            ),
            "position": location,
        })
    return intervals


def missing_intervals(series: pd.Series, internal_only: bool = True) -> list[dict]:
    """Describe consecutive missing rows without filling source values."""
    return _missing_intervals_from_sorted(
        to_datetime_sorted(series),
        internal_only=internal_only,
    )


def describe_missingness(series: pd.Series) -> dict:
    """Count leading, internal, and trailing missing observations."""
    result = to_datetime_sorted(series)
    total = int(result.isna().sum())
    bounds = _valid_bounds(result)

    if bounds is None:
        return {
            "total_nan_count": total,
            "leading_nan_count": total,
            "internal_nan_count": 0,
            "trailing_nan_count": 0,
            "record_start": None,
            "record_end": None,
            "internal_intervals": [],
            "all_intervals": _missing_intervals_from_sorted(result, False),
        }

    first, last = bounds
    valid_record = result.iloc[first:last + 1]
    return {
        "total_nan_count": total,
        "leading_nan_count": int(result.iloc[:first].isna().sum()),
        "internal_nan_count": int(valid_record.isna().sum()),
        "trailing_nan_count": int(result.iloc[last + 1:].isna().sum()),
        "record_start": str(result.index[first]),
        "record_end": str(result.index[last]),
        "internal_intervals": _missing_intervals_from_sorted(result, True),
        "all_intervals": _missing_intervals_from_sorted(result, False),
    }
