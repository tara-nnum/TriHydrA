import pandas as pd


def to_datetime_sorted(series: pd.Series) -> pd.Series:
    """Return a copy with datetime index sorted."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def get_valid_record(series: pd.Series) -> pd.Series:
    """
    Keep only the period between first and last valid value.
    Leading/trailing NaNs are ignored.
    """
    s = to_datetime_sorted(series)

    first_valid = s.first_valid_index()
    last_valid = s.last_valid_index()

    if first_valid is None or last_valid is None:
        return s.iloc[0:0]

    return s.loc[first_valid:last_valid]


def timestamps_to_strings(index) -> list:
    """Convert timestamps to readable strings."""
    return [str(x) for x in index]


def missing_intervals(series: pd.Series, internal_only: bool = True) -> list[dict]:
    """Describe consecutive runs of missing observations without filling them.

    Parameters
    ----------
    series:
        Source series. A datetime-sorted copy is inspected; values are not
        changed.
    internal_only:
        When true, inspect only the first-to-last-valid record. Otherwise
        include leading and trailing missing runs from the supplied time axis.

    Returns
    -------
    list of dict
        Each dictionary contains the first and last timestamp, number of
        missing observations, calendar duration (inclusive for daily data), and
        whether the interval is leading, internal, or trailing.
    """
    full = to_datetime_sorted(series)
    if full.empty:
        return []

    first_valid = full.first_valid_index()
    last_valid = full.last_valid_index()
    if first_valid is None or last_valid is None:
        if internal_only:
            return []
        return [{
            "start": str(full.index[0]),
            "end": str(full.index[-1]),
            "missing_count": int(len(full)),
            "calendar_duration_days": int(
                (full.index[-1].normalize() - full.index[0].normalize()).days + 1
            ),
            "position": "all",
        }]

    inspected = full.loc[first_valid:last_valid] if internal_only else full
    mask = inspected.isna()
    if not mask.any():
        return []

    groups = (mask != mask.shift(fill_value=False)).cumsum()
    intervals = []
    for _, group_mask in mask.groupby(groups):
        if not bool(group_mask.iloc[0]):
            continue
        timestamps = group_mask.index
        start, end = timestamps[0], timestamps[-1]
        if end < first_valid:
            position = "leading"
        elif start > last_valid:
            position = "trailing"
        else:
            position = "internal"
        intervals.append({
            "start": str(start),
            "end": str(end),
            "missing_count": int(len(timestamps)),
            "calendar_duration_days": int(
                (end.normalize() - start.normalize()).days + 1
            ),
            "position": position,
        })
    return intervals


def describe_missingness(series: pd.Series) -> dict:
    """Return transparent total, leading, internal, and trailing NaN counts."""
    full = to_datetime_sorted(series)
    total = int(full.isna().sum())
    first_valid = full.first_valid_index()
    last_valid = full.last_valid_index()

    if first_valid is None or last_valid is None:
        return {
            "total_nan_count": total,
            "leading_nan_count": total,
            "internal_nan_count": 0,
            "trailing_nan_count": 0,
            "record_start": None,
            "record_end": None,
            "internal_intervals": [],
            "all_intervals": missing_intervals(full, internal_only=False),
        }

    leading = int(full.loc[:first_valid].iloc[:-1].isna().sum())
    trailing = int(full.loc[last_valid:].iloc[1:].isna().sum())
    valid_record = full.loc[first_valid:last_valid]
    internal = int(valid_record.isna().sum())
    return {
        "total_nan_count": total,
        "leading_nan_count": leading,
        "internal_nan_count": internal,
        "trailing_nan_count": trailing,
        "record_start": str(first_valid),
        "record_end": str(last_valid),
        "internal_intervals": missing_intervals(full, internal_only=True),
        "all_intervals": missing_intervals(full, internal_only=False),
    }
