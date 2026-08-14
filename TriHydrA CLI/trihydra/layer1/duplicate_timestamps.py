"""Detect repeated timestamps without resolving or deleting source rows."""

import pandas as pd

from trihydra.layer1.timeseries_validity import timestamps_to_strings, to_datetime_sorted


def check_duplicate_timestamps(series: pd.Series) -> dict:
    """Report every repeated timestamp group and its source values."""
    s = to_datetime_sorted(series)

    if s.empty:
        return {
            "check": "duplicate_timestamps",
            "flag": False,
            "value": None,
            "execution_status": "skipped",
            "finding_status": "not_assessed",
            "duplicate_group_rows": 0,
            "extra_duplicate_rows": 0,
            "conflicting_duplicate_groups": 0,
            "duplicate_groups": [],
            "flagged_timestamps": [],
            "message": "No timestamps found.",
        }

    duplicate_mask = s.index.duplicated(keep=False)
    duplicated_index = s.index[duplicate_mask]
    unique_dates = pd.DatetimeIndex(duplicated_index.unique()).sort_values()
    duplicate_group_rows = int(duplicate_mask.sum())
    extra_rows = int(s.index.duplicated(keep="first").sum())
    duplicate_count = int(len(unique_dates))
    groups = []
    for timestamp in unique_dates:
        values = s.loc[s.index == timestamp]
        groups.append({
            "timestamp": str(timestamp),
            "occurrences": int(len(values)),
            "values": [
                None if pd.isna(value) else float(value)
                for value in values.to_numpy()
            ],
            "conflicting_values": bool(values.nunique(dropna=False) > 1),
        })
    conflict_count = sum(group["conflicting_values"] for group in groups)

    return {
        "check": "duplicate_timestamps",
        "flag": duplicate_count > 0,
        "value": duplicate_count,
        "execution_status": "completed",
        "finding_status": (
            "candidate_detected" if duplicate_count else "passed"
        ),
        "duplicate_group_rows": duplicate_group_rows,
        "extra_duplicate_rows": extra_rows,
        "conflicting_duplicate_groups": int(conflict_count),
        "duplicate_groups": groups,
        "flagged_timestamps": timestamps_to_strings(unique_dates),
        "message": (
            f"Unique duplicated dates = {duplicate_count}; rows in duplicate "
            f"groups = {duplicate_group_rows}; extra rows = {extra_rows}; "
            f"conflicting groups = {conflict_count}."
        ),
    }
