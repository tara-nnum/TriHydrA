"""Detect non-daily spacing and backward timestamp order in the raw record."""

import pandas as pd

from trihydra.layer1.timeseries_validity import timestamps_to_strings


def check_timestep_consistency(series: pd.Series) -> dict:
    """Check the first-to-last-valid record against a daily time axis."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    valid_mask = s.notna()
    if valid_mask.any():
        record_start = s.index[valid_mask].min()
        record_end = s.index[valid_mask].max()
        s = s.loc[(s.index >= record_start) & (s.index <= record_end)]
    else:
        s = s.iloc[0:0]
    out_of_order_transitions = [
        {
            "previous_timestamp": str(s.index[position - 1]),
            "timestamp": str(s.index[position]),
            "interval_days": float(
                (s.index[position] - s.index[position - 1])
                / pd.Timedelta(days=1)
            ),
        }
        for position in range(1, len(s.index))
        if s.index[position] < s.index[position - 1]
    ]
    out_of_order = bool(out_of_order_transitions)

    # Duplicate dates are assessed independently. Spacing therefore uses a
    # sorted unique time axis so one defect is not scored twice.
    unique_index = pd.DatetimeIndex(s.index.unique()).sort_values()

    if len(unique_index) < 2:
        return {
            "check": "timestep_consistency",
            "flag": False,
            "value": None,
            "execution_status": "skipped",
            "finding_status": "not_assessed",
            "out_of_order": out_of_order,
            "out_of_order_transitions": out_of_order_transitions,
            "irregular_spacing_count": 0,
            "irregular_intervals": [],
            "flagged_timestamps": [],
            "message": "Not enough timestamps to check timestep consistency.",
        }

    diffs = unique_index.to_series().diff().dropna()
    expected = pd.Timedelta(days=1)

    irregular_diffs = diffs[diffs != expected]
    spacing_count = int(len(irregular_diffs))
    issue_count = spacing_count + int(out_of_order)
    intervals = [
        {
            "previous_timestamp": str(timestamp - difference),
            "timestamp": str(timestamp),
            "interval_days": float(difference / pd.Timedelta(days=1)),
        }
        for timestamp, difference in irregular_diffs.items()
    ]
    flagged = pd.DatetimeIndex([
        *irregular_diffs.index,
        *(pd.Timestamp(item["timestamp"]) for item in out_of_order_transitions),
    ]).unique().sort_values()

    return {
        "check": "timestep_consistency",
        "flag": issue_count > 0,
        "value": issue_count,
        "execution_status": "completed",
        "finding_status": "candidate_detected" if issue_count else "passed",
        "out_of_order": out_of_order,
        "out_of_order_transitions": out_of_order_transitions,
        "irregular_spacing_count": spacing_count,
        "irregular_intervals": intervals,
        "flagged_timestamps": timestamps_to_strings(flagged),
        "message": (
            f"Irregular unique-date intervals = {spacing_count}; source time "
            f"axis out of order = {out_of_order}. Duplicate dates "
            "are assessed separately."
        ),
    }
