"""Detect consecutive internal missing observations in daily discharge."""

import pandas as pd

from trihydra.layer1.timeseries_validity import (
    get_valid_record,
    missing_intervals,
    timestamps_to_strings,
)


def check_long_gaps(series: pd.Series, minimum_reported_gap_days: int = 3) -> dict:
    """Report internal NaN runs longer than the evidence threshold.

    Consecutive rows represent days under TriHydrA's daily input contract.
    Absent calendar timestamps are assessed by ``timestep_consistency.py``.
    """
    if minimum_reported_gap_days < 0:
        raise ValueError("minimum_reported_gap_days must be non-negative.")

    valid_record = get_valid_record(series)

    if valid_record.empty:
        return {
            "check": "long_gaps",
            "flag": False,
            "value": None,
            "execution_status": "skipped",
            "finding_status": "not_assessed",
            "flagged_timestamps": [],
            "all_missing_intervals": [],
            "long_gap_intervals": [],
            "message": "No valid data found.",
        }

    all_gaps = missing_intervals(valid_record, internal_only=True)
    long_gaps = [
        gap
        for gap in all_gaps
        if gap["missing_count"] > minimum_reported_gap_days
    ]
    long_gap_mask = pd.Series(False, index=valid_record.index)
    for gap in long_gaps:
        long_gap_mask.loc[gap["start"]:gap["end"]] = True
    flagged_timestamps = valid_record.index[
        valid_record.isna() & long_gap_mask
    ]
    longest_gap = max((gap["missing_count"] for gap in all_gaps), default=0)

    return {
        "check": "long_gaps",
        "flag": bool(long_gaps),
        "value": longest_gap,
        "execution_status": "completed",
        "finding_status": "candidate_detected" if long_gaps else "passed",
        "minimum_reported_gap_days": int(minimum_reported_gap_days),
        "flagged_timestamps": timestamps_to_strings(flagged_timestamps),
        "all_missing_intervals": all_gaps,
        "long_gap_intervals": long_gaps,
        "message": (
            f"Long gaps = {len(long_gaps)}; longest internal missing gap = "
            f"{longest_gap} observation(s); evidence threshold = "
            f">{minimum_reported_gap_days}. Composite concern thresholds are "
            "configured separately."
        ),
    }
