"""Long-gap check for TriHydrA Layer 1.

Purpose
-------
Groups consecutive internal missing observations and flags every timestamp in a run longer than the permitted number of daily timesteps. It reports the longest run even if nothing is flagged.

Data contract
-------------
The public function accepts a pandas Series with observation timestamps as its
index and discharge as its values. Shared record preparation is delegated to
``timeseries_validity.py``; behavioural reference quantities are delegated to
``behaviour_profile.py`` where applicable. Source observations are never
silently replaced, deleted, or permanently modified.

Result contract
---------------
The function returns the standard Layer 1 dictionary. ``check`` is the stable
machine name; ``flag`` is the overall finding; ``value`` is the principal
scalar; ``flagged_timestamps`` is serialisable evidence; and ``message``
explains the outcome. Check-specific diagnostics are retained for plots and
summary tables.

Configuration, edge cases, and side effects
-------------------------------------------
This module owns only ``check_long_gaps``. Defaults remain on the function to preserve
current behaviour. The orchestrator owns execution order and can later receive
``config.py`` integration. Empty or insufficient records produce explicit
structured outcomes. This module writes no files and creates no plots. The
function docstring and inline comments document its detailed statistical steps.
"""

import pandas as pd

from src.trihydra.layer1.timeseries_validity import (
    get_valid_record,
    missing_intervals,
    timestamps_to_strings,
)


def check_long_gaps(series: pd.Series, max_gap_days: int = 3) -> dict:
    """
    Flag internal missing gaps longer than max_gap_days.
    Assumes daily data, so consecutive missing values are counted as missing days.
    """
    valid_record = get_valid_record(series)

    if valid_record.empty:
        return {
            "check": "long_gaps",
            "flag": False,
            "value": 0,
            "flagged_timestamps": [],
            "message": "No valid data found.",
        }

    missing_mask = valid_record.isna()
    all_gaps = missing_intervals(valid_record, internal_only=True)
    long_gaps = [
        {**gap, "threshold_days": int(max_gap_days)}
        for gap in all_gaps
        if gap["missing_count"] > max_gap_days
    ]
    flagged_timestamps = valid_record.index[
        missing_mask
        & pd.Series(
            [
                any(
                    pd.Timestamp(gap["start"]) <= timestamp <= pd.Timestamp(gap["end"])
                    for gap in long_gaps
                )
                for timestamp in valid_record.index
            ],
            index=valid_record.index,
        )
    ]
    longest_gap = max(
        (gap["missing_count"] for gap in all_gaps),
        default=0,
    )

    return {
        "check": "long_gaps",
        "flag": bool(long_gaps),
        "value": longest_gap,
        "execution_status": "completed",
        "finding_status": "candidate_detected" if long_gaps else "passed",
        "threshold": f">{max_gap_days} consecutive missing days",
        "flagged_timestamps": timestamps_to_strings(flagged_timestamps),
        "all_missing_intervals": all_gaps,
        "long_gap_intervals": long_gaps,
        "message": (
            f"Long gaps = {len(long_gaps)}; longest internal missing gap = "
            f"{longest_gap} observation(s); threshold = >{max_gap_days}."
        ),
    }
