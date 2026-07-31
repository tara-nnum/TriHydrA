"""Missing-value check for TriHydrA Layer 1.

Purpose
-------
Locates missing observations inside the valid record bounded by the first and last non-missing measurements. Leading and trailing padding are deliberately excluded because they are outside the observed record.

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
This module owns only ``check_missing_values``. Defaults remain on the function to preserve
current behaviour. The orchestrator owns execution order and can later receive
``config.py`` integration. Empty or insufficient records produce explicit
structured outcomes. This module writes no files and creates no plots. The
function docstring and inline comments document its detailed statistical steps.
"""

import pandas as pd

from src.trihydra.layer1.timeseries_validity import (
    describe_missingness,
    get_valid_record,
    timestamps_to_strings,
)


def check_missing_values(series: pd.Series) -> dict:
    """
    Flag missing values only inside the valid record period.
    For daily data, missing value count equals missing day count.
    """
    missingness = describe_missingness(series)
    valid_record = get_valid_record(series)

    if valid_record.empty:
        return {
            "check": "missing_values",
            "flag": True,
            "value": None,
            "execution_status": "skipped",
            "finding_status": "not_assessed",
            "flagged_timestamps": [],
            **missingness,
            "message": "No valid data found.",
        }

    missing_mask = valid_record.isna()
    missing_count = int(missing_mask.sum())

    return {
        "check": "missing_values",
        "flag": missing_count > 0,
        "value": missing_count,
        "execution_status": "completed",
        "finding_status": "candidate_detected" if missing_count else "passed",
        "record_start": str(valid_record.index.min()),
        "record_end": str(valid_record.index.max()),
        "flagged_timestamps": timestamps_to_strings(valid_record.index[missing_mask]),
        **missingness,
        "message": (
            f"Internal missing observation count = {missing_count}; total NaNs on "
            f"the supplied time axis = {missingness['total_nan_count']} "
            f"(leading = {missingness['leading_nan_count']}, trailing = "
            f"{missingness['trailing_nan_count']})."
        ),
    }
