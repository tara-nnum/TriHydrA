"""Detect explicit missing observations inside the observed record."""

import pandas as pd

from trihydra.layer1.timeseries_validity import (
    describe_missingness,
    get_valid_record,
    timestamps_to_strings,
)


def check_missing_values(series: pd.Series) -> dict:
    """Report NaNs between the first and last valid observations.

    Leading/trailing padding is excluded. Missing calendar timestamps are
    assessed separately by ``timestep_consistency.py``.
    """
    missingness = describe_missingness(series)
    valid_record = get_valid_record(series)

    if valid_record.empty:
        return {
            "check": "missing_values",
            "flag": False,
            "value": None,
            "execution_status": "skipped",
            "finding_status": "not_assessed",
            "flagged_timestamps": [],
            "internal_missing_percentage": None,
            **missingness,
            "message": "No valid data found.",
        }

    missing_mask = valid_record.isna()
    missing_count = int(missing_mask.sum())
    missing_percentage = 100.0 * missing_count / len(valid_record)

    return {
        "check": "missing_values",
        "flag": missing_count > 0,
        "value": missing_count,
        "execution_status": "completed",
        "finding_status": "candidate_detected" if missing_count else "passed",
        "internal_missing_percentage": missing_percentage,
        "flagged_timestamps": timestamps_to_strings(
            valid_record.index[missing_mask]
        ),
        **missingness,
        "message": (
            f"Internal missing observation count = {missing_count}; total NaNs on "
            f"the supplied time axis = {missingness['total_nan_count']} "
            f"(leading = {missingness['leading_nan_count']}, trailing = "
            f"{missingness['trailing_nan_count']})."
        ),
    }
