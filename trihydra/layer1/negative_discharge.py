"""Detect discharge values below a configurable numerical tolerance."""

import pandas as pd

from trihydra.layer1.timeseries_validity import get_valid_record, timestamps_to_strings


def check_negative_discharge(
    series: pd.Series,
    tolerance: float = 0.001,
) -> dict:
    """Report values below ``-tolerance`` without modifying the series."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    s = get_valid_record(series)

    if s.empty:
        return {
            "check": "negative_discharge",
            "flag": False,
            "value": None,
            "execution_status": "skipped",
            "finding_status": "not_assessed",
            "maximum_negative_magnitude": None,
            "negative_observations": [],
            "flagged_timestamps": [],
            "message": "No valid data found.",
        }

    negative_mask = s < -float(tolerance)
    negative_count = int(negative_mask.sum())
    flagged_values = s.loc[negative_mask]
    maximum_magnitude = (
        abs(float(flagged_values.min())) if not flagged_values.empty else 0.0
    )
    observations = [
        {"timestamp": str(timestamp), "value": float(value)}
        for timestamp, value in flagged_values.items()
    ]

    return {
        "check": "negative_discharge",
        "flag": negative_count > 0,
        "value": negative_count,
        "execution_status": "completed",
        "finding_status": (
            "candidate_detected" if negative_count else "passed"
        ),
        "maximum_negative_magnitude": maximum_magnitude,
        "negative_observations": observations,
        "flagged_timestamps": timestamps_to_strings(s.index[negative_mask]),
        "message": (
            f"Negative discharge count = {negative_count} below "
            f"-{float(tolerance):g} source units."
        ),
        "tolerance": float(tolerance),
    }
