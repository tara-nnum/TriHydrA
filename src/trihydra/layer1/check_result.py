"""Shared construction of behavioural-check result dictionaries.

This helper performs no analysis. It normalises the Boolean flag, attaches the
standard group and completion status, serialises timestamps through
``timeseries_validity.py``, merges check-specific diagnostics, and returns a
fresh dictionary without mutating the supplied details.
"""

from src.trihydra.layer1.timeseries_validity import timestamps_to_strings


def make_result(
    check: str,
    flag: bool,
    value,
    flagged_timestamps=None,
    series_type: str = "unknown",
    message: str = "",
    status: str = "completed",
    finding_status: str | None = None,
    reason_skipped: str | None = None,
    **details,
) -> dict:
    """Build the standard result while keeping legacy fields compatible."""
    if finding_status is None:
        if status == "skipped":
            finding_status = "not_assessed"
        elif status == "descriptor":
            finding_status = "descriptor"
        else:
            finding_status = "candidate_detected" if flag else "passed"
    result = {
        "check": check,
        "check_group": "behavioural",
        "series_type": series_type,
        "status": status,
        "execution_status": "skipped" if status == "skipped" else "completed",
        "finding_status": finding_status,
        "reason_skipped": reason_skipped,
        "flag": bool(flag),
        "value": value,
        "flagged_timestamps": flagged_timestamps or [],
        "message": message,
    }
    result.update(details)
    return result
