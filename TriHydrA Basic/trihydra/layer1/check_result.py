"""Build consistent result dictionaries for behavioural Layer 1 checks."""


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
    """Combine the shared result fields with check-specific details."""
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
        "flagged_timestamps": (
            [] if flagged_timestamps is None else list(flagged_timestamps)
        ),
        "message": message,
    }
    result.update(details)
    return result
