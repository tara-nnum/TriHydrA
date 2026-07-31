"""Duplicate-timestamp check for TriHydrA Layer 1.

Purpose
-------
Detects every occurrence of repeated timestamps after safe datetime conversion and sorting; all members of a duplicate group are reported.

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
This module owns only ``check_duplicate_timestamps``. Defaults remain on the function to preserve
current behaviour. The orchestrator owns execution order and can later receive
``config.py`` integration. Empty or insufficient records produce explicit
structured outcomes. This module writes no files and creates no plots. The
function docstring and inline comments document its detailed statistical steps.
"""

import pandas as pd

from src.trihydra.layer1.timeseries_validity import timestamps_to_strings, to_datetime_sorted


def check_duplicate_timestamps(series: pd.Series) -> dict:
    """
    Check whether the time index contains duplicate timestamps.
    """
    s = to_datetime_sorted(series)

    duplicate_mask = s.index.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())

    return {
        "check": "duplicate_timestamps",
        "flag": duplicate_count > 0,
        "value": duplicate_count,
        "flagged_timestamps": timestamps_to_strings(s.index[duplicate_mask]),
        "message": f"Duplicate timestamp count = {duplicate_count}.",
    }
