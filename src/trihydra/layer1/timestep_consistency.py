"""Timestep-consistency check for TriHydrA Layer 1.

Purpose
-------
Verifies that consecutive timestamps inside the valid record are exactly one day apart. A flagged timestamp is the later endpoint of an irregular interval.

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
This module owns only ``check_timestep_consistency``. Defaults remain on the function to preserve
current behaviour. The orchestrator owns execution order and can later receive
``config.py`` integration. Empty or insufficient records produce explicit
structured outcomes. This module writes no files and creates no plots. The
function docstring and inline comments document its detailed statistical steps.
"""

import pandas as pd

from src.trihydra.layer1.timeseries_validity import get_valid_record, timestamps_to_strings


def check_timestep_consistency(series: pd.Series) -> dict:
    """
    Check whether timestamps are consistently daily inside the valid record.
    """
    s = get_valid_record(series)

    if len(s) < 2:
        return {
            "check": "timestep_consistency",
            "flag": False,
            "value": 0,
            "flagged_timestamps": [],
            "message": "Not enough timestamps to check timestep consistency.",
        }

    diffs = s.index.to_series().diff().dropna()
    expected = pd.Timedelta(days=1)

    irregular_mask = diffs != expected
    irregular_count = int(irregular_mask.sum())

    return {
        "check": "timestep_consistency",
        "flag": irregular_count > 0,
        "value": irregular_count,
        "expected_timestep": "1 day",
        "flagged_timestamps": timestamps_to_strings(diffs.index[irregular_mask]),
        "message": f"Irregular daily timestep count = {irregular_count}.",
    }
