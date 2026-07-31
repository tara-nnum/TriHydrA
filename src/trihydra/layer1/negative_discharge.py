"""Negative-discharge check for TriHydrA Layer 1.

Purpose
-------
Detects physically invalid negative streamflow while configurable decimal rounding prevents tiny floating-point artefacts from being treated as real negative flow.

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
This module owns only ``check_negative_discharge``. Defaults remain on the function to preserve
current behaviour. The orchestrator owns execution order and can later receive
``config.py`` integration. Empty or insufficient records produce explicit
structured outcomes. This module writes no files and creates no plots. The
function docstring and inline comments document its detailed statistical steps.
"""

import pandas as pd

from src.trihydra.layer1.timeseries_validity import get_valid_record, timestamps_to_strings


def check_negative_discharge(series: pd.Series, decimals: int = 3) -> dict:
    """
    Check whether meaningful negative discharge values exist.
    Tiny negative numerical artefacts are ignored after rounding.
    """
    s = get_valid_record(series)

    if s.empty:
        return {
            "check": "negative_discharge",
            "flag": False,
            "value": 0,
            "flagged_timestamps": [],
            "message": "No valid data found.",
        }

    rounded = s.round(decimals)
    negative_mask = rounded < 0
    negative_count = int(negative_mask.sum())

    return {
        "check": "negative_discharge",
        "flag": negative_count > 0,
        "value": negative_count,
        "flagged_timestamps": timestamps_to_strings(s.index[negative_mask]),
        "message": f"Negative discharge count = {negative_count}.",
    }
