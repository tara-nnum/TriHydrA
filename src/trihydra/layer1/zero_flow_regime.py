"""Zero-flow-regime check for TriHydrA Layer 1.

Purpose
-------
Measures the presence and frequency of zero discharge. Rounding prevents negligible numerical noise from being mistaken for real flow.

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
This module owns only ``check_zero_flow_regime``. Defaults remain on the function to preserve
current behaviour. The orchestrator owns execution order and can later receive
``config.py`` integration. Empty or insufficient records produce explicit
structured outcomes. This module writes no files and creates no plots. The
function docstring and inline comments document its detailed statistical steps.
"""

import pandas as pd

from src.trihydra.layer1.check_result import make_result
from src.trihydra.layer1.timeseries_validity import get_valid_record


def check_zero_flow_regime(
    series: pd.Series,
    series_type: str = "unknown",
    decimals: int = 3,
) -> dict:
    """
    Describe zero-flow behaviour.

    Zero flow is valid data, not missing data.
    This check reports zero-flow behaviour as a descriptor,
    not as an automatic anomaly flag.
    """
    s = get_valid_record(series)

    if s.empty:
        return make_result(
            check="zero_flow_regime",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="No valid data found.",
        )

    rounded = s.round(decimals)
    zero_mask = rounded == 0

    zero_count = int(zero_mask.sum())
    zero_ratio = float(zero_mask.mean())

    spell_id = (zero_mask != zero_mask.shift()).cumsum()
    spell_lengths = zero_mask.groupby(spell_id).sum()
    zero_spell_lengths = spell_lengths[spell_lengths > 0].astype(int)

    zero_spell_count = int(len(zero_spell_lengths))
    longest_zero_spell = (
        int(zero_spell_lengths.max()) if not zero_spell_lengths.empty else 0
    )

    monthly_zero_ratio = zero_mask.groupby(zero_mask.index.month).mean()
    zero_flow_months = monthly_zero_ratio[monthly_zero_ratio > 0].index.tolist()
    zero_spells = []
    for group, length in spell_lengths.items():
        if int(length) <= 0:
            continue
        dates = zero_mask.index[(spell_id == group) & zero_mask]
        zero_spells.append({
            "start": str(dates[0]),
            "end": str(dates[-1]),
            "observation_count": int(length),
            "calendar_duration_days": int((dates[-1] - dates[0]).days + 1),
        })

    return make_result(
        check="zero_flow_regime",
        flag=False,
        value=zero_ratio,
        flagged_timestamps=[],
        series_type=series_type,
        status="descriptor",
        finding_status="descriptor",
        message=(
            f"Zero-flow ratio = {zero_ratio:.3f}; "
            f"zero-flow count = {zero_count}; "
            f"zero-flow spells = {zero_spell_count}; "
            f"longest zero-flow spell = {longest_zero_spell} day(s); "
            f"zero-flow months = {zero_flow_months}."
        ),
        zero_count=zero_count,
        zero_ratio=zero_ratio,
        zero_spell_count=zero_spell_count,
        longest_zero_spell=longest_zero_spell,
        zero_flow_months=zero_flow_months,
        zero_flow_spells=zero_spells,
    )
