"""Describe the prevalence and persistence of valid zero discharge."""

import pandas as pd

from trihydra.layer1.check_result import make_result
from trihydra.layer1.timeseries_validity import get_valid_record


def check_zero_flow_regime(
    series: pd.Series,
    series_type: str = "unknown",
    decimals: int = 3,
) -> dict:
    """Describe zero flow without treating it as an anomaly flag."""
    if decimals < 0:
        raise ValueError("decimals must be non-negative.")

    s = get_valid_record(series)

    if s.empty:
        return make_result(
            check="zero_flow_regime",
            flag=False,
            value=None,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="No valid data found.",
        )

    numeric = pd.to_numeric(s, errors="coerce")
    valid_mask = numeric.notna()
    rounded = numeric.round(decimals)
    zero_mask = valid_mask & rounded.eq(0)

    zero_count = int(zero_mask.sum())
    valid_count = int(valid_mask.sum())
    zero_ratio = float(zero_count / valid_count) if valid_count else 0.0

    daily_continuity = s.index.to_series().diff().eq(pd.Timedelta(days=1))
    spell_boundary = zero_mask.ne(zero_mask.shift(fill_value=False)) | ~daily_continuity
    spell_id = spell_boundary.cumsum()
    spell_lengths = zero_mask.groupby(spell_id).sum()
    zero_spell_lengths = spell_lengths[spell_lengths > 0].astype(int)

    zero_spell_count = int(len(zero_spell_lengths))
    longest_zero_spell_observations = (
        int(zero_spell_lengths.max()) if not zero_spell_lengths.empty else 0
    )

    monthly_zero_count = zero_mask.groupby(zero_mask.index.month).sum()
    monthly_valid_count = valid_mask.groupby(valid_mask.index.month).sum()
    monthly_zero_ratio = (
        monthly_zero_count / monthly_valid_count.replace(0, pd.NA)
    ).astype(float)
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
    longest_zero_spell_days = max(
        (item["calendar_duration_days"] for item in zero_spells), default=0
    )

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
            f"zero-flow count = {zero_count} of {valid_count} valid observations; "
            f"zero-flow spells = {zero_spell_count}; "
            f"longest zero-flow spell = {longest_zero_spell_days} day(s); "
            f"zero-flow months = {zero_flow_months}."
        ),
        zero_count=zero_count,
        valid_observation_count=valid_count,
        zero_ratio=zero_ratio,
        zero_spell_count=zero_spell_count,
        longest_zero_spell_days=longest_zero_spell_days,
        longest_zero_spell_observations=longest_zero_spell_observations,
        rounding_decimals=int(decimals),
        zero_flow_months=zero_flow_months,
        monthly_zero_ratio={
            int(month): float(value)
            for month, value in monthly_zero_ratio.dropna().items()
        },
        zero_flow_spells=zero_spells,
    )
