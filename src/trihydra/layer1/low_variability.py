"""Low-variability check for TriHydrA Layer 1.

Purpose
-------
Finds unusually persistent, nearly constant discharge from the rolling variability profile in behaviour_profile.py using seasonal thresholds and persistence logic.

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
This module owns only ``check_low_variability``. Defaults remain on the function to preserve
current behaviour. The orchestrator owns execution order and can later receive
``config.py`` integration. Empty or insufficient records produce explicit
structured outcomes. This module writes no files and creates no plots. The
function docstring and inline comments document its detailed statistical steps.
"""

import pandas as pd

from src.trihydra.layer1.behaviour_profile import calculate_rolling_variability_series
from src.trihydra.layer1.check_result import make_result
from src.trihydra.layer1.timeseries_validity import timestamps_to_strings


def check_low_variability(
    series: pd.Series,
    series_type: str = "unknown",
    window: int = 21,
    lower_quantile: float = 0.01,
    spell_quantile: float = 0.99,
    decimals: int = 3,
) -> dict:
    """
    Detect unusually persistent low non-zero variability.

    Uses rolling variability metrics from behaviour_profile.py.
    Low variability thresholds are learned month-wise from the same series.
    """
    rolling = calculate_rolling_variability_series(
        series=series,
        window=window,
        decimals=decimals,
    )

    if rolling.empty or rolling["rolling_cv"].dropna().empty:
        return make_result(
            check="low_variability",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="Not enough valid rolling variability data.",
        )

    low_var_mask = pd.Series(False, index=rolling.index)

    for month in range(1, 13):
        month_mask = rolling["month"] == month

        valid_month_mask = month_mask & ~rolling["is_zero_flow_window"]

        month_cv = rolling.loc[valid_month_mask, "rolling_cv"].dropna()
        month_range = rolling.loc[valid_month_mask, "rolling_range"].dropna()

        if month_cv.empty or month_range.empty:
            continue

        cv_threshold = month_cv.quantile(lower_quantile)
        range_threshold = month_range.quantile(lower_quantile)

        low_var_mask.loc[month_mask] = (
                (rolling.loc[month_mask, "rolling_cv"] <= cv_threshold)
                & (rolling.loc[month_mask, "rolling_range"] <= range_threshold)
                & ~rolling.loc[month_mask, "is_zero_flow_window"]
        ).fillna(False)

    spell_id = (low_var_mask != low_var_mask.shift()).cumsum()
    spell_lengths = low_var_mask.groupby(spell_id).sum()
    low_var_spell_lengths = spell_lengths[spell_lengths > 0].astype(int)

    if low_var_spell_lengths.empty:
        return make_result(
            check="low_variability",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="completed",
            message="No low-variability spells detected.",
        )

    spell_length_threshold = low_var_spell_lengths.quantile(spell_quantile)

    flagged_timestamps = []
    flagged_periods = []

    for group, length in spell_lengths.items():
        if length > spell_length_threshold:
            group_mask = spell_id == group
            dates = rolling.index[group_mask & low_var_mask]
            flagged_timestamps.extend(dates)
            flagged_periods.append({
                "start": str(dates[0]),
                "end": str(dates[-1]),
                "observation_count": int(length),
                "calendar_duration_days": int((dates[-1] - dates[0]).days + 1),
                "mean_rolling_cv": float(
                    rolling.loc[dates, "rolling_cv"].mean()
                ),
                "mean_rolling_range": float(
                    rolling.loc[dates, "rolling_range"].mean()
                ),
                "minimum_rolling_cv": float(
                    rolling.loc[dates, "rolling_cv"].min()
                ),
                "maximum_rolling_range": float(
                    rolling.loc[dates, "rolling_range"].max()
                ),
            })

    return make_result(
        check="low_variability",
        flag=len(flagged_timestamps) > 0,
        value=len(flagged_timestamps),
        flagged_timestamps=timestamps_to_strings(flagged_timestamps),
        series_type=series_type,
        status="completed",
        finding_status=(
            "candidate_detected" if flagged_timestamps else "passed"
        ),
        message=(
            f"Low variability uses {window}-day rolling CV and rolling range; "
            f"month-wise lower quantile = {lower_quantile}; "
            f"spell-length threshold learned from series = "
            f"{spell_length_threshold:.1f} timestep(s); "
            f"flagged timesteps = {len(flagged_timestamps)}."
        ),
        low_variability_periods=flagged_periods,
        spell_length_threshold=float(spell_length_threshold),
        window_days=int(window),
    )
