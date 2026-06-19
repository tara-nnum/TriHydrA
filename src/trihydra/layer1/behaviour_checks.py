import numpy as np
import pandas as pd

from src.trihydra.layer1.timeseries_validity import get_valid_record, timestamps_to_strings
from src.trihydra.layer1.behaviour_profile import calculate_rolling_variability_series
from src.trihydra.layer1.behaviour_profile import calculate_local_deviation_series
from src.trihydra.layer1.behaviour_profile import calculate_step_score_series
from src.trihydra.layer1.behaviour_profile import calculate_monthly_by_year_profile


def make_result(
    check: str,
    flag: bool,
    value,
    flagged_timestamps=None,
    series_type: str = "unknown",
    message: str = "",
    status: str = "completed",
) -> dict:
    """Standard output format for behavioural checks."""
    return {
        "check": check,
        "check_group": "behavioural",
        "series_type": series_type,
        "status": status,
        "flag": bool(flag),
        "value": value,
        "flagged_timestamps": flagged_timestamps or [],
        "message": message,
    }

# Zero flow regime check

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

    return make_result(
        check="zero_flow_regime",
        flag=False,
        value=zero_ratio,
        flagged_timestamps=[],
        series_type=series_type,
        status="descriptor",
        message=(
            f"Zero-flow ratio = {zero_ratio:.3f}; "
            f"zero-flow count = {zero_count}; "
            f"zero-flow spells = {zero_spell_count}; "
            f"longest zero-flow spell = {longest_zero_spell} day(s); "
            f"zero-flow months = {zero_flow_months}."
        ),
    )

# Low variability

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

    for group, length in spell_lengths.items():
        if length > spell_length_threshold:
            group_mask = spell_id == group
            flagged_timestamps.extend(rolling.index[group_mask & low_var_mask])

    return make_result(
        check="low_variability",
        flag=len(flagged_timestamps) > 0,
        value=len(flagged_timestamps),
        flagged_timestamps=timestamps_to_strings(flagged_timestamps),
        series_type=series_type,
        status="completed",
        message=(
            f"Low variability uses {window}-day rolling CV and rolling range; "
            f"month-wise lower quantile = {lower_quantile}; "
            f"spell-length threshold learned from series = "
            f"{spell_length_threshold:.1f} timestep(s); "
            f"flagged timesteps = {len(flagged_timestamps)}."
        ),
    )

# Spike and Dip check

def check_spike_dip(
    series: pd.Series,
    series_type: str = "unknown",
    deviation_quantile: float = 0.99,
    neighbour_quantile: float = 0.50,
) -> dict:
    """
    Detect isolated single-point spike/dip candidates.

    Logic:
    - Calculate how far each point is from the average of its neighbours.
    - Learn the extreme local-deviation threshold from the series itself.
    - Flag points with extreme local deviation only when neighbours are relatively consistent.
    """
    local = calculate_local_deviation_series(series)

    if local.empty:
        return make_result(
            check="spike_dip",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="Not enough data for spike/dip detection.",
        )

    deviation = local["local_deviation"].dropna()
    neighbour_difference = local["neighbour_difference"].dropna()

    if deviation.empty or neighbour_difference.empty:
        return make_result(
            check="spike_dip",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="No valid local deviation values found.",
        )

    deviation_threshold = deviation.quantile(deviation_quantile)
    neighbour_threshold = neighbour_difference.quantile(neighbour_quantile)

    q_threshold = local["q"].quantile(0.90)

    spike_dip_mask = (
            (local["local_deviation"] > deviation_threshold)
            & (local["neighbour_difference"] <= neighbour_threshold)
            & (local["q"] >= q_threshold)
    ).fillna(False)

    flagged_timestamps = local.index[spike_dip_mask]

    return make_result(
        check="spike_dip",
        flag=len(flagged_timestamps) > 0,
        value=len(flagged_timestamps),
        flagged_timestamps=timestamps_to_strings(flagged_timestamps),
        series_type=series_type,
        status="completed",
        message=(
            f"Spike/dip candidates = {len(flagged_timestamps)}; "
            f"local deviation threshold learned from Q{deviation_quantile:.3f}; "
            f"neighbour consistency threshold learned from Q{neighbour_quantile:.3f}; "
            f"minimum discharge threshold learned from Q0.900."
        ),
    )

# Step Shift check

def check_step_shift(
    series: pd.Series,
    series_type: str = "unknown",
    window: int = 14,
    score_quantile: float = 0.995,
    stability_quantile: float = 0.50,
    persistence_limit: float = 0.25,
) -> dict:
    """
    Detect step-shift candidates.

    Logic:
    - Step score must be extreme.
    - Before and after windows must be relatively stable.
    - Future median should stay close to after median.
    """
    steps = calculate_step_score_series(series, window=window)

    if steps.empty:
        return make_result(
            check="step_shift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="Not enough data for step-shift detection.",
        )

    valid_scores = steps["step_score"].dropna()

    if valid_scores.empty:
        return make_result(
            check="step_shift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="No valid step scores calculated.",
        )

    score_threshold = valid_scores.quantile(score_quantile)

    before_stability_threshold = steps["before_stability"].dropna().quantile(stability_quantile)
    after_stability_threshold = steps["after_stability"].dropna().quantile(stability_quantile)

    persistence_score = (
        (steps["future_median"] - steps["after_median"]).abs()
        / steps["after_median"].abs().replace(0, np.nan)
    )

    step_mask = (
        (steps["step_score"] > score_threshold)
        & (steps["before_stability"] <= before_stability_threshold)
        & (steps["after_stability"] <= after_stability_threshold)
        & (persistence_score <= persistence_limit)
    ).fillna(False)

    flagged_timestamps = steps.index[step_mask]

    return make_result(
        check="step_shift",
        flag=len(flagged_timestamps) > 0,
        value=len(flagged_timestamps),
        flagged_timestamps=timestamps_to_strings(flagged_timestamps),
        series_type=series_type,
        status="completed",
        message=(
            f"Step-shift candidates = {len(flagged_timestamps)}; "
            f"method = {window}-day before/after rolling median; "
            f"threshold learned from step-score Q{score_quantile:.3f}; "
            f"requires stable before/after windows and persistent after-level."
        ),
    )

# Gradual Drift

def check_gradual_drift(
    series: pd.Series,
    series_type: str = "unknown",
    min_years: int = 5,
    change_quantile: float = 0.95,
) -> dict:
    """
    Detect gradual drift candidates.

    For observed data, drift is treated as a descriptor because it may reflect
    real hydrological or climatic change.

    For simulated/ML data, drift is treated as a soft flag because it may
    indicate model-output drift.
    """
    monthly_year = calculate_monthly_by_year_profile(series)

    if monthly_year.empty:
        return make_result(
            check="gradual_drift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="No monthly-by-year profile available.",
        )

    years = sorted(monthly_year["year"].dropna().unique())

    if len(years) < min_years:
        return make_result(
            check="gradual_drift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message=f"Not enough years for gradual drift check. Years available = {len(years)}.",
        )

    split = len(years) // 2
    early_years = years[:split]
    late_years = years[split:]

    drift_months = []
    directions = []

    for month in range(1, 13):
        month_data = monthly_year[monthly_year["month"] == month].copy()

        if month_data.empty:
            continue

        month_data = month_data.sort_values("year")

        early_values = month_data[
            month_data["year"].isin(early_years)
        ]["median"].dropna()

        late_values = month_data[
            month_data["year"].isin(late_years)
        ]["median"].dropna()

        if early_values.empty or late_values.empty:
            continue

        early_median = early_values.median()
        late_median = late_values.median()
        early_late_change = late_median - early_median

        year_to_year_change = month_data["median"].diff().abs().dropna()

        if year_to_year_change.empty:
            continue

        change_threshold = year_to_year_change.quantile(change_quantile)

        if abs(early_late_change) > change_threshold:
            drift_months.append(month)
            directions.append("increase" if early_late_change > 0 else "decrease")

    if not drift_months:
        return make_result(
            check="gradual_drift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="completed",
            message="No gradual drift candidate detected.",
        )

    increase_count = directions.count("increase")
    decrease_count = directions.count("decrease")

    dominant_direction = "increase" if increase_count >= decrease_count else "decrease"
    consistent_months = max(increase_count, decrease_count)

    # Interpretation differs by series type.
    if series_type.lower() in ["obs", "observed", "observation"]:
        flag = False
        status = "descriptor"
        interpretation = (
            "Drift detected in observed series, but treated as descriptor "
            "because it may reflect real hydrological change."
        )
    else:
        flag = True
        status = "soft_flag"
        interpretation = (
            "Drift detected in simulated/ML series. This may indicate "
            "model-output drift and should be compared against observations."
        )

    return make_result(
        check="gradual_drift",
        flag=flag,
        value=len(drift_months),
        flagged_timestamps=[],
        series_type=series_type,
        status=status,
        message=(
            f"Gradual drift candidate months = {drift_months}; "
            f"dominant direction = {dominant_direction}; "
            f"months with same direction = {consistent_months}; "
            f"threshold learned from month-wise year-to-year median change "
            f"Q{change_quantile:.2f}. {interpretation}"
        ),
    )

def run_behavioural_checks(series: pd.Series, series_type: str = "unknown") -> list:
    """
    Run all behavioural Layer 1 diagnostics.
    """
    return [
        check_zero_flow_regime(series, series_type=series_type),
        check_low_variability(series, series_type=series_type),
        check_spike_dip(series, series_type=series_type),
        check_step_shift(series, series_type=series_type),
        check_gradual_drift(series, series_type=series_type),
    ]