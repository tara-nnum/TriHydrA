"""
Simple Layer 1 QA/QC checks for TriHydrA.

These checks are metadata-independent:
they only inspect the discharge time series itself.
"""

import pandas as pd


def timestamps_to_strings(index) -> list:
    """
    Convert timestamps/index values to readable strings.
    """

    return [str(x) for x in index]

def trim_to_valid_period(series):
    """
    Trim a series to the period between the first and last valid observation.

    This prevents leading or trailing NaN values caused by dataset alignment
    from being interpreted as internal missing-data gaps.
    """
    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()

    if first_valid is None or last_valid is None:
        return series

    return series.loc[first_valid:last_valid]

def check_missing_values(series, threshold=0.05):
    """
    Check for missing values in the series.

    This reports whether missing values exist, and separately whether the
    missing ratio exceeds the configured threshold.
    """
    if series.empty:
        return {
            "check": "missing_values",
            "flag": True,
            "value": 1.0,
            "threshold": threshold,
            "flagged_timestamps": [],
            "message": "Series is empty.",
        }

    missing_mask = series.isna()
    missing_count = int(missing_mask.sum())
    missing_ratio = float(missing_mask.mean())
    flagged_timestamps = timestamps_to_strings(series.index[missing_mask])

    threshold_exceeded = missing_ratio > threshold
    has_missing_values = missing_count > 0

    return {
        "check": "missing_values",
        "flag": has_missing_values,
        "value": missing_ratio,
        "threshold": threshold,
        "flagged_timestamps": flagged_timestamps,
        "message": (
            f"Missing count = {missing_count}; "
            f"missing ratio = {missing_ratio:.2%}; "
            f"threshold exceeded = {threshold_exceeded}"
        ),
    }


def check_long_gaps(series, max_gap=3):
    """
    Identify long internal gaps of consecutive missing values.

    Leading and trailing NaN blocks are ignored because they often represent
    unavailable record periods after alignment with another dataset.
    """
    if series.empty:
        return {
            "check": "long_gaps",
            "flag": False,
            "value": 0,
            "threshold": max_gap,
            "flagged_timestamps": [],
            "message": "Series is empty.",
        }

    working_series = trim_to_valid_period(series)

    if working_series.empty:
        return {
            "check": "long_gaps",
            "flag": False,
            "value": 0,
            "threshold": max_gap,
            "flagged_timestamps": [],
            "message": "No valid data period found.",
        }

    missing_mask = working_series.isna()

    longest_gap = 0
    current_gap = 0
    flagged_timestamps = []

    current_gap_timestamps = []

    for timestamp, is_missing in missing_mask.items():
        if is_missing:
            current_gap += 1
            current_gap_timestamps.append(timestamp)
        else:
            if current_gap > longest_gap:
                longest_gap = current_gap

            if current_gap > max_gap:
                flagged_timestamps.extend(current_gap_timestamps)

            current_gap = 0
            current_gap_timestamps = []

    # Check final internal gap, if the valid-period trimming still leaves one
    if current_gap > longest_gap:
        longest_gap = current_gap

    if current_gap > max_gap:
        flagged_timestamps.extend(current_gap_timestamps)

    flagged_timestamps = timestamps_to_strings(flagged_timestamps)

    return {
        "check": "long_gaps",
        "flag": longest_gap > max_gap,
        "value": int(longest_gap),
        "threshold": max_gap,
        "flagged_timestamps": flagged_timestamps,
        "message": f"Longest internal missing gap = {longest_gap} timestep(s).",
    }


def check_negative_discharge(series: pd.Series, tolerance: float = -1e-6) -> dict:
    """
    Check whether discharge values are physically negative.
    """

    negative_mask = series < tolerance
    negative_count = int(negative_mask.sum())
    flagged_timestamps = timestamps_to_strings(series.index[negative_mask])

    return {
        "check": "negative_discharge",
        "flag": negative_count > 0,
        "value": negative_count,
        "threshold": tolerance,
        "flagged_timestamps": flagged_timestamps,
        "message": f"Negative discharge count = {negative_count}",
    }


def check_duplicate_timestamps(series: pd.Series) -> dict:
    """
    Check whether the time index contains duplicate timestamps.
    """

    duplicate_mask = series.index.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    flagged_timestamps = timestamps_to_strings(series.index[duplicate_mask])

    return {
        "check": "duplicate_timestamps",
        "flag": duplicate_count > 0,
        "value": duplicate_count,
        "threshold": 0,
        "flagged_timestamps": flagged_timestamps,
        "message": f"Duplicate timestamp count = {duplicate_count}",
    }

def check_timestep_consistency(
    series: pd.Series,
    dominant_ratio_threshold: float = 0.95,
) -> dict:
    """
    Check whether the time series has a consistent timestep.

    This infers the dominant timestep from the datetime index.
    For daily data, the dominant timestep should usually be 1 day.
    """

    if len(series) < 2:
        return {
            "check": "timestep_consistency",
            "flag": True,
            "value": None,
            "threshold": dominant_ratio_threshold,
            "flagged_timestamps": [],
            "message": "Series has fewer than 2 timesteps.",
        }

    index = pd.to_datetime(series.index)
    sorted_index = index.sort_values()

    timestep_diffs = sorted_index.to_series().diff().dropna()

    if timestep_diffs.empty:
        return {
            "check": "timestep_consistency",
            "flag": True,
            "value": None,
            "threshold": dominant_ratio_threshold,
            "flagged_timestamps": [],
            "message": "Could not calculate timestep differences.",
        }

    dominant_timestep = timestep_diffs.mode().iloc[0]
    dominant_count = int((timestep_diffs == dominant_timestep).sum())
    total_intervals = len(timestep_diffs)
    dominant_ratio = dominant_count / total_intervals

    irregular_mask = timestep_diffs != dominant_timestep
    flagged_timestamps = timestamps_to_strings(timestep_diffs.index[irregular_mask])

    return {
        "check": "timestep_consistency",
        "flag": dominant_ratio < dominant_ratio_threshold,
        "value": dominant_ratio,
        "threshold": dominant_ratio_threshold,
        "flagged_timestamps": flagged_timestamps,
        "message": (
            f"Dominant timestep = {dominant_timestep}; "
            f"dominant timestep ratio = {dominant_ratio:.2%}"
        ),
    }


def check_low_variability_flow(
    series: pd.Series,
    window_size: int = 7,
    min_duration: int = 7,
    cv_threshold: float = 0.005,
    zero_flow_threshold: float = 1e-6,
) -> dict:
    """
    Check for persistent low-variability non-zero flow.

    This combines old duplicate-value and flatline logic.
    Zero-flow periods are excluded here because they are handled separately
    by check_zero_flow_regime().
    """

    q = series.copy()

    # Exclude missing values and zero-flow periods from this check
    non_zero_q = q.where(q > zero_flow_threshold)

    rolling_mean = non_zero_q.rolling(window=window_size, min_periods=window_size).mean()
    rolling_std = non_zero_q.rolling(window=window_size, min_periods=window_size).std()

    rolling_cv = rolling_std / rolling_mean

    low_variability_mask = rolling_cv < cv_threshold

    if not low_variability_mask.any():
        return {
            "check": "low_variability_flow",
            "flag": False,
            "value": 0,
            "threshold": cv_threshold,
            "flagged_timestamps": [],
            "message": "No persistent low-variability non-zero flow detected.",
        }

    groups = (low_variability_mask != low_variability_mask.shift()).cumsum()
    run_lengths = low_variability_mask.groupby(groups).sum()
    run_lengths = run_lengths[run_lengths > 0]

    max_run = int(run_lengths.max()) if not run_lengths.empty else 0

    flagged_timestamps = []

    for group_id, run_length in run_lengths.items():
        if run_length >= min_duration:
            group_mask = groups == group_id
            flagged_timestamps.extend(series.index[group_mask & low_variability_mask])

    flagged_timestamps = timestamps_to_strings(flagged_timestamps)

    return {
        "check": "low_variability_flow",
        "flag": len(flagged_timestamps) > 0,
        "value": max_run,
        "threshold": f"CV < {cv_threshold} for >= {min_duration} timesteps",
        "flagged_timestamps": flagged_timestamps,
        "message": (
            f"Longest low-variability non-zero run = {max_run} timestep(s)"
        ),
    }


def check_zero_flow_regime(
    series: pd.Series,
    zero_flow_threshold: float = 1e-6,
    min_zero_flow_spell: int = 3,
    seasonal_month_frequency_threshold: float = 0.5,
) -> dict:
    """
    Check zero-flow behaviour.

    This does not automatically treat zero flow as an anomaly.
    It reports zero-flow periods and gives a simple seasonal signal.
    """

    q = series.copy()
    zero_mask = q <= zero_flow_threshold
    zero_count = int(zero_mask.sum())

    if zero_count == 0:
        return {
            "check": "zero_flow_regime",
            "flag": False,
            "value": 0,
            "threshold": zero_flow_threshold,
            "flagged_timestamps": [],
            "message": "No zero-flow values detected.",
        }

    groups = (zero_mask != zero_mask.shift()).cumsum()
    zero_spell_lengths = zero_mask.groupby(groups).sum()
    zero_spell_lengths = zero_spell_lengths[zero_spell_lengths > 0]

    max_zero_spell = int(zero_spell_lengths.max()) if not zero_spell_lengths.empty else 0

    flagged_timestamps = []

    for group_id, spell_length in zero_spell_lengths.items():
        if spell_length >= min_zero_flow_spell:
            group_mask = groups == group_id
            flagged_timestamps.extend(series.index[group_mask & zero_mask])

    flagged_timestamps = timestamps_to_strings(flagged_timestamps)

    # Monthly zero-flow frequency
    temp = pd.DataFrame({"q": q})
    temp["month"] = temp.index.month
    temp["is_zero"] = zero_mask

    monthly_zero_frequency = temp.groupby("month")["is_zero"].mean()
    seasonal_months = monthly_zero_frequency[
        monthly_zero_frequency >= seasonal_month_frequency_threshold
    ].index.tolist()

    if seasonal_months:
        message = (
            f"Zero-flow detected. Longest zero-flow spell = {max_zero_spell} timestep(s). "
            f"Possible seasonal zero-flow months: {seasonal_months}"
        )
    else:
        message = (
            f"Zero-flow detected. Longest zero-flow spell = {max_zero_spell} timestep(s). "
            "No strong monthly zero-flow seasonality detected."
        )

    return {
        "check": "zero_flow_regime",
        "flag": max_zero_spell >= min_zero_flow_spell,
        "value": max_zero_spell,
        "threshold": f">= {min_zero_flow_spell} zero-flow timesteps",
        "flagged_timestamps": flagged_timestamps,
        "message": message,
    }


def check_single_point_spike_dip(
    series: pd.Series,
    spike_factor: float = 3.0,
    dip_factor: float = None,
    neighbour_factor: float = 1.5,
    small_value: float = 1e-6,
) -> dict:
    """
    Detect isolated single-point spikes and dips using local neighbour ratios.

    This check avoids absolute discharge thresholds, z-scores, and gradient scores.
    It compares each point with its immediate previous and next timesteps.

    Missing values, zero-flow periods, negative discharge, duplicate timestamps,
    and non-consecutive timesteps are excluded from this check because they are
    handled by other Layer 1 checks.
    """

    if dip_factor is None:
        dip_factor = 1 / spike_factor

    threshold_text = (
        f"spike >= {spike_factor}x local reference; "
        f"dip <= {dip_factor:.3f}x local reference; "
        f"neighbour ratio <= {neighbour_factor}x"
    )

    if series.empty:
        return {
            "check": "single_point_spike_dip",
            "flag": False,
            "value": 0,
            "threshold": threshold_text,
            "flagged_timestamps": [],
            "flagged_details": [],
            "message": "Series is empty.",
        }

    # Remove duplicated timestamps because shift-based neighbour logic is not
    # meaningful when two rows have the same timestamp.
    q = series[~series.index.duplicated(keep=False)].copy()
    q = q.sort_index()

    if len(q) < 3:
        return {
            "check": "single_point_spike_dip",
            "flag": False,
            "value": 0,
            "threshold": threshold_text,
            "flagged_timestamps": [],
            "flagged_details": [],
            "message": "Not enough data for single-point spike/dip check.",
        }

    # Keep original alignment. Do not drop invalid values before shifting,
    # otherwise the function may accidentally compare across gaps.
    prev_q = q.shift(1)
    curr_q = q
    next_q = q.shift(-1)

    # Require previous, current, and next values to be valid positive discharge.
    valid_value_mask = (
        prev_q.notna()
        & curr_q.notna()
        & next_q.notna()
        & (prev_q > small_value)
        & (curr_q > small_value)
        & (next_q > small_value)
    )

    # Require previous/current/next to be consecutive in time.
    time_index = pd.Series(pd.to_datetime(q.index), index=q.index)
    time_step_before = time_index.diff()
    time_step_after = time_index.shift(-1) - time_index

    positive_steps = time_step_before[time_step_before > pd.Timedelta(0)]

    if positive_steps.empty:
        return {
            "check": "single_point_spike_dip",
            "flag": False,
            "value": 0,
            "threshold": threshold_text,
            "flagged_timestamps": [],
            "flagged_details": [],
            "message": "Could not infer dominant timestep for spike/dip check.",
        }

    dominant_step = positive_steps.mode().iloc[0]

    consecutive_time_mask = (
        (time_step_before == dominant_step)
        & (time_step_after == dominant_step)
    )

    local_reference = (prev_q + next_q) / 2

    neighbour_values = pd.concat([prev_q, next_q], axis=1)
    neighbour_max = neighbour_values.max(axis=1)
    neighbour_min = neighbour_values.min(axis=1)
    neighbour_ratio = neighbour_max / neighbour_min

    neighbours_are_consistent = neighbour_ratio <= neighbour_factor

    current_ratio = curr_q / local_reference

    valid_candidate_mask = (
        valid_value_mask
        & consecutive_time_mask
        & (local_reference > small_value)
        & neighbours_are_consistent
    )

    spike_mask = valid_candidate_mask & (current_ratio >= spike_factor)
    dip_mask = valid_candidate_mask & (current_ratio <= dip_factor)

    anomaly_mask = spike_mask | dip_mask

    flagged_timestamps = timestamps_to_strings(q.index[anomaly_mask])

    flagged_details = []

    for timestamp in q.index[anomaly_mask]:
        anomaly_type = "spike" if spike_mask.loc[timestamp] else "dip"

        flagged_details.append(
            {
                "timestamp": str(timestamp),
                "type": anomaly_type,
                "previous_q": round(float(prev_q.loc[timestamp]), 4),
                "current_q": round(float(curr_q.loc[timestamp]), 4),
                "next_q": round(float(next_q.loc[timestamp]), 4),
                "local_reference": round(float(local_reference.loc[timestamp]), 4),
                "current_ratio": round(float(current_ratio.loc[timestamp]), 4),
                "neighbour_ratio": round(float(neighbour_ratio.loc[timestamp]), 4),
            }
        )

    spike_count = int(spike_mask.sum())
    dip_count = int(dip_mask.sum())
    anomaly_count = spike_count + dip_count

    return {
        "check": "single_point_spike_dip",
        "flag": anomaly_count > 0,
        "value": anomaly_count,
        "threshold": threshold_text,
        "flagged_timestamps": flagged_timestamps,
        "flagged_details": flagged_details,
        "message": (
            f"Single-point spike count = {spike_count}; "
            f"single-point dip count = {dip_count}; "
            f"dominant timestep = {dominant_step}."
        ),
    }

def check_step_shift(
    series: pd.Series,
    window_size: int = 14,
    relative_change_threshold: float = 0.5,
) -> dict:
    """
    Detect abrupt persistent level shifts using before/after rolling medians.

    This is a simple first-pass check.
    """

    q = series.copy()

    before_median = q.rolling(window=window_size, min_periods=window_size).median()
    after_median = q[::-1].rolling(window=window_size, min_periods=window_size).median()[::-1]

    relative_change = (after_median - before_median).abs() / before_median.abs().replace(0, pd.NA)

    step_mask = relative_change > relative_change_threshold

    flagged_timestamps = timestamps_to_strings(series.index[step_mask.fillna(False)])
    max_change = relative_change.max(skipna=True)

    if pd.isna(max_change):
        max_change = 0

    return {
        "check": "step_shift",
        "flag": len(flagged_timestamps) > 0,
        "value": float(max_change),
        "threshold": relative_change_threshold,
        "flagged_timestamps": flagged_timestamps,
        "message": f"Maximum before/after relative level change = {float(max_change):.2%}",
    }


def check_gradual_drift(
    series: pd.Series,
    min_years: int = 3,
    baseline_years: int = 2,
    recent_years: int = 2,
    drift_threshold: float = 0.25,
    min_flagged_months: int = 3,
) -> dict:
    """
    Detect gradual drift using seasonal/monthly median behaviour.

    This check compares early-period and late-period monthly medians.
    It avoids comparing the entire time series as one block, because streamflow
    is strongly seasonal and hydrologically non-stationary.

    A drift candidate is flagged only if multiple months show a relative change
    greater than the configured threshold.
    """

    q = series.dropna().copy()

    if len(q) == 0:
        return {
            "check": "seasonal_gradual_drift",
            "flag": False,
            "value": 0,
            "threshold": drift_threshold,
            "flagged_timestamps": [],
            "flagged_details": [],
            "message": "Series has no valid data for seasonal drift check.",
        }

    q.index = pd.to_datetime(q.index)

    temp = pd.DataFrame({"q": q})
    temp["year"] = temp.index.year
    temp["month"] = temp.index.month

    available_years = sorted(temp["year"].unique())

    if len(available_years) < min_years:
        return {
            "check": "seasonal_gradual_drift",
            "flag": False,
            "value": 0,
            "threshold": drift_threshold,
            "flagged_timestamps": [],
            "flagged_details": [],
            "message": (
                f"Not enough years for seasonal drift check. "
                f"Available years = {len(available_years)}; required = {min_years}."
            ),
        }

    early_years = available_years[:baseline_years]
    late_years = available_years[-recent_years:]

    monthly_medians = (
        temp.groupby(["year", "month"])["q"]
        .median()
        .reset_index()
    )

    flagged_details = []

    for month in range(1, 13):
        early_values = monthly_medians[
            (monthly_medians["month"] == month)
            & (monthly_medians["year"].isin(early_years))
        ]["q"]

        late_values = monthly_medians[
            (monthly_medians["month"] == month)
            & (monthly_medians["year"].isin(late_years))
        ]["q"]

        if len(early_values) == 0 or len(late_values) == 0:
            continue

        early_median = early_values.median()
        late_median = late_values.median()

        if abs(early_median) <= 1e-6:
            continue

        relative_change = (late_median - early_median) / abs(early_median)

        if abs(relative_change) > drift_threshold:
            flagged_details.append({
                "month": month,
                "early_years": f"{early_years[0]}-{early_years[-1]}",
                "late_years": f"{late_years[0]}-{late_years[-1]}",
                "early_monthly_median": round(float(early_median), 4),
                "late_monthly_median": round(float(late_median), 4),
                "relative_change": round(float(relative_change), 4),
            })

    flagged_month_count = len(flagged_details)
    flag = flagged_month_count >= min_flagged_months

    flagged_timestamps = []

    if flag:
        start_time = q.index.min()
        end_time = q.index.max()
        flagged_timestamps = timestamps_to_strings([start_time, end_time])

    return {
        "check": "seasonal_gradual_drift",
        "flag": flag,
        "value": flagged_month_count,
        "threshold": (
            f"monthly median change > {drift_threshold:.0%}; "
            f"flag if >= {min_flagged_months} months"
        ),
        "flagged_timestamps": flagged_timestamps,
        "flagged_details": flagged_details,
        "message": (
            f"Seasonal drift candidate: {flagged_month_count} month(s) exceeded "
            f"{drift_threshold:.0%} relative change between early years "
            f"({early_years[0]}-{early_years[-1]}) and late years "
            f"({late_years[0]}-{late_years[-1]})."
        ),
    }