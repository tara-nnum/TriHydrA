import numpy as np
import pandas as pd
import ruptures as rpt
import pymannkendall as mk
from scipy.stats import mannwhitneyu
from statsmodels.tsa.seasonal import STL

from src.trihydra.layer1.timeseries_validity import get_valid_record, timestamps_to_strings
from src.trihydra.layer1.behaviour_profile import calculate_rolling_variability_series


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
) -> dict:
    """
    Detect isolated single-point spike/dip candidates.

    Logic:
    - A spike/dip must reverse direction immediately (a turning point):
      the change into the point and the change out of the point have
      opposite signs.
    - The two-sided jump is the SMALLER of the two changes either side,
      so the point must both depart and return strongly.
    - A recovery score is highest when the values immediately before
      and after the candidate are close to each other. This down-
      weights points sitting inside a real, still-developing rise or
      recession, where before and after naturally differ.
    - The jump is scaled against the normal daily-change magnitude for
      that calendar month, learned from the series itself.
    - Both the raw-discharge and log-discharge scales are checked,
      since a change can be extreme in relative terms without being
      extreme in absolute terms, or vice versa.
    - The cutoff between "candidate" and "flagged" is the single
      largest natural gap in the sorted candidate scores, rather than
      a fixed or learned quantile threshold.
    """
    s = get_valid_record(series)

    if s.empty:
        return make_result(
            check="spike_dip",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="No valid data found.",
        )

    q = s.astype(float)

    previous_q = q.shift(1)
    next_q = q.shift(-1)
    change_into_point = q - previous_q
    change_out_of_point = next_q - q

    def _turning_point_scores(values: pd.Series):
        """Score every turning-point candidate on a given scale (raw or log)."""
        previous = values.shift(1)
        nxt = values.shift(-1)

        change_into = values - previous
        change_out = nxt - values

        turning_point = (
            (change_into * change_out < 0)
            & previous.notna()
            & values.notna()
            & nxt.notna()
        )

        two_sided_jump = np.minimum(change_into.abs(), change_out.abs())

        recovery = 1 - (
            (nxt - previous).abs()
            / (change_into.abs() + change_out.abs())
        )

        normal_change = (
            values.diff()
            .abs()
            .groupby(values.index.month)
            .transform("median")
        )

        # A calendar month can have zero typical day-to-day movement
        # on a heavily regulated or quantized river (e.g. a fixed
        # release schedule, or coarse rounding) -- dividing by that
        # zero silently produces inf, not a crash, and the crash only
        # happens later when multiple inf scores collide during
        # gap-finding below. Fall back to the smallest genuinely
        # nonzero day-to-day change found anywhere in this record
        # (its own quantization/resolution floor) rather than an
        # arbitrary constant, so a real jump during an otherwise-flat
        # month is judged against the finest movement this river
        # actually makes, instead of exploding to infinity.
        all_diffs = values.diff().abs()
        nonzero_diffs = all_diffs[all_diffs > 0]
        fallback_normal_change = (
            float(nonzero_diffs.min()) if not nonzero_diffs.empty else np.nan
        )
        normal_change = normal_change.where(normal_change > 0, fallback_normal_change)

        score = (two_sided_jump / normal_change) * recovery

        candidate_scores = (
            score[turning_point]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(ascending=False)
        )

        if candidate_scores.empty:
            return score, pd.Index([])

        gaps = pd.Series(
            candidate_scores.iloc[:-1].to_numpy()
            - candidate_scores.iloc[1:].to_numpy()
        )

        if len(gaps) > 0:
            gap_position = gaps.idxmax()
            cutoff = (
                candidate_scores.iloc[gap_position]
                + candidate_scores.iloc[gap_position + 1]
            ) / 2
            flags = candidate_scores[candidate_scores >= cutoff].index
        else:
            flags = candidate_scores.index

        return score, flags

    raw_score, raw_flags = _turning_point_scores(q)
    log_score, log_flags = _turning_point_scores(np.log1p(q))

    flagged_timestamps = raw_flags.union(log_flags).sort_values()

    if len(flagged_timestamps) == 0:
        return make_result(
            check="spike_dip",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="completed",
            message="No spike/dip candidates detected.",
        )

    candidate_details = []

    for timestamp in flagged_timestamps:
        if timestamp in raw_flags and timestamp in log_flags:
            detection = "raw and log scales"
        elif timestamp in log_flags:
            detection = "log scale only"
        else:
            detection = "raw scale only"

        candidate_details.append({
            "timestamp": str(timestamp),
            "type": "spike" if change_into_point.loc[timestamp] > 0 else "dip",
            "previous_value": float(previous_q.loc[timestamp]),
            "flagged_value": float(q.loc[timestamp]),
            "next_value": float(next_q.loc[timestamp]),
            "raw_score": float(raw_score.loc[timestamp]) if pd.notna(raw_score.loc[timestamp]) else None,
            "log_score": float(log_score.loc[timestamp]) if pd.notna(log_score.loc[timestamp]) else None,
            "detection": detection,
        })

    result = make_result(
        check="spike_dip",
        flag=True,
        value=len(flagged_timestamps),
        flagged_timestamps=timestamps_to_strings(flagged_timestamps),
        series_type=series_type,
        status="completed",
        message=(
            f"Spike/dip candidates = {len(flagged_timestamps)}; "
            f"method = two-sided turning-point jump scaled against month-wise "
            f"normal change and weighted by before/after recovery, checked on "
            f"both raw and log-discharge scales; cutoff = largest natural gap "
            f"in the sorted candidate scores."
        ),
    )
    result["candidate_details"] = candidate_details

    return result

# Step Shift check

def check_step_shift(
    series: pd.Series,
    series_type: str = "unknown",
    min_regime_days: int = 180,
    significance_level: float = 0.01,
    include_flow_duration_curve: bool = True,
) -> dict:
    """
    Detect step-shift candidates with a multi-changepoint PELT scan.

    Logic:
    - Deseasonalize on the RAW scale (additive), by subtracting each
      calendar day-of-year's own MEDIAN across all years. Median, not
      mean, keeps the climatology itself from being dragged around by
      floods -- consistent with every other check in this file. Raw
      scale, not log, is used because a fixed/additive shift becomes
      value-dependent (and can nearly vanish at high flow) once
      log-transformed; see check_spike_dip for where log scale is the
      right call instead (a multiplicative, not additive, question).
    - Run PELT (an exact, penalised multi-changepoint search) on the
      anomaly, so however many real regimes exist get found in one
      global fit, instead of one changepoint at a time.
    - The penalty follows the standard BIC shape (variance x log(n)),
      but the inflation multiplier is learned from the series' own
      lag-1 autocorrelation instead of a fixed number. Daily discharge
      is highly persistent (today looks like yesterday), which
      violates the independence the plain BIC penalty assumes and
      would otherwise flag far too many spurious breaks; a more
      persistent series gets a larger inflation automatically, a less
      persistent one gets a smaller one.
    - min_regime_days sets the shortest span that counts as its own
      regime (and the comparison window for the significance test), so
      a change must hold for a meaningful duration to register --
      permanent or temporary/bounded shifts are both covered, since
      only a minimum duration is required, not that it lasts forever.
    - Each candidate boundary is confirmed with a two-sided
      Mann-Whitney U test (anomaly values before vs. after), which
      does not assume normally-distributed data. The significance
      level is Bonferroni-corrected by how many boundaries were found,
      so testing more candidates does not, by itself, inflate the
      false-positive rate.
    - A flow-duration curve is kept per regime, so a boundary can be
      characterised further downstream: a genuine additive shift moves
      every percentile of the curve by roughly the same amount, rather
      than only the flood peaks or only the low flows.
    """
    s = get_valid_record(series)

    if s.empty or len(s) < 2 * min_regime_days:
        return make_result(
            check="step_shift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message=(
                "Not enough valid data for step-shift detection "
                f"(need at least {2 * min_regime_days} points)."
            ),
        )

    q = s.astype(float)

    # Deseasonalize: each calendar day-of-year's own median across all
    # years is its climatology.
    daily_climatology = q.groupby(q.index.dayofyear).transform("median")
    anomaly_series = q - daily_climatology

    # PELT cannot handle NaN. get_valid_record() only trims leading/
    # trailing gaps, so a real record can still have internal missing
    # days -- drop those just for the PELT input, and keep a separate
    # gap-free index to translate array positions back into dates. The
    # comparison windows below therefore count valid days, not
    # calendar days, when a gap sits inside a window.
    anomaly_valid = anomaly_series.dropna()
    valid_index = anomaly_valid.index
    anomaly = anomaly_valid.to_numpy()

    n_dropped_internal = int(anomaly_series.isna().sum())

    n_samples = len(anomaly)

    if n_samples < 2 * min_regime_days:
        return make_result(
            check="step_shift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message=(
                "Not enough valid (non-gap) data for step-shift detection "
                f"after excluding {n_dropped_internal} internal gap day(s) "
                f"(need at least {2 * min_regime_days} valid points)."
            ),
        )

    variance = float(np.var(anomaly))

    # Learn the penalty's inflation factor from this series' own
    # persistence, instead of assuming a fixed multiplier for every
    # station. The raw (1+rho)/(1-rho) variance-inflation factor for
    # an AR(1)-like process was tested directly and found far too
    # aggressive in practice (it drove the penalty high enough to find
    # zero regimes on real, heavily persistent daily discharge) -- a
    # square-root-damped version tracks the same idea (more persistent
    # series get a bigger inflation, less persistent ones get a
    # smaller one) while staying in a range PELT can actually use.
    lag1_autocorr = anomaly_valid.autocorr(lag=1)
    lag1_autocorr = 0.0 if pd.isna(lag1_autocorr) else float(np.clip(lag1_autocorr, 0.0, 0.99))
    autocorrelation_inflation = np.sqrt((1 + lag1_autocorr) / (1 - lag1_autocorr))

    dynamic_penalty = autocorrelation_inflation * variance * np.log(n_samples)

    algo = rpt.Pelt(model="l2", min_size=min_regime_days).fit(anomaly)
    breakpoints = algo.predict(pen=dynamic_penalty)

    if not breakpoints or breakpoints[-1] != n_samples:
        breakpoints.append(n_samples)

    boundary_indices = breakpoints[:-1]

    # Regime boundaries (in dates) and segments are built from
    # valid_index / q directly below, rather than from positional
    # slicing of the gap-containing q, so gaps elsewhere in the record
    # cannot shift a later regime's reported dates.
    regime_dates = [valid_index[0]] + [valid_index[i] for i in boundary_indices] + [q.index[-1]]

    if not boundary_indices:
        result = make_result(
            check="step_shift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="completed",
            message=(
                f"No regime boundaries found; dynamic penalty = {dynamic_penalty:.2f} "
                f"(autocorrelation inflation = {autocorrelation_inflation:.2f} learned "
                f"from lag-1 autocorrelation = {lag1_autocorr:.2f})."
            ),
        )
        result["regime_boundaries"] = []
        result["regime_summary"] = []
        return result

    # Bonferroni-correct the significance level by how many boundaries
    # are being tested here, so a series with more candidates doesn't
    # automatically get a higher false-positive rate.
    corrected_alpha = significance_level / len(boundary_indices)

    flagged_timestamps = []
    regime_boundaries = []

    for end_idx in boundary_indices:
        window_before = anomaly[max(0, end_idx - min_regime_days):end_idx]
        window_after = anomaly[end_idx:end_idx + min_regime_days]

        if len(window_before) < 2 or len(window_after) < 2:
            continue

        _, p_value = mannwhitneyu(window_before, window_after, alternative="two-sided")

        is_robust = bool(p_value < corrected_alpha)
        boundary_timestamp = valid_index[end_idx]

        if is_robust:
            flagged_timestamps.append(boundary_timestamp)

        regime_boundaries.append({
            "boundary_timestamp": str(boundary_timestamp),
            "before_median": float(np.median(window_before)),
            "after_median": float(np.median(window_after)),
            "p_value": float(p_value),
            "robust": is_robust,
        })

    # Segment-level summary, including a flow-duration curve per
    # regime for downstream comparison (e.g. checking whether a whole
    # regime's distribution shifted together, not just its centre).
    # Sliced by DATE on the original (gap-containing) series, so every
    # valid day in each regime is included, not just the gap-free
    # subset PELT itself operated on.
    regime_summary = []

    for i in range(len(regime_dates) - 1):
        start_date, end_date = regime_dates[i], regime_dates[i + 1]
        is_last = (i == len(regime_dates) - 2)

        segment = (
            q.loc[start_date:end_date]
            if is_last
            else q.loc[start_date:end_date].iloc[:-1]
        )

        if segment.empty:
            continue

        entry = {
            "start": str(segment.index[0]),
            "end": str(segment.index[-1]),
            "n_days": int(len(segment)),
            "mean_flow": float(segment.mean()),
            "median_flow": float(segment.median()),
        }

        if include_flow_duration_curve:
            sorted_flows = np.sort(segment.to_numpy())[::-1]
            ranks = np.arange(1, len(sorted_flows) + 1)
            exceedance_percent = (ranks / (len(sorted_flows) + 1)) * 100
            entry["flow_duration_curve"] = {
                "exceedance_percent": exceedance_percent.round(2).tolist(),
                "flow": sorted_flows.round(3).tolist(),
            }

        regime_summary.append(entry)

    result = make_result(
        check="step_shift",
        flag=len(flagged_timestamps) > 0,
        value=len(flagged_timestamps),
        flagged_timestamps=timestamps_to_strings(flagged_timestamps),
        series_type=series_type,
        status="completed",
        message=(
            f"Step-shift candidates = {len(flagged_timestamps)} of {len(boundary_indices)} "
            f"regime boundaries found; method = PELT multi-changepoint scan on "
            f"day-of-year deseasonalized anomaly (min regime length = {min_regime_days} "
            f"days); penalty = variance x log(n), inflated x{autocorrelation_inflation:.2f} "
            f"for this series' own lag-1 autocorrelation of {lag1_autocorr:.2f}; each "
            f"boundary confirmed with a two-sided Mann-Whitney U test, Bonferroni-"
            f"corrected to alpha = {corrected_alpha:.4g}."
            + (
                f" {n_dropped_internal} internal gap day(s) were excluded before "
                "the PELT scan (treated as sequential, which can slightly bias "
                "results across a gap)."
                if n_dropped_internal > 0 else ""
            )
        ),
    )
    result["regime_boundaries"] = regime_boundaries
    result["regime_summary"] = regime_summary

    return result

# Gradual Drift

def check_gradual_drift(
    series: pd.Series,
    series_type: str = "unknown",
    significance_level: float = 0.05,
    min_daily_values_per_month: int = 20,
    min_record_years: float = 3.0,
) -> dict:
    """
    Detect gradual drift using a Seasonal Mann-Kendall test on monthly
    median streamflow, with an STL decomposition for context.

    Logic:
    - Aggregate to monthly medians (duplicate daily timestamps are
      merged by their own median first). A month needs at least
      min_daily_values_per_month valid days to count; short internal
      gaps (up to 2 months) are time-interpolated, long gaps stop the
      check rather than invent data.
    - The Seasonal Mann-Kendall test (period=12) compares each
      calendar month only against its own same-month history (January
      vs. January, and so on), so the seasonal cycle itself is never
      mistaken for a trend.
    - Sen's slope estimates the yearly rate of change; the estimated
      total change over the record is compared, as a percentage,
      against a record-length-dependent screening threshold (the same
      provisional TriHydrA thresholds used elsewhere in this project:
      2%/5%/8% for <=5/<=10/>10 year records).
    - A flag requires BOTH statistical significance and a meaningful
      magnitude, the same two-part logic as every other check here.
    - An STL decomposition (trend + seasonal + residual, robust to
      outliers) provides seasonal_strength and trend_strength as
      supporting context, not part of the flagging decision itself.
    - For observed data, drift is reported as a descriptor rather than
      a flag, since it may reflect a real hydrological or climatic
      change. For simulated/ML data, it is reported as a soft flag,
      since it may indicate model-output drift.
    - Note: a Mann-Kendall-family test cannot tell a genuinely gradual
      change apart from a lasting abrupt one (e.g. a real step shift
      near the end of the record can also look like "increasing
      drift") -- it only tests whether later values rank consistently
      higher or lower than earlier ones.
    """
    s = get_valid_record(series)

    if s.empty:
        return make_result(
            check="gradual_drift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="No valid data found.",
        )

    daily_flow = s.astype(float)
    daily_flow = daily_flow.groupby(daily_flow.index).median().sort_index()

    daily_index = pd.date_range(daily_flow.index.min(), daily_flow.index.max(), freq="D")
    daily_flow = daily_flow.reindex(daily_index)

    monthly_flow = daily_flow.resample("MS").median()
    monthly_valid_count = daily_flow.resample("MS").count()
    monthly_flow = monthly_flow.where(monthly_valid_count >= min_daily_values_per_month)

    first_valid_month = monthly_flow.first_valid_index()
    last_valid_month = monthly_flow.last_valid_index()

    if first_valid_month is None or last_valid_month is None:
        return make_result(
            check="gradual_drift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message="No valid monthly streamflow values.",
        )

    monthly_flow = monthly_flow.loc[first_valid_month:last_valid_month]
    missing_months_before_fill = int(monthly_flow.isna().sum())

    # Fill only short internal gaps; long gaps stop the check rather
    # than invent hydrological behaviour.
    monthly_analysis = monthly_flow.interpolate(method="time", limit=2, limit_area="inside")

    if monthly_analysis.isna().any():
        missing_dates = monthly_analysis[monthly_analysis.isna()].index.strftime("%Y-%m").tolist()
        return make_result(
            check="gradual_drift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message=(
                "Monthly series still contains long gaps after short-gap "
                f"interpolation: {missing_dates}."
            ),
        )

    record_months = len(monthly_analysis)
    record_years = record_months / 12

    if record_years < min_record_years:
        return make_result(
            check="gradual_drift",
            flag=False,
            value=0,
            flagged_timestamps=[],
            series_type=series_type,
            status="skipped",
            message=(
                f"At least {min_record_years} years of monthly data are required; "
                f"only {record_years:.2f} available."
            ),
        )

    seasonal_mk = mk.seasonal_test(monthly_analysis.values, period=12, alpha=significance_level)

    drift_direction = seasonal_mk.trend
    drift_significant = bool(seasonal_mk.h)
    drift_p_value = float(seasonal_mk.p)
    mann_kendall_z = float(seasonal_mk.z)
    mann_kendall_tau = float(seasonal_mk.Tau)
    sen_slope_m3s_per_year = float(seasonal_mk.slope)

    estimated_total_change_m3s = sen_slope_m3s_per_year * max(record_years - 1, 0)
    reference_flow_m3s = float(monthly_analysis.median())

    if reference_flow_m3s != 0:
        estimated_relative_change_percent = estimated_total_change_m3s / reference_flow_m3s * 100
    else:
        estimated_relative_change_percent = np.nan

    # Provisional TriHydrA screening thresholds; can later be
    # calibrated across all stations.
    if record_years <= 5:
        magnitude_threshold_percent = 2.0
    elif record_years <= 10:
        magnitude_threshold_percent = 5.0
    else:
        magnitude_threshold_percent = 8.0

    meaningful_magnitude = (
        np.isfinite(estimated_relative_change_percent)
        and abs(estimated_relative_change_percent) >= magnitude_threshold_percent
    )

    gradual_drift_flag = drift_significant and meaningful_magnitude

    # STL decomposition for context only -- not part of the flagging
    # decision itself.
    stl_result = STL(monthly_analysis, period=12, robust=True).fit()
    stl_trend = pd.Series(stl_result.trend, index=monthly_analysis.index)
    stl_seasonal = pd.Series(stl_result.seasonal, index=monthly_analysis.index)
    stl_residual = pd.Series(stl_result.resid, index=monthly_analysis.index)

    seasonal_denominator = np.var(stl_residual + stl_seasonal)
    trend_denominator = np.var(stl_residual + stl_trend)

    seasonal_strength = (
        max(0.0, 1 - np.var(stl_residual) / seasonal_denominator)
        if seasonal_denominator > 0 else np.nan
    )
    trend_strength = (
        max(0.0, 1 - np.var(stl_residual) / trend_denominator)
        if trend_denominator > 0 else np.nan
    )

    if gradual_drift_flag:
        drift_status = f"Gradual {drift_direction} drift detected"
    elif drift_significant:
        drift_status = (
            f"Statistically significant {drift_direction} trend, but the estimated "
            "magnitude is below the screening threshold"
        )
    elif meaningful_magnitude:
        drift_status = (
            f"Estimated {drift_direction} change is large, but Seasonal Mann-Kendall "
            "does not show statistical significance"
        )
    else:
        drift_status = "No clear gradual drift detected"

    # Interpretation differs by series type, matching the rest of this
    # project: real drift in an observed record may reflect genuine
    # hydrological or climatic change, so it is reported as a
    # descriptor rather than an automatic flag; the same pattern in
    # simulated/ML output may indicate model drift, so it is flagged.
    if gradual_drift_flag and series_type.lower() in ["obs", "observed", "observation"]:
        flag = False
        status = "descriptor"
        interpretation = (
            "Drift detected in observed series, but treated as descriptor "
            "because it may reflect real hydrological change."
        )
    elif gradual_drift_flag:
        flag = True
        status = "soft_flag"
        interpretation = (
            "Drift detected in simulated/ML series. This may indicate "
            "model-output drift and should be compared against observations."
        )
    else:
        flag = False
        status = "completed"
        interpretation = ""

    result = make_result(
        check="gradual_drift",
        flag=flag,
        value=(
            round(estimated_relative_change_percent, 3)
            if np.isfinite(estimated_relative_change_percent) else 0
        ),
        flagged_timestamps=[],
        series_type=series_type,
        status=status,
        message=(
            f"{drift_status}. Seasonal Mann-Kendall direction = {drift_direction}; "
            f"p-value = {drift_p_value:.4f} (alpha = {significance_level}); "
            f"Sen slope = {sen_slope_m3s_per_year:.2f} per year; estimated total "
            f"change = {estimated_relative_change_percent:.1f}% over "
            f"{record_years:.1f} years (magnitude threshold = "
            f"{magnitude_threshold_percent:.1f}%). {interpretation}"
        ).strip(),
    )
    result["drift_diagnostics"] = {
        "record_years": round(record_years, 2),
        "direction": drift_direction,
        "p_value": drift_p_value,
        "z_statistic": mann_kendall_z,
        "kendall_tau": mann_kendall_tau,
        "significant": drift_significant,
        "sen_slope_m3s_per_year": sen_slope_m3s_per_year,
        "estimated_total_change_m3s": estimated_total_change_m3s,
        "reference_flow_m3s": reference_flow_m3s,
        "estimated_relative_change_percent": estimated_relative_change_percent,
        "magnitude_threshold_percent": magnitude_threshold_percent,
        "meaningful_magnitude": meaningful_magnitude,
        "gradual_drift_flag": gradual_drift_flag,
        "seasonal_strength": float(seasonal_strength) if np.isfinite(seasonal_strength) else None,
        "trend_strength": float(trend_strength) if np.isfinite(trend_strength) else None,
        "missing_months_filled": missing_months_before_fill,
    }

    return result

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