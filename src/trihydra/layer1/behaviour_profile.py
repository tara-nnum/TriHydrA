import numpy as np
import pandas as pd

from src.trihydra.layer1.timeseries_validity import get_valid_record


def safe_cv(series: pd.Series) -> float:
    """Coefficient of variation, safely handled when mean is zero."""
    mean_value = series.mean()

    if pd.isna(mean_value) or mean_value == 0:
        return np.nan

    return series.std() / mean_value


def q(series: pd.Series, quantile: float) -> float:
    """Safe quantile calculation."""
    clean = series.dropna()

    if clean.empty:
        return np.nan

    return clean.quantile(quantile)


def calculate_summary_metrics(series: pd.Series) -> dict:
    """
    Calculate basic distribution and availability metrics.
    """
    s = get_valid_record(series)

    if s.empty:
        return {
            "record_start": None,
            "record_end": None,
            "total_count": 0,
            "valid_count": 0,
            "missing_count": 0,
            "missing_ratio": np.nan,
        }

    valid = s.dropna()

    return {
        "record_start": str(s.index.min()),
        "record_end": str(s.index.max()),
        "total_count": int(len(s)),
        "valid_count": int(valid.count()),
        "missing_count": int(s.isna().sum()),
        "missing_ratio": float(s.isna().mean()),
        "min": float(valid.min()) if not valid.empty else np.nan,
        "max": float(valid.max()) if not valid.empty else np.nan,
        "mean": float(valid.mean()) if not valid.empty else np.nan,
        "median": float(valid.median()) if not valid.empty else np.nan,
        "std": float(valid.std()) if not valid.empty else np.nan,
        "cv": float(safe_cv(valid)),
        "iqr": float(q(valid, 0.75) - q(valid, 0.25)),
        "q01": float(q(valid, 0.01)),
        "q05": float(q(valid, 0.05)),
        "q10": float(q(valid, 0.10)),
        "q25": float(q(valid, 0.25)),
        "q75": float(q(valid, 0.75)),
        "q90": float(q(valid, 0.90)),
        "q95": float(q(valid, 0.95)),
        "q99": float(q(valid, 0.99)),
    }


def calculate_zero_flow_metrics(series: pd.Series, decimals: int = 3) -> dict:
    """
    Calculate zero-flow frequency and spell metrics.
    Values are rounded first to avoid tiny numerical artefacts.
    """
    s = get_valid_record(series)

    if s.empty:
        return {
            "zero_count": 0,
            "zero_ratio": np.nan,
            "zero_spell_count": 0,
            "max_zero_spell_length": 0,
            "median_zero_spell_length": np.nan,
            "q95_zero_spell_length": np.nan,
        }

    rounded = s.round(decimals)
    zero_mask = rounded == 0

    spell_id = (zero_mask != zero_mask.shift()).cumsum()
    spell_lengths = zero_mask.groupby(spell_id).sum()
    zero_spell_lengths = spell_lengths[spell_lengths > 0].astype(int)

    return {
        "zero_count": int(zero_mask.sum()),
        "zero_ratio": float(zero_mask.mean()),
        "zero_spell_count": int(len(zero_spell_lengths)),
        "max_zero_spell_length": int(zero_spell_lengths.max()) if not zero_spell_lengths.empty else 0,
        "median_zero_spell_length": float(zero_spell_lengths.median()) if not zero_spell_lengths.empty else np.nan,
        "q95_zero_spell_length": float(zero_spell_lengths.quantile(0.95)) if not zero_spell_lengths.empty else np.nan,
    }


def calculate_change_metrics(series: pd.Series) -> dict:
    """
    Calculate daily change behaviour metrics.
    """
    s = get_valid_record(series).dropna()

    if len(s) < 2:
        return {}

    daily_change = s.diff().dropna()
    abs_change = daily_change.abs()

    # Log change is useful because it reduces scale effects.
    # Clip at zero because discharge should not go negative after basic checks.
    log_q = np.log1p(s.clip(lower=0))
    log_change = log_q.diff().abs().dropna()

    return {
        "daily_change_mean": float(daily_change.mean()),
        "daily_change_median": float(daily_change.median()),
        "daily_change_std": float(daily_change.std()),
        "daily_change_q95": float(q(daily_change.abs(), 0.95)),
        "daily_change_q99": float(q(daily_change.abs(), 0.99)),
        "abs_change_mean": float(abs_change.mean()),
        "abs_change_median": float(abs_change.median()),
        "abs_change_std": float(abs_change.std()),
        "abs_change_q95": float(q(abs_change, 0.95)),
        "abs_change_q99": float(q(abs_change, 0.99)),
        "log_change_median": float(log_change.median()) if not log_change.empty else np.nan,
        "log_change_q95": float(q(log_change, 0.95)),
        "log_change_q99": float(q(log_change, 0.99)),
    }


def calculate_rolling_metrics(series: pd.Series, windows=(7, 14, 30)) -> dict:
    """
    Calculate rolling variability summary metrics.
    """
    s = get_valid_record(series).dropna()
    results = {}

    if s.empty:
        return results

    for window in windows:
        rolling_mean = s.rolling(window).mean()
        rolling_std = s.rolling(window).std()
        rolling_cv = rolling_std / rolling_mean.replace(0, np.nan)
        rolling_range = s.rolling(window).max() - s.rolling(window).min()

        prefix = f"rolling_{window}d"

        results.update({
            f"{prefix}_std_median": float(rolling_std.median()),
            f"{prefix}_std_q05": float(q(rolling_std, 0.05)),
            f"{prefix}_std_q95": float(q(rolling_std, 0.95)),
            f"{prefix}_cv_median": float(rolling_cv.median()),
            f"{prefix}_cv_q01": float(q(rolling_cv, 0.01)),
            f"{prefix}_cv_q05": float(q(rolling_cv, 0.05)),
            f"{prefix}_cv_q95": float(q(rolling_cv, 0.95)),
            f"{prefix}_cv_q99": float(q(rolling_cv, 0.99)),
            f"{prefix}_range_median": float(rolling_range.median()),
            f"{prefix}_range_q05": float(q(rolling_range, 0.05)),
            f"{prefix}_range_q95": float(q(rolling_range, 0.95)),
        })

    return results



def calculate_local_deviation_metrics(series: pd.Series) -> dict:
    """
    Calculate local deviation metrics for isolated spike/dip detection.
    """
    s = get_valid_record(series).dropna()

    if len(s) < 3:
        return {}

    prev_q = s.shift(1)
    next_q = s.shift(-1)

    local_reference = (prev_q + next_q) / 2
    local_reference = local_reference.replace(0, np.nan)

    local_deviation = (s - local_reference).abs() / local_reference
    neighbour_difference = (next_q - prev_q).abs() / local_reference

    local_deviation = local_deviation.replace([np.inf, -np.inf], np.nan).dropna()
    neighbour_difference = neighbour_difference.replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "local_deviation_median": float(local_deviation.median()) if not local_deviation.empty else np.nan,
        "local_deviation_q95": float(q(local_deviation, 0.95)),
        "local_deviation_q99": float(q(local_deviation, 0.99)),
        "local_deviation_q995": float(q(local_deviation, 0.995)),
        "neighbour_difference_median": float(neighbour_difference.median()) if not neighbour_difference.empty else np.nan,
        "neighbour_difference_q95": float(q(neighbour_difference, 0.95)),
        "neighbour_difference_q99": float(q(neighbour_difference, 0.99)),
    }

def calculate_local_deviation_series(series: pd.Series) -> pd.DataFrame:
    """
    Calculate timestamp-wise local deviation values.

    This is used by behavioural_checks.py to flag actual spike/dip timestamps.
    """
    s = get_valid_record(series).dropna()

    if len(s) < 3:
        return pd.DataFrame()

    prev_q = s.shift(1)
    next_q = s.shift(-1)

    local_reference = (prev_q + next_q) / 2
    local_reference = local_reference.replace(0, np.nan)

    local_deviation = (s - local_reference).abs() / local_reference
    neighbour_difference = (next_q - prev_q).abs() / local_reference

    out = pd.DataFrame({
        "q": s,
        "local_deviation": local_deviation,
        "neighbour_difference": neighbour_difference,
    })

    return out.replace([np.inf, -np.inf], np.nan)

def calculate_step_score_metrics(series: pd.Series, window: int = 14) -> dict:
    """
    Calculate rolling before/after median difference metrics for step-shift candidates.
    """
    s = get_valid_record(series).dropna()

    if len(s) < window * 2:
        return {}

    before_median = s.rolling(window).median().shift(1)
    after_median = s[::-1].rolling(window).median()[::-1].shift(-1)

    reference = ((before_median.abs() + after_median.abs()) / 2).replace(0, np.nan)
    step_score = (after_median - before_median).abs() / reference
    step_score = step_score.replace([np.inf, -np.inf], np.nan).dropna()

    return {
        f"step_{window}d_score_median": float(step_score.median()) if not step_score.empty else np.nan,
        f"step_{window}d_score_q95": float(q(step_score, 0.95)),
        f"step_{window}d_score_q99": float(q(step_score, 0.99)),
        f"step_{window}d_score_q995": float(q(step_score, 0.995)),
    }

def calculate_step_score_series(
    series: pd.Series,
    window: int = 14,
) -> pd.DataFrame:
    """
    Calculate timestamp-wise step-change scores.

    A step shift should compare relatively stable before/after windows,
    not flood recession limbs.
    """
    s = get_valid_record(series).dropna()

    if len(s) < window * 3:
        return pd.DataFrame()

    before_median = s.rolling(window).median().shift(1)
    before_iqr = (
        s.rolling(window).quantile(0.75)
        - s.rolling(window).quantile(0.25)
    ).shift(1)

    after_median = (
        s[::-1]
        .rolling(window)
        .median()[::-1]
        .shift(-1)
    )

    after_iqr = (
        s[::-1]
        .rolling(window).quantile(0.75)[::-1]
        - s[::-1]
        .rolling(window).quantile(0.25)[::-1]
    ).shift(-1)

    future_median = (
        s[::-1]
        .rolling(window)
        .median()[::-1]
        .shift(-(window + 1))
    )

    reference = ((before_median.abs() + after_median.abs()) / 2).replace(0, np.nan)

    step_score = (after_median - before_median).abs() / reference
    before_stability = before_iqr / before_median.abs().replace(0, np.nan)
    after_stability = after_iqr / after_median.abs().replace(0, np.nan)

    out = pd.DataFrame({
        "q": s,
        "before_median": before_median,
        "after_median": after_median,
        "future_median": future_median,
        "step_score": step_score,
        "before_stability": before_stability,
        "after_stability": after_stability,
    })

    return out.replace([np.inf, -np.inf], np.nan)

def calculate_profile(series: pd.Series, series_name: str = "series") -> dict:
    """
    Calculate all whole-series behaviour metrics for one discharge series.
    """
    profile = {"series": series_name}

    profile.update(calculate_summary_metrics(series))
    profile.update(calculate_zero_flow_metrics(series))
    profile.update(calculate_change_metrics(series))
    profile.update(calculate_rolling_metrics(series))
    profile.update(calculate_local_deviation_metrics(series))
    profile.update(calculate_step_score_metrics(series))

    return profile

def calculate_grouped_profile(series: pd.Series, freq: str, label_name: str) -> pd.DataFrame:
    """
    Calculate grouped profile metrics using pandas resampling.
    Examples:
    freq="ME" for monthly
    freq="QE" for quarterly/3-month
    freq="YE" for yearly
    """
    s = get_valid_record(series)

    rows = []

    for period_end, group in s.resample(freq):
        valid = group.dropna()

        if group.empty:
            continue

        row = {
            label_name: str(period_end.date()),
            "count": int(len(group)),
            "valid_count": int(valid.count()),
            "missing_count": int(group.isna().sum()),
            "missing_ratio": float(group.isna().mean()),
        }

        if valid.empty:
            row.update({
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "cv": np.nan,
                "iqr": np.nan,
                "q05": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "q95": np.nan,
                "zero_ratio": np.nan,
            })
        else:
            rounded = valid.round(3)

            row.update({
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "median": float(valid.median()),
                "std": float(valid.std()),
                "cv": float(safe_cv(valid)),
                "iqr": float(q(valid, 0.75) - q(valid, 0.25)),
                "q05": float(q(valid, 0.05)),
                "q25": float(q(valid, 0.25)),
                "q75": float(q(valid, 0.75)),
                "q95": float(q(valid, 0.95)),
                "zero_ratio": float((rounded == 0).mean()),
            })

        rows.append(row)

    return pd.DataFrame(rows)

def calculate_rolling_variability_series(
    series: pd.Series,
    window: int = 7,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Calculate rolling variability time series for behavioural checks.

    This returns per-timestamp rolling metrics, not just summary statistics.
    """
    s = get_valid_record(series).dropna()

    if s.empty:
        return pd.DataFrame()

    rounded = s.round(decimals)

    rolling_mean = rounded.rolling(window).mean()
    rolling_std = rounded.rolling(window).std()
    rolling_cv = rolling_std / rolling_mean.replace(0, np.nan)
    rolling_range = rounded.rolling(window).max() - rounded.rolling(window).min()

    out = pd.DataFrame({
        "q": rounded,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "rolling_cv": rolling_cv,
        "rolling_range": rolling_range,
    })

    out["month"] = out.index.month
    out["is_zero_flow_window"] = out["rolling_mean"].round(decimals) == 0

    return out.replace([np.inf, -np.inf], np.nan)


def calculate_monthly_profile(series: pd.Series) -> pd.DataFrame:
    """Calculate monthly behaviour metrics."""
    return calculate_grouped_profile(series, freq="ME", label_name="month_end")


def calculate_seasonal_profile(series: pd.Series) -> pd.DataFrame:
    """Calculate 3-month / quarterly behaviour metrics."""
    return calculate_grouped_profile(series, freq="QE", label_name="season_end")


def calculate_yearly_profile(series: pd.Series) -> pd.DataFrame:
    """Calculate yearly behaviour metrics."""
    return calculate_grouped_profile(series, freq="YE", label_name="year_end")

def calculate_monthly_by_year_profile(series: pd.Series) -> pd.DataFrame:
    """
    Calculate month-by-year metrics for gradual drift diagnostics.
    Example: January median for each year, February median for each year, etc.
    """
    s = get_valid_record(series)

    if s.empty:
        return pd.DataFrame()

    df = pd.DataFrame({"q": s})
    df["year"] = df.index.year
    df["month"] = df.index.month

    rows = []

    for (year, month), group in df.groupby(["year", "month"]):
        values = group["q"]
        valid = values.dropna()

        row = {
            "year": int(year),
            "month": int(month),
            "count": int(len(values)),
            "valid_count": int(valid.count()),
            "missing_count": int(values.isna().sum()),
            "missing_ratio": float(values.isna().mean()),
        }

        if valid.empty:
            row.update({
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "cv": np.nan,
                "q05": np.nan,
                "q95": np.nan,
            })
        else:
            row.update({
                "mean": float(valid.mean()),
                "median": float(valid.median()),
                "std": float(valid.std()),
                "cv": float(safe_cv(valid)),
                "q05": float(q(valid, 0.05)),
                "q95": float(q(valid, 0.95)),
            })

        rows.append(row)

    return pd.DataFrame(rows)



def calculate_all_profiles(series: pd.Series, series_name: str = "series") -> dict:
    """
    Calculate all behaviour profiles for one discharge time series.
    """
    return {
        "summary": calculate_profile(series, series_name=series_name),
        "monthly": calculate_monthly_profile(series),
        "seasonal": calculate_seasonal_profile(series),
        "yearly": calculate_yearly_profile(series),
        "monthly_by_year": calculate_monthly_by_year_profile(series),
    }