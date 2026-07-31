"""
TriHydrA Layer 2: OBS-ML Signature Comparison
=============================================

Dynamic, CSV-driven comparison of observed and ML-modelled discharge. This
module depends only on pandas/numpy and `layer2_hydrological_signatures.py`
placed in the same Python environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from layer2_hydrological_signatures import (
    SignatureResult,
    calculate_all_hydrological_signatures,
    calculate_high_flow_signatures,
    calculate_low_flow_signatures,
    calculate_threshold_event_hydrographs,
    signatures_to_summary_table,
)
from comparison_preparation import prepare_layer2_inputs
from display_format import format_display_number
from src.trihydra.layer1.behaviour_profile import calculate_profile



_DATE_EXACT_CANDIDATES = ("date", "datetime", "time", "timestamp")

_CIRCULAR_MONTH_METRICS = {
    "wettest_month",
    "driest_month",
    "dominant_peak_month",
    "circular_mean_peak_month",
}

# Metrics that are just echoed input settings (thresholds, percentiles,
# coverage requirements, filter parameters) rather than genuine
# OBS-vs-ML comparables. OBS and ML are run with the same settings, so
# these always show zero difference and only add clutter to a table
# that's already 50+ rows once every signature group is included.
_SETTINGS_ECHO_METRICS = {
    "zero_threshold",
    "minimum_month_coverage",
    "rolling_window_days",
    "threshold_percentile",
    "alpha",
    "passes",
    "smoothing_window_days",
    "min_consecutive_rising_days",
    "low_flow_percentile",
    "high_flow_percentile",
}
_OBS_EXACT_CANDIDATES = (
    "obs",
    "observed",
    "observation",
    "observations",
    "observed_m3s",
    "observed_discharge",
)
_ML_EXACT_CANDIDATES = (
    "ml",
    "sim",
    "simulation",
    "simulated",
    "simulated_m3s",
    "modelled",
    "modeled",
    "forecast",
    "prediction",
)


def _detect_column(
    columns: Iterable[str],
    exact_candidates: Iterable[str],
    contains_candidates: Iterable[str],
    role: str,
) -> str:
    """Detect one date/OBS/ML column and reject ambiguous matches."""
    original = list(columns)
    normalised = {str(column).lower().strip(): str(column) for column in original}

    for candidate in exact_candidates:
        if candidate in normalised:
            return normalised[candidate]

    matches = [
        str(column)
        for column in original
        if any(
            keyword in str(column).lower()
            for keyword in contains_candidates
        )
    ]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"No {role} column was detected. Available columns: {original}"
        )
    raise ValueError(
        f"Several possible {role} columns were detected: {matches}. "
        f"Pass {role}_column explicitly."
    )


def _prepare_series(
    series: pd.Series,
    name: str,
) -> pd.Series:
    """Prepare one discharge series while preserving internal NaN values."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")

    prepared = series.copy()
    prepared.index = pd.to_datetime(prepared.index, errors="coerce")
    prepared = prepared.loc[~prepared.index.isna()]
    prepared = prepared.sort_index()

    if prepared.index.has_duplicates:
        prepared = prepared.groupby(level=0).median()

    prepared = pd.to_numeric(prepared, errors="coerce").astype(float)
    prepared.name = name

    first_valid = prepared.first_valid_index()
    last_valid = prepared.last_valid_index()

    if first_valid is None or last_valid is None:
        return prepared.iloc[0:0]

    return prepared.loc[first_valid:last_valid]


def load_obs_ml_csv(
    csv_path: str | Path,
    date_column: Optional[str] = None,
    obs_column: Optional[str] = None,
    ml_column: Optional[str] = None,
    latest_years: Optional[float] = None,
    reindex_daily: bool = True,
    **read_csv_kwargs: Any,
) -> tuple[pd.Series, pd.Series]:
    """
    Load OBS and ML series from one CSV without a hard-coded basin or path.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    frame = pd.read_csv(path, **read_csv_kwargs)

    if date_column is None:
        date_column = _detect_column(
            frame.columns,
            _DATE_EXACT_CANDIDATES,
            ("date", "time"),
            "date",
        )
    if obs_column is None:
        obs_column = _detect_column(
            frame.columns,
            _OBS_EXACT_CANDIDATES,
            ("observed", "observation", "obs"),
            "obs",
        )
    if ml_column is None:
        ml_column = _detect_column(
            frame.columns,
            _ML_EXACT_CANDIDATES,
            ("simulated", "simulation", "modelled", "modeled", "forecast", "ml"),
            "ml",
        )

    required = {date_column, obs_column, ml_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {frame.columns.tolist()}"
        )

    frame = frame[[date_column, obs_column, ml_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[obs_column] = pd.to_numeric(frame[obs_column], errors="coerce")
    frame[ml_column] = pd.to_numeric(frame[ml_column], errors="coerce")
    frame = frame.dropna(subset=[date_column]).sort_values(date_column)

    grouped = frame.groupby(date_column, sort=True)[[obs_column, ml_column]].median()

    obs = grouped[obs_column].rename("obs")
    ml = grouped[ml_column].rename("ml")

    if latest_years is not None:
        if latest_years <= 0:
            raise ValueError("latest_years must be positive.")
        last_valid_candidates = [
            item
            for item in (obs.last_valid_index(), ml.last_valid_index())
            if item is not None
        ]
        if last_valid_candidates:
            common_last = min(last_valid_candidates)
            start = common_last - pd.DateOffset(
                days=int(round(float(latest_years) * 365.25))
            )
            obs = obs.loc[start:common_last]
            ml = ml.loc[start:common_last]

    if reindex_daily and (not obs.empty or not ml.empty):
        start_candidates = [
            item for item in (obs.index.min(), ml.index.min()) if pd.notna(item)
        ]
        end_candidates = [
            item for item in (obs.index.max(), ml.index.max()) if pd.notna(item)
        ]
        if start_candidates and end_candidates:
            full_index = pd.date_range(
                min(start_candidates),
                max(end_candidates),
                freq="D",
            )
            obs = obs.reindex(full_index)
            ml = ml.reindex(full_index)
            obs.name = "obs"
            ml.name = "ml"

    return obs.astype(float), ml.astype(float)


def prepare_obs_ml_series(
    obs_series: pd.Series,
    ml_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Trim OBS and ML to their overlapping valid period and use the OBS calendar.
    """
    obs = _prepare_series(obs_series, "obs")
    ml = _prepare_series(ml_series, "ml")

    if obs.empty or ml.empty:
        raise ValueError("OBS or ML has no valid data after trimming.")

    common_start = max(obs.index.min(), ml.index.min())
    common_end = min(obs.index.max(), ml.index.max())

    if common_start > common_end:
        raise ValueError("OBS and ML series do not overlap in time.")

    obs_aligned = obs.loc[common_start:common_end]
    ml_aligned = ml.loc[common_start:common_end].reindex(obs_aligned.index)

    if obs_aligned.empty:
        raise ValueError("OBS and ML could not be aligned on common dates.")

    return obs_aligned.rename("obs"), ml_aligned.rename("ml")


def fill_obs_for_layer2(
    obs_aligned: pd.Series,
    method: str = "none",
    window_days: int = 15,
    min_samples: int = 5,
) -> pd.Series:
    """
    Optionally fill internal OBS gaps on a temporary copy.

    `method="none"` is the safe default. The seasonal-climatology option
    preserves the notebook method and should be used only when explicitly
    required.
    """
    obs_l2 = _prepare_series(obs_aligned, "obs")

    if method == "none":
        return obs_l2
    if method == "ffill":
        return obs_l2.ffill()
    if method != "seasonal_climatology":
        raise ValueError(
            "method must be 'none', 'seasonal_climatology', or 'ffill'."
        )
    if window_days < 0:
        raise ValueError("window_days cannot be negative.")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1.")

    valid = obs_l2.dropna()
    if valid.empty:
        return obs_l2

    valid_doy = valid.index.dayofyear.to_numpy()
    valid_values = valid.to_numpy(dtype=float)

    for missing_date in obs_l2.index[obs_l2.isna()]:
        target = int(missing_date.dayofyear)
        distance = np.abs(valid_doy - target)
        circular_distance = np.minimum(distance, 366 - distance)
        candidates = valid_values[circular_distance <= window_days]

        if len(candidates) >= min_samples:
            obs_l2.loc[missing_date] = float(np.median(candidates))

    return obs_l2


def calculate_summary_profile(
    series: pd.Series,
    series_name: str = "discharge",
) -> dict[str, Any]:
    """Create the compact whole-record profile used by the old notebook."""
    prepared = _prepare_series(series, series_name)
    valid = prepared.dropna()

    if valid.empty:
        return {
            "series_name": series_name,
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "q05": np.nan,
            "q95": np.nan,
            "zero_ratio": np.nan,
        }

    return {
        "series_name": series_name,
        "count": int(len(valid)),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "minimum": float(valid.min()),
        "maximum": float(valid.max()),
        "q05": float(valid.quantile(0.05)),
        "q95": float(valid.quantile(0.95)),
        "zero_ratio": float((valid <= 1e-6).mean()),
    }


def calculate_grouped_profile(
    series: pd.Series,
    frequency: str,
) -> pd.DataFrame:
    """Calculate count/mean/median/min/max/std for regular time groups."""
    prepared = _prepare_series(series, series.name or "discharge")

    if prepared.empty:
        return pd.DataFrame(
            columns=["period_start", "count", "mean", "median", "min", "max", "std"]
        )

    grouped = prepared.resample(frequency).agg(
        ["count", "mean", "median", "min", "max", "std"]
    )
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={grouped.columns[0]: "period_start"})
    return grouped


def calculate_monthly_by_year_profile(
    series: pd.Series,
) -> pd.DataFrame:
    """Calculate monthly statistics with explicit year and month columns."""
    profile = calculate_grouped_profile(series, "MS")
    if profile.empty:
        profile["year"] = pd.Series(dtype=int)
        profile["month"] = pd.Series(dtype=int)
        return profile

    profile["year"] = pd.to_datetime(profile["period_start"]).dt.year
    profile["month"] = pd.to_datetime(profile["period_start"]).dt.month
    return profile


def calculate_layer2_profiles(
    obs_l2: pd.Series,
    ml_aligned: pd.Series,
) -> dict[str, Any]:
    """
    Reproduce the notebook's whole/monthly/seasonal/yearly profile bundle.
    """
    return {
        "obs_summary": calculate_summary_profile(obs_l2, "obs"),
        "ml_summary": calculate_summary_profile(ml_aligned, "ml"),
        "obs_monthly": calculate_grouped_profile(obs_l2, "MS"),
        "ml_monthly": calculate_grouped_profile(ml_aligned, "MS"),
        "obs_seasonal": calculate_grouped_profile(obs_l2, "QS-DEC"),
        "ml_seasonal": calculate_grouped_profile(ml_aligned, "QS-DEC"),
        "obs_yearly": calculate_grouped_profile(obs_l2, "YS"),
        "ml_yearly": calculate_grouped_profile(ml_aligned, "YS"),
        "obs_monthly_by_year": calculate_monthly_by_year_profile(obs_l2),
        "ml_monthly_by_year": calculate_monthly_by_year_profile(ml_aligned),
    }


def extract_signature_profile(
    summary_profile: dict[str, Any],
    monthly_profile: pd.DataFrame,
    yearly_profile: pd.DataFrame,
) -> dict[str, Any]:
    """
    Reproduce the original seven-signature catalogue from behaviour profiles.
    """
    low_flow_q05 = summary_profile.get("q05", np.nan)
    high_flow_q95 = summary_profile.get("q95", np.nan)
    zero_ratio = summary_profile.get("zero_ratio", np.nan)

    annual_max_median = (
        float(yearly_profile["max"].median())
        if not yearly_profile.empty and "max" in yearly_profile
        else np.nan
    )

    if monthly_profile.empty or "median" not in monthly_profile:
        seasonality_index = np.nan
        wettest_month = None
        driest_month = None
    else:
        monthly = monthly_profile.copy()
        date_column = (
            "period_start"
            if "period_start" in monthly
            else "month_end"
            if "month_end" in monthly
            else None
        )
        if date_column is None:
            seasonality_index = np.nan
            wettest_month = None
            driest_month = None
        else:
            monthly["month"] = pd.to_datetime(monthly[date_column]).dt.month
            climatology = (
                monthly.dropna(subset=["median"])
                .groupby("month")["median"]
                .median()
            )
            if climatology.empty:
                seasonality_index = np.nan
                wettest_month = None
                driest_month = None
            else:
                wettest_month = int(climatology.idxmax())
                driest_month = int(climatology.idxmin())
                seasonality_index = (
                    float(climatology.max() / climatology.min())
                    if climatology.min() != 0
                    else np.nan
                )

    return {
        "low_flow_q05": float(low_flow_q05)
        if pd.notna(low_flow_q05) else np.nan,
        "high_flow_q95": float(high_flow_q95)
        if pd.notna(high_flow_q95) else np.nan,
        "zero_ratio": float(zero_ratio)
        if pd.notna(zero_ratio) else np.nan,
        "annual_max_median": annual_max_median,
        "seasonality_index": seasonality_index,
        "wettest_month": wettest_month,
        "driest_month": driest_month,
    }


def calculate_obs_ml_signature_results(
    obs_series: pd.Series,
    ml_series: Optional[pd.Series] = None,
    fill_method: str = "seasonal_climatology",
    fill_window_days: int = 15,
    fill_min_samples: int = 5,
    use_obs_thresholds_for_ml: bool = True,
    signature_kwargs: Optional[dict[str, Any]] = None,
    layer1_obs_profile: Optional[dict[str, Any]] = None,
    discharge_unit: str = "source units",
) -> dict[str, Any]:
    """
    Align OBS/ML and calculate the complete signature set for both.

    When `use_obs_thresholds_for_ml=True`, ML low/high-flow frequency and
    threshold-event metrics are recalculated using OBS-derived thresholds.
    This makes those frequency comparisons genuinely comparable.
    """
    signature_kwargs = (
        {} if signature_kwargs is None else dict(signature_kwargs)
    )

    prepared = prepare_layer2_inputs(
        obs_series,
        ml_series,
        fill_method=fill_method,
        window_days=fill_window_days,
        min_samples=fill_min_samples,
    )
    obs_aligned = prepared.obs_aligned
    ml_aligned = prepared.model_aligned
    obs_analysis = prepared.obs_analysis

    # Low/high-flow definitions must originate from raw Layer 1 information,
    # never from temporarily imputed Layer 2 values. If the caller has not
    # passed the Layer 1 profile, use the same Layer 1 profile function on the
    # raw aligned OBS record as an explicit compatibility fallback.
    if layer1_obs_profile is None:
        threshold_profile = calculate_profile(
            prepared.obs_aligned, series_name="obs"
        )
        threshold_profile_source = "Layer 1 behaviour profile fallback"
    else:
        threshold_profile = dict(layer1_obs_profile)
        threshold_profile_source = "supplied Layer 1 behaviour profile"

    low_threshold = threshold_profile.get("q05", np.nan)
    high_threshold = threshold_profile.get("q95", np.nan)
    threshold_provenance = {
        "source": threshold_profile_source,
        "input_values": "raw valid OBS only; temporary fills excluded",
        "period_start": prepared.coverage["common_start"],
        "period_end": prepared.coverage["common_end"],
        "raw_valid_observation_count": int(prepared.obs_aligned.notna().sum()),
        "unit": discharge_unit,
        "low_flow_definition": "Q <= raw OBS Q05",
        "low_flow_percentile": 0.05,
        "low_flow_threshold": low_threshold,
        "high_flow_definition": "Q >= raw OBS Q95",
        "high_flow_percentile": 0.95,
        "high_flow_threshold": high_threshold,
    }
    signature_kwargs["low_flow_threshold"] = low_threshold
    signature_kwargs["high_flow_threshold"] = high_threshold

    obs_results = calculate_all_hydrological_signatures(
        obs_analysis,
        **signature_kwargs,
    )
    ml_results = (
        None
        if prepared.model_analysis is None
        else calculate_all_hydrological_signatures(
            prepared.model_analysis, **signature_kwargs
        )
    )

    for result_set in (obs_results, ml_results):
        if result_set is None:
            continue
        for group in ("low_flow", "high_flow", "threshold_event_hydrographs"):
            result_set[group].metadata["threshold_provenance"] = dict(
                threshold_provenance
            )

    summaries = [signatures_to_summary_table(obs_results, "obs")]
    if ml_results is not None:
        summaries.append(signatures_to_summary_table(ml_results, "model"))
    combined_summary = pd.concat(summaries, ignore_index=True)

    return {
        "mode": prepared.mode,
        "coverage": prepared.coverage,
        "imputation_log": prepared.imputation_log,
        "threshold_provenance": threshold_provenance,
        "obs_aligned": obs_aligned,
        "ml_aligned": ml_aligned,
        "obs_analysis": obs_analysis,
        "ml_analysis": prepared.model_analysis,
        "obs_results": obs_results,
        "ml_results": ml_results,
        "summary": combined_summary,
    }


def extract_compact_signature_profile(
    results: dict[str, SignatureResult],
) -> dict[str, Any]:
    """Extract a dashboard-sized profile from complete signature results."""
    magnitude = results["flow_magnitude"].metrics
    zero = results["zero_flow"].metrics
    baseflow = results["baseflow"].metrics
    annual_maximum = results["annual_maximum"].metrics
    seasonality = results["seasonality"].metrics
    flashiness = results["flashiness"].metrics
    autocorrelation = results["autocorrelation"].metrics
    events = results["threshold_event_hydrographs"].metrics
    peaks = results["peaks"].metrics

    return {
        "mean_flow": magnitude.get("mean_flow", np.nan),
        "median_flow": magnitude.get("median_flow", np.nan),
        "low_flow_q05": magnitude.get("q05", np.nan),
        "high_flow_q95": magnitude.get("q95", np.nan),
        "zero_ratio": zero.get("zero_flow_ratio", np.nan),
        "baseflow_index": baseflow.get("baseflow_index", np.nan),
        "annual_max_median": annual_maximum.get(
            "median_annual_maximum", np.nan
        ),
        "seasonality_index": seasonality.get(
            "seasonality_index_max_to_min", np.nan
        ),
        "walsh_lawler_seasonality_index": seasonality.get(
            "walsh_lawler_seasonality_index", np.nan
        ),
        "wettest_month": seasonality.get("wettest_month", None),
        "driest_month": seasonality.get("driest_month", None),
        "flashiness_index": flashiness.get(
            "whole_record_flashiness_index", np.nan
        ),
        "lag1_autocorrelation": autocorrelation.get(
            "autocorrelation_lag_1", np.nan
        ),
        "peak_frequency_per_year": peaks.get(
            "peak_frequency_per_year", np.nan
        ),
        "event_count": events.get("event_count", np.nan),
        "median_rising_limb_rate": events.get(
            "median_rising_limb_rate", np.nan
        ),
        "median_recession_limb_slope": events.get(
            "median_recession_limb_slope", np.nan
        ),
        "median_log_recession_slope": events.get(
            "median_log_recession_slope", np.nan
        ),
        "median_time_to_peak_days": events.get(
            "median_time_to_peak_days", np.nan
        ),
    }


# One representative, interpretable result for each of the 13 Layer 2
# signature groups. Detailed scalar metrics remain available separately.
_PRIMARY_SIGNATURE_METRICS = {
    "flow_magnitude": ("mean_flow", "Mean discharge"),
    "low_flow": ("low_flow_frequency", "Low-flow frequency"),
    "high_flow": ("high_flow_frequency", "High-flow frequency"),
    "annual_maximum": ("median_annual_maximum", "Median annual maximum"),
    "zero_flow": ("zero_flow_ratio", "Zero-flow proportion"),
    "baseflow": ("baseflow_index", "Baseflow index"),
    "flashiness": ("whole_record_flashiness_index", "Flashiness index"),
    "autocorrelation": ("autocorrelation_lag_1", "Lag-1 autocorrelation"),
    "rising_limb": ("median_rising_rate", "Median rising-limb rate"),
    "recession_limb": ("median_recession_rate", "Median recession-limb rate"),
    "peaks": ("peak_frequency_per_year", "Peak frequency per year"),
    "seasonality": (
        "walsh_lawler_seasonality_index",
        "Walsh-Lawler seasonality index",
    ),
    "threshold_event_hydrographs": (
        "event_count",
        "Threshold-event count",
    ),
}


def build_primary_signature_comparison(
    obs_results: dict[str, SignatureResult],
    model_results: Optional[dict[str, SignatureResult]] = None,
    model_name: str = "ML",
    coverage: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Build the user-facing diagnostic table with exactly 13 rows.

    Each row represents one signature group. OBS-only runs retain the same
    schema but leave model comparison fields empty. In comparison mode,
    direction and both absolute and percentage magnitude are reported; this
    makes statements such as "annual maximum is higher by ..." explicit.
    """
    rows = []
    coverage = {} if coverage is None else coverage
    for group, (metric, label) in _PRIMARY_SIGNATURE_METRICS.items():
        obs_result = obs_results[group]
        obs_value = obs_result.metrics.get(metric, np.nan)
        model_result = None if model_results is None else model_results[group]
        model_value = (
            np.nan if model_result is None else model_result.metrics.get(metric, np.nan)
        )
        if pd.notna(obs_value) and pd.notna(model_value):
            difference = float(model_value) - float(obs_value)
            percentage = relative_difference_percent(obs_value, model_value)
            direction = (
                "equal" if difference == 0 else ("higher" if difference > 0 else "lower")
            )
            magnitude = format_display_number(abs(difference))
            if pd.notna(percentage):
                assessment = (
                    f"{model_name} is {direction} than OBS by {magnitude} "
                    f"({format_display_number(abs(percentage))}%)."
                )
            else:
                assessment = f"{model_name} is {direction} than OBS by {magnitude}."
        else:
            difference, percentage, direction = np.nan, np.nan, None
            assessment = (
                "OBS signature calculated; no model input supplied."
                if model_results is None
                else "Insufficient data for this comparison."
            )
        rows.append(
            {
                "signature_group": group,
                "diagnostic_signature": label,
                "representative_metric": metric,
                "obs_value": obs_value,
                f"{model_name}_value": model_value,
                "absolute_difference": difference,
                "relative_difference_percent": percentage,
                "model_direction": direction,
                "assessment": assessment,
                "obs_status": obs_result.status,
                "model_status": None if model_result is None else model_result.status,
                "analysis_start": coverage.get("common_start"),
                "analysis_end": coverage.get("common_end"),
                "temporary_imputation_used": bool(
                    coverage.get("obs_temporarily_filled", 0)
                    or coverage.get("model_temporarily_filled", 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def relative_difference_percent(
    obs_value: Any,
    ml_value: Any,
) -> float:
    """Calculate ((ML - OBS) / OBS) * 100."""
    if pd.isna(obs_value) or pd.isna(ml_value) or obs_value == 0:
        return np.nan
    return float(((ml_value - obs_value) / obs_value) * 100)


def circular_month_difference(
    obs_month: Any,
    ml_month: Any,
) -> float:
    """
    Return the shortest separation between two month-like values, on a
    12-month wheel (December to January is a distance of 1, not 11).

    Works for both discrete calendar months (1-12) and continuous
    circular-mean month values (e.g. 6.7), so it applies to
    wettest_month/driest_month as well as circular_mean_peak_month.
    """
    if pd.isna(obs_month) or pd.isna(ml_month):
        return np.nan
    direct = abs(float(ml_month) - float(obs_month))
    return float(min(direct, 12 - direct))


def _diagnostic_message(
    signature: str,
    obs_value: Any,
    ml_value: Any,
) -> str:
    """Create direction-aware diagnostic text without calling it an error."""
    if pd.isna(obs_value) or pd.isna(ml_value):
        return "Insufficient data for comparison"

    if signature in _CIRCULAR_MONTH_METRICS:
        if int(obs_value) == int(ml_value):
            return f"ML matches OBS {signature.replace('_', ' ')}"
        return f"ML {signature.replace('_', ' ')} differs from OBS"

    if ml_value == obs_value:
        return "ML matches OBS"

    direction = "higher" if ml_value > obs_value else "lower"

    labels = {
        "mean_flow": "mean flow",
        "median_flow": "median flow",
        "low_flow_q05": "low-flow magnitude",
        "high_flow_q95": "high-flow magnitude",
        "zero_ratio": "zero-flow frequency",
        "baseflow_index": "baseflow index",
        "annual_max_median": "annual extremes",
        "seasonality_index": "seasonal contrast",
        "walsh_lawler_seasonality_index": "Walsh-Lawler seasonality",
        "flashiness_index": "flashiness",
        "lag1_autocorrelation": "short-term persistence",
        "peak_frequency_per_year": "peak frequency",
        "event_count": "threshold-event count",
        "median_rising_limb_rate": "rising-limb rate",
        "median_recession_limb_slope": "recession-limb slope",
        "median_log_recession_slope": "log recession slope",
        "median_time_to_peak_days": "time to peak",
    }
    label = labels.get(signature, signature.replace("_", " "))
    return f"ML {label} is {direction} than OBS"


def compare_signature_profiles(
    basin_id: str,
    obs_signature_profile: dict[str, Any],
    ml_signature_profile: dict[str, Any],
    relative_tolerance_percent: float = 0.0,
    month_tolerance: int = 0,
) -> pd.DataFrame:
    """
    Compare compact profiles and flag differences outside chosen tolerances.

    A zero relative tolerance reproduces the original behaviour where any
    non-zero difference is diagnostic.
    """
    if relative_tolerance_percent < 0:
        raise ValueError("relative_tolerance_percent cannot be negative.")
    if month_tolerance < 0:
        raise ValueError("month_tolerance cannot be negative.")

    names = list(
        dict.fromkeys(
            list(obs_signature_profile)
            + list(ml_signature_profile)
        )
    )

    rows: list[dict[str, Any]] = []

    for signature in names:
        obs_value = obs_signature_profile.get(signature, np.nan)
        ml_value = ml_signature_profile.get(signature, np.nan)

        if pd.isna(obs_value) or pd.isna(ml_value):
            difference = np.nan
            relative_difference = np.nan
            comparison_distance = np.nan
            flag = False
        elif signature in _CIRCULAR_MONTH_METRICS:
            difference = float(ml_value - obs_value)
            relative_difference = np.nan
            comparison_distance = circular_month_difference(
                obs_value,
                ml_value,
            )
            flag = comparison_distance > month_tolerance
        else:
            difference = float(ml_value - obs_value)
            relative_difference = relative_difference_percent(
                obs_value,
                ml_value,
            )
            comparison_distance = abs(relative_difference)
            if pd.isna(comparison_distance):
                flag = difference != 0
            else:
                flag = comparison_distance > relative_tolerance_percent

        rows.append(
            {
                "basin_id": basin_id,
                "signature": signature,
                "obs_value": obs_value,
                "ml_value": ml_value,
                "difference": difference,
                "relative_difference_percent": relative_difference,
                "comparison_distance": comparison_distance,
                "diagnostic": _diagnostic_message(
                    signature,
                    obs_value,
                    ml_value,
                ),
                "flag": bool(flag),
            }
        )

    return pd.DataFrame(rows)


def compare_all_scalar_metrics(
    basin_id: str,
    obs_results: dict[str, SignatureResult],
    ml_results: dict[str, SignatureResult],
    relative_tolerance_percent: float = 0.0,
    month_tolerance: int = 0,
    exclude_settings_echo: bool = True,
) -> pd.DataFrame:
    """
    Compare every common scalar metric produced by the signature module.

    Two fixes versus a naive generic comparison:
    - `wettest_month`/`driest_month` (and anything else in
      _CIRCULAR_MONTH_METRICS) use circular month distance, the same
      as the compact comparison, instead of a raw percentage
      difference that would make December vs. January look like a
      huge mismatch instead of "1 month apart".
    - Metrics that are just echoed input settings (thresholds,
      percentiles, filter parameters -- see _SETTINGS_ECHO_METRICS)
      are excluded by default, since OBS and ML are run with the same
      settings and these always show zero difference.
    """
    obs = signatures_to_summary_table(obs_results, "obs").rename(
        columns={
            "value": "obs_value",
            "status": "obs_status",
        }
    )
    ml = signatures_to_summary_table(ml_results, "ml").rename(
        columns={
            "value": "ml_value",
            "status": "ml_status",
        }
    )

    merged = obs.merge(
        ml,
        on=["signature_group", "metric"],
        how="outer",
    )

    if exclude_settings_echo:
        merged = merged[~merged["metric"].isin(_SETTINGS_ECHO_METRICS)]

    rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        obs_value = row.get("obs_value", np.nan)
        ml_value = row.get("ml_value", np.nan)
        metric = row["metric"]

        if metric in _CIRCULAR_MONTH_METRICS:
            if pd.isna(obs_value) or pd.isna(ml_value):
                difference = np.nan
                relative_difference = np.nan
                comparison_distance = np.nan
                flag = False
            else:
                difference = float(ml_value) - float(obs_value)
                relative_difference = np.nan
                comparison_distance = circular_month_difference(
                    obs_value,
                    ml_value,
                )
                flag = comparison_distance > month_tolerance
        else:
            try:
                obs_number = float(obs_value)
                ml_number = float(ml_value)
            except (TypeError, ValueError):
                obs_number = np.nan
                ml_number = np.nan

            difference = (
                ml_number - obs_number
                if pd.notna(obs_number) and pd.notna(ml_number)
                else np.nan
            )
            relative_difference = relative_difference_percent(
                obs_number,
                ml_number,
            )
            comparison_distance = (
                abs(relative_difference)
                if pd.notna(relative_difference) else np.nan
            )
            if pd.isna(comparison_distance):
                flag = pd.notna(difference) and difference != 0
            else:
                flag = comparison_distance > relative_tolerance_percent

        rows.append(
            {
                "basin_id": basin_id,
                "signature_group": row["signature_group"],
                "metric": metric,
                "obs_value": obs_value,
                "ml_value": ml_value,
                "difference": difference,
                "relative_difference_percent": relative_difference,
                "comparison_distance": comparison_distance,
                "obs_status": row.get("obs_status"),
                "ml_status": row.get("ml_status"),
                "flag": bool(flag),
            }
        )

    return pd.DataFrame(rows)


def run_obs_ml_comparison_from_csv(
    csv_path: str | Path,
    date_column: Optional[str] = None,
    obs_column: Optional[str] = None,
    ml_column: Optional[str] = None,
    basin_id: Optional[str] = None,
    latest_years: Optional[float] = None,
    fill_method: str = "none",
    relative_tolerance_percent: float = 0.0,
    month_tolerance: int = 0,
    use_obs_thresholds_for_ml: bool = True,
    loader_kwargs: Optional[dict[str, Any]] = None,
    signature_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    One dynamic entry point for a CSV containing OBS and ML columns.
    """
    loader_kwargs = {} if loader_kwargs is None else dict(loader_kwargs)

    obs, ml = load_obs_ml_csv(
        csv_path=csv_path,
        date_column=date_column,
        obs_column=obs_column,
        ml_column=ml_column,
        latest_years=latest_years,
        **loader_kwargs,
    )

    calculated = calculate_obs_ml_signature_results(
        obs_series=obs,
        ml_series=ml,
        fill_method=fill_method,
        use_obs_thresholds_for_ml=use_obs_thresholds_for_ml,
        signature_kwargs=signature_kwargs,
    )

    obs_profile = extract_compact_signature_profile(
        calculated["obs_results"]
    )
    ml_profile = extract_compact_signature_profile(
        calculated["ml_results"]
    )

    resolved_basin_id = (
        str(basin_id)
        if basin_id is not None
        else Path(csv_path).stem
    )

    compact_comparison = compare_signature_profiles(
        basin_id=resolved_basin_id,
        obs_signature_profile=obs_profile,
        ml_signature_profile=ml_profile,
        relative_tolerance_percent=relative_tolerance_percent,
        month_tolerance=month_tolerance,
    )

    full_comparison = compare_all_scalar_metrics(
        basin_id=resolved_basin_id,
        obs_results=calculated["obs_results"],
        ml_results=calculated["ml_results"],
        relative_tolerance_percent=relative_tolerance_percent,
    )

    return {
        **calculated,
        "obs_profile": obs_profile,
        "ml_profile": ml_profile,
        "compact_comparison": compact_comparison,
        "full_comparison": full_comparison,
    }


__all__ = [
    "load_obs_ml_csv",
    "prepare_obs_ml_series",
    "fill_obs_for_layer2",
    "calculate_summary_profile",
    "calculate_grouped_profile",
    "calculate_monthly_by_year_profile",
    "calculate_layer2_profiles",
    "extract_signature_profile",
    "calculate_obs_ml_signature_results",
    "extract_compact_signature_profile",
    "build_primary_signature_comparison",
    "relative_difference_percent",
    "circular_month_difference",
    "compare_signature_profiles",
    "compare_all_scalar_metrics",
    "run_obs_ml_comparison_from_csv",
]
