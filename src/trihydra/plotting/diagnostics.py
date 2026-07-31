"""
diagnostics.py

All summary-table logic for Layer 1 and Layer 2, in one file.

visualisation.py handles plotting and deliberately does NOT re-run any
check itself -- every plotting function in that file consumes the
already-computed results this file returns. That way, calling
diagnostics then visualisation in the same notebook cell can never
disagree about what a check found, and a missing/renamed result can be
handled with a clear message instead of a crash.

Expects this file to live at <project_root>/src/trihydra/plotting/,
alongside layer1/, layer2/, layer3/, and io/ as siblings under
trihydra/ (matching the actual TriHydrA repo layout).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------
# Path setup.
#
# This file lives at: <project_root>/src/trihydra/plotting/diagnostics.py
# TRIHYDRA_DIR       = <project_root>/src/trihydra          (2 parents up)
# PROJECT_ROOT       = <project_root>                       (4 parents up;
#                       the folder that CONTAINS "src", needed on sys.path
#                       for "from src.trihydra..." imports to resolve)
# LAYER2_DIR         = <project_root>/src/trihydra/layer2   (layer2's own
#                       files use flat imports between each other, so that
#                       folder itself needs to be importable too)
# ----------------------------------------------------------------
TRIHYDRA_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LAYER2_DIR = TRIHYDRA_DIR / "layer2"

for p in (PROJECT_ROOT, LAYER2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.trihydra.layer1.basic_checks import run_basic_checks
from src.trihydra.layer1.behaviour_checks import run_behavioural_checks

from layer2_hydrological_signatures import (
    calculate_all_hydrological_signatures,
    percentile_diagnostic,
)
from layer2_obs_ml_comparison import (
    build_primary_signature_comparison,
    compare_all_scalar_metrics,
    calculate_obs_ml_signature_results,
)


# ==========================================================
# SHARED: check display names
# ==========================================================
# Mentor feedback: "change low flow variability -> low variability",
# and every check should have one clean, consistent printable name.

LAYER1_CHECK_LABELS = {
    "missing_values": "Missing values",
    "long_gaps": "Long gaps",
    "negative_discharge": "Negative discharge",
    "duplicate_timestamps": "Duplicate timestamps",
    "timestep_consistency": "Timestep consistency",
    "zero_flow_regime": "Zero-flow regime",
    "low_variability": "Low variability",
    "spike_dip": "Spike / dip",
    "step_shift": "Step shift",
    "gradual_drift": "Gradual drift",
}

LAYER2_CHECK_LABELS = {
    "flow_magnitude": "Flow magnitude",
    "low_flow": "Low flow",
    "high_flow": "High flow",
    "annual_maximum": "Annual maximum",
    "zero_flow": "Zero flow",
    "baseflow": "Baseflow",
    "seasonality": "Seasonality",
    "flashiness": "Flashiness",
    "autocorrelation": "Autocorrelation",
    "rising_limb": "Rising limb",
    "recession_limb": "Recession limb",
    "peaks": "Peaks",
    "threshold_event_hydrographs": "Threshold-based events",
}


# ==========================================================
# LAYER 1
# ==========================================================

def eda_summary_table(series: pd.Series, series_type: str) -> pd.DataFrame:
    """
    Quick-look EDA row: count, min, q05, median, mean, q95, max --
    the same summary the earliest Layer 1 demo notebook produced,
    kept as its own table since it's useful context before reading
    any check result.
    """
    valid = series.dropna()

    if valid.empty:
        row = {stat: np.nan for stat in ["count", "min", "q05", "median", "mean", "q95", "max"]}
        row["count"] = 0
    else:
        row = {
            "count": int(valid.count()),
            "min": float(valid.min()),
            "q05": float(valid.quantile(0.05)),
            "median": float(valid.median()),
            "mean": float(valid.mean()),
            "q95": float(valid.quantile(0.95)),
            "max": float(valid.max()),
        }

    row["series_type"] = series_type
    row["start_date"] = str(valid.index.min()) if not valid.empty else None
    row["end_date"] = str(valid.index.max()) if not valid.empty else None

    return pd.DataFrame([row])[
        ["series_type", "start_date", "end_date", "count", "min", "q05", "median", "mean", "q95", "max"]
    ]


def _describe_check_value(check_result: dict, record_length: int) -> str:
    """
    Describe the "value" field in plain, consistent terms -- mentor
    feedback: a bare count is ambiguous and means something different
    for every check. Where the value is a count, also report it as a
    percentage of available time steps.
    """
    check = check_result["check"]
    value = check_result.get("value")

    if value is None:
        return "Not calculated"

    def pct_of_record(count: float) -> str:
        if not record_length:
            return ""
        return f" ({count / record_length * 100:.2f}% of available time steps)"

    if check == "missing_values":
        return f"{value:.0f} missing day(s)" + pct_of_record(value)
    if check == "long_gaps":
        return f"longest internal gap = {value:.0f} day(s)"
    if check == "negative_discharge":
        return f"{value:.0f} negative day(s)" + pct_of_record(value)
    if check == "duplicate_timestamps":
        return f"{value:.0f} duplicated timestamp(s)"
    if check == "timestep_consistency":
        return f"{value:.0f} irregular time step(s)"
    if check == "zero_flow_regime":
        return f"zero-flow ratio = {value:.3f} ({value * 100:.1f}% of days)"
    if check == "low_variability":
        return f"{value:.0f} low-variability day(s)" + pct_of_record(value)
    if check == "spike_dip":
        return f"{value:.0f} candidate spike/dip day(s)"
    if check == "step_shift":
        return f"{value:.0f} confirmed regime boundary(ies)"
    if check == "gradual_drift":
        return f"largest detected period change = {value:.3f}%"

    return str(value)


def _run_layer1_checks(
    series: pd.Series,
    series_type: str,
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict]:
    results = run_basic_checks(series, config=config)
    for r in results:
        r.setdefault("check_group", "basic")
        r.setdefault("execution_status", "completed")
        r.setdefault(
            "finding_status",
            "candidate_detected" if r.get("flag") else "passed",
        )
        r.setdefault("reason_skipped", None)
        r["series_type"] = series_type
    results.extend(
        run_behavioural_checks(
            series, series_type=series_type, config=config
        )
    )
    return results


def _layer1_summary_rows(results: list[dict], record_length: int) -> list[dict]:
    rows = []
    for r in results:
        rows.append({
            "series_type": r["series_type"],
            "check": r["check"],
            "check_label": LAYER1_CHECK_LABELS.get(r["check"], r["check"]),
            "check_group": r.get("check_group", "basic"),
            "status": r.get("status"),
            "execution_status": r.get("execution_status", "completed"),
            "finding_status": r.get("finding_status"),
            "reason_skipped": r.get("reason_skipped"),
            "flag": r["flag"],
            "value_description": _describe_check_value(r, record_length),
            "message": r.get("message", ""),
        })
    return rows


def _layer1_detail_tables(raw_results: dict) -> dict[str, pd.DataFrame]:
    """Flatten structured check evidence into human-readable tables."""
    rows_by_table: dict[str, list[dict]] = {
        "missing_intervals": [],
        "long_gaps": [],
        "spike_dip_candidates": [],
        "step_shift_boundaries": [],
        "step_shift_regimes": [],
        "gradual_drift_segments": [],
        "gradual_drift_gaps": [],
        "zero_flow_spells": [],
        "low_variability_periods": [],
    }
    mapping = {
        "missing_values": [("internal_intervals", "missing_intervals")],
        "long_gaps": [("long_gap_intervals", "long_gaps")],
        "spike_dip": [("candidate_details", "spike_dip_candidates")],
        "step_shift": [
            ("regime_boundaries", "step_shift_boundaries"),
            ("regime_summary", "step_shift_regimes"),
        ],
        "gradual_drift": [
            ("drift_segments", "gradual_drift_segments"),
            ("unresolved_gaps", "gradual_drift_gaps"),
        ],
        "zero_flow_regime": [("zero_flow_spells", "zero_flow_spells")],
        "low_variability": [
            ("low_variability_periods", "low_variability_periods"),
        ],
    }
    for series_type, results in raw_results.items():
        for result in results:
            for field, table_name in mapping.get(result["check"], []):
                for item in result.get(field, []) or []:
                    row = {"series_type": series_type}
                    row.update(item)
                    # Plot-support arrays belong in raw machine results, not CSV.
                    row.pop("monthly_dates", None)
                    row.pop("monthly_values", None)
                    row.pop("trend_dates", None)
                    row.pop("trend_values", None)
                    row.pop("flow_duration_curve", None)
                    rows_by_table[table_name].append(row)
    return {
        name: pd.DataFrame(rows)
        for name, rows in rows_by_table.items()
    }


def run_layer1_diagnostics(
    obs_series: pd.Series,
    sim_series: Optional[pd.Series] = None,
    model_name: str = "AIFL",
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    """
    Run the currently enabled Layer 1 checks on `obs_series` (and, if given,
    `sim_series` tagged with `model_name` -- not just "sim", since
    future work may evaluate more than one model) and return every
    table visualisation.py needs, plus the raw per-check result dicts
    so visualisation.py never has to call a check itself.

    Returns a dict with:
      eda_summary       : DataFrame, one row per series
      summary_all       : DataFrame, one row per check per series
      summary_flagged   : DataFrame, summary_all filtered to flag == True
      raw_results       : {series_type: [check result dicts]}
    """
    obs_clean = obs_series.dropna()
    record_length = len(obs_clean) if not obs_clean.empty else len(obs_series)

    eda_rows = [eda_summary_table(obs_series, "obs")]
    raw_results = {
        "obs": _run_layer1_checks(obs_series, "obs", config=config)
    }
    summary_rows = _layer1_summary_rows(raw_results["obs"], record_length)

    if sim_series is not None:
        eda_rows.append(eda_summary_table(sim_series, model_name))
        raw_results[model_name] = _run_layer1_checks(
            sim_series, model_name, config=config
        )
        summary_rows.extend(_layer1_summary_rows(raw_results[model_name], record_length))

    summary_all = pd.DataFrame(summary_rows)
    summary_flagged = summary_all[summary_all["flag"]].reset_index(drop=True)
    detail_tables = _layer1_detail_tables(raw_results)

    return {
        "eda_summary": pd.concat(eda_rows, ignore_index=True),
        "summary_all": summary_all,
        "summary_flagged": summary_flagged,
        "raw_results": raw_results,
        "detail_tables": detail_tables,
    }


# ==========================================================
# LAYER 2
# ==========================================================

def percentile_diagnostic_table(obs_results: dict) -> pd.DataFrame:
    """
    Compare the most recent retained year's flashiness, rising rate,
    recession rate, and time-to-peak against that same metric's own
    historical P05-P95 envelope built from every OTHER retained year
    (or event) in the record. Table-only, matching layer2_hydsign's
    original percentile-diagnostic block, which never had a plot of
    its own.
    """
    rows = []

    # --- flashiness: most recent retained year vs. the rest ---
    annual_flashiness = obs_results["flashiness"].tables.get("annual_flashiness", pd.DataFrame())
    retained = annual_flashiness[annual_flashiness.get("retained", True) == True] if not annual_flashiness.empty else annual_flashiness

    if not retained.empty and len(retained) >= 2:
        retained_sorted = retained.sort_values("year")
        latest = retained_sorted.iloc[-1]
        history = retained_sorted.iloc[:-1]["flashiness_index"].dropna()

        if len(history) >= 2:
            p05, p95 = history.quantile(0.05), history.quantile(0.95)
            message, flag = percentile_diagnostic(
                value=latest["flashiness_index"], p05=p05, p95=p95,
                low_message=f"{int(latest['year'])} flashiness is unusually LOW vs. this station's own history",
                normal_message=f"{int(latest['year'])} flashiness is within this station's normal historical range",
                high_message=f"{int(latest['year'])} flashiness is unusually HIGH vs. this station's own history",
            )
            rows.append({
                "metric": "flashiness_index", "reference_period": f"year {int(latest['year'])}",
                "value": latest["flashiness_index"], "historical_p05": p05, "historical_p95": p95,
                "diagnostic": message, "flag": flag,
            })

    # --- rising rate / recession rate / time-to-peak: most recent
    #     event vs. every earlier event ---
    events = obs_results["threshold_event_hydrographs"].tables.get("events", pd.DataFrame())
    if not events.empty and len(events) >= 2:
        events_sorted = events.sort_values("event_start")
        latest_event = events_sorted.iloc[-1]
        history_events = events_sorted.iloc[:-1]

        for column, label in [
            ("median_positive_rising_rate", "rising_rate"),
            ("median_recession_slope", "recession_rate"),
            ("time_to_peak_days", "time_to_peak_days"),
        ]:
            history_values = history_events[column].dropna()
            if len(history_values) < 2 or pd.isna(latest_event.get(column)):
                continue

            p05, p95 = history_values.quantile(0.05), history_values.quantile(0.95)
            message, flag = percentile_diagnostic(
                value=latest_event[column], p05=p05, p95=p95,
                low_message=f"Latest event's {label} is unusually LOW vs. this station's own event history",
                normal_message=f"Latest event's {label} is within this station's normal event history",
                high_message=f"Latest event's {label} is unusually HIGH vs. this station's own event history",
            )
            rows.append({
                "metric": label, "reference_period": f"event starting {latest_event['event_start']}",
                "value": latest_event[column], "historical_p05": p05, "historical_p95": p95,
                "diagnostic": message, "flag": flag,
            })

    return pd.DataFrame(rows)


def run_layer2_diagnostics(
    obs_series: pd.Series,
    ml_series: Optional[pd.Series] = None,
    model_name: str = "AIFL",
    fill_method: str = "seasonal_climatology",
    layer1_obs_profile: Optional[dict] = None,
    discharge_unit: str = "source units",
    fill_window_days: int = 15,
    fill_min_samples: int = 5,
    signature_kwargs: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Run all 13 Layer 2 signature checks on OBS and, when supplied, compare
    a model over exactly the same period. Missing values are retained in the
    aligned records and filled only in temporary calculation copies.

    Returns a dict with:
      signature_comparison     : DataFrame, exactly 13 signature rows
      full_comparison          : optional detailed scalar comparison
      percentile_diagnostics   : DataFrame, this-station-vs-its-own-history table
      obs_results, model_results : raw {group_name: SignatureResult}
    """
    calculated = calculate_obs_ml_signature_results(
        obs_series,
        ml_series,
        fill_method=fill_method,
        layer1_obs_profile=layer1_obs_profile,
        discharge_unit=discharge_unit,
        fill_window_days=fill_window_days,
        fill_min_samples=fill_min_samples,
        signature_kwargs=signature_kwargs,
    )
    obs_results = calculated["obs_results"]
    model_results = calculated["ml_results"]
    primary_comparison = build_primary_signature_comparison(
        obs_results,
        model_results,
        model_name=model_name,
        coverage=calculated["coverage"],
    )

    if model_results is None:
        full_comparison = pd.DataFrame()
    else:
        full_comparison = compare_all_scalar_metrics(
            basin_id="station",
            obs_results=obs_results,
            ml_results=model_results,
            relative_tolerance_percent=float("inf"),
        ).rename(columns={"ml_value": f"{model_name}_value"})
        full_comparison = full_comparison.drop(
            columns=["comparison_distance", "flag"], errors="ignore"
        )
        full_comparison["comparison_status"] = "descriptive_only"

    return {
        "mode": calculated["mode"],
        "coverage": calculated["coverage"],
        "imputation_log": calculated["imputation_log"],
        "threshold_provenance": calculated["threshold_provenance"],
        "obs_aligned": calculated["obs_aligned"],
        "model_aligned": calculated["ml_aligned"],
        "obs_analysis": calculated["obs_analysis"],
        "model_analysis": calculated["ml_analysis"],
        "signature_comparison": primary_comparison,
        # Compatibility alias: this is now intentionally the same 13-row table.
        "compact_comparison": primary_comparison,
        "full_comparison": full_comparison,
        "percentile_diagnostics": percentile_diagnostic_table(obs_results),
        "obs_results": obs_results,
        "model_results": model_results,
        "model_name": model_name,
    }


# ==========================================================
# LAYER 3
# ==========================================================
# Layer 3 has no model/ML component: the network NetCDF used for
# context candidates only has OBSERVED discharge, so there is nothing
# to compare a model against for any gauge except the handful with
# their own full CSVs. Layer 3 is deliberately OBS-only; Layer 2 is
# where model evaluation happens.

_CONTEXT_TIER_MEANING = {
    "strong": "At least two same-river candidates are available -- if they agree, that's fairly strong corroboration.",
    "moderate": "One same-river candidate plus another same-catchment candidate -- reasonable corroboration, not as strong as 'strong'.",
    "weak": "At least two same-catchment candidates, but none share the target's named river -- corroboration is weaker and less direct.",
    "limited": "Only one candidate exists, so no majority agreement between candidates is possible.",
    "unavailable": "No suitable same-catchment candidate exists. This is a valid outcome, not a data-quality flag -- it just means this target has no usable context in the current network.",
}


def run_layer3_diagnostics(
    candidate_result: dict,
    comparison_table: pd.DataFrame,
) -> dict:
    """
    Build the Layer 3 tables from already-computed results (from
    gauge_network.find_context_candidates and
    discharge_comparison.compare_target_with_candidates) -- this
    function does no candidate-finding or discharge comparison itself.

    Returns a dict with:
      context_summary : DataFrame, one row describing this target's
                         context availability (status tier, radius,
                         candidate counts)
      comparison_table : DataFrame, the per-candidate comparison
                          metrics, passed through unchanged
      interpretation  : str, plain-language meaning of this target's
                         status tier
    """
    status = candidate_result["status"]
    candidates = candidate_result["candidates"]

    context_summary = pd.DataFrame([{
        "target_gauge_id": candidate_result["target_id"],
        "catchment": candidate_result["target"].get("Catchment"),
        "river": candidate_result["target"].get("River"),
        "candidate_selection_method": candidate_result["selection_method"],
        "context_status": status,
        "candidate_count": len(candidates),
        "same_river_candidate_count": int(candidates["same_river"].sum()) if not candidates.empty else 0,
        "best_area_similarity": (
            candidates["area_similarity"].max() if not candidates.empty else np.nan
        ),
    }])

    return {
        "context_summary": context_summary,
        "comparison_table": comparison_table,
        "interpretation": _CONTEXT_TIER_MEANING.get(status, "Unknown status tier."),
    }


if __name__ == "__main__":
    print(
        "This module is meant to be imported. Call "
        "run_layer1_diagnostics(obs_series, sim_series, model_name=...), "
        "run_layer2_diagnostics(obs_series, ml_series, model_name=...), "
        "and/or run_layer3_diagnostics(candidate_result, comparison_table) "
        "-- see the docstrings above. Pass their output straight into "
        "visualisation.py's generate_layer1_visuals / generate_layer2_visuals "
        "/ generate_layer3_visuals."
    )
