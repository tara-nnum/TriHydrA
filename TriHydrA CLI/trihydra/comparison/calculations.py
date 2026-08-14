"""Generic calculations for prepared same-station comparisons."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from trihydra.comparison.preparation import PreparedComparison
from trihydra.settings.defaults import DEFAULT_COMPARISON_CONFIG, merge_config
from trihydra.layer2.diagnostics import run_layer2_diagnostics
from trihydra.layer1.diagnostics import run_layer1_diagnostics
from trihydra.composite import score_layer1, score_layer2_comparison


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def calculate_daily_metrics(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Calculate deterministic daily metrics on pairwise-valid values only."""
    frame = pairwise[["reference", "candidate"]].dropna().astype(float)
    if frame.empty:
        raise ValueError("Daily metrics require pairwise-valid values.")
    reference = frame["reference"]
    candidate = frame["candidate"]
    residual = candidate - reference
    reference_mean = float(reference.mean())
    denominator = float(((reference - reference_mean) ** 2).sum())
    nse = 1.0 - float((residual**2).sum()) / denominator if denominator > 0 else np.nan
    pearson = reference.corr(candidate, method="pearson")
    spearman = reference.corr(candidate, method="spearman")
    reference_std = float(reference.std(ddof=0))
    candidate_std = float(candidate.std(ddof=0))
    alpha = candidate_std / reference_std if reference_std > 0 else np.nan
    beta = float(candidate.mean()) / reference_mean if reference_mean != 0 else np.nan
    kge = (
        1.0 - np.sqrt((pearson - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
        if all(np.isfinite(value) for value in (pearson, alpha, beta))
        else np.nan
    )
    percent_bias = (
        100.0 * float(residual.sum()) / float(reference.sum())
        if float(reference.sum()) != 0 else np.nan
    )
    rows = [
        ("pairwise_valid_days", float(len(frame)), "days"),
        ("reference_mean", reference_mean, "source units"),
        ("candidate_mean", float(candidate.mean()), "source units"),
        ("mean_bias", float(residual.mean()), "source units"),
        ("mean_absolute_error", float(residual.abs().mean()), "source units"),
        ("root_mean_squared_error", float(np.sqrt((residual**2).mean())), "source units"),
        ("percent_bias", percent_bias, "%"),
        ("pearson_correlation", pearson, "dimensionless"),
        ("spearman_correlation", spearman, "dimensionless"),
        ("nash_sutcliffe_efficiency", nse, "dimensionless"),
        ("kling_gupta_efficiency", kge, "dimensionless"),
        ("variability_ratio", alpha, "dimensionless"),
        ("mean_ratio", beta, "dimensionless"),
    ]
    return pd.DataFrame(
        [{"metric": name, "value": _finite(value), "unit": unit} for name, value, unit in rows]
    )


def _circular_month_distance(reference, candidate) -> float:
    if pd.isna(reference) or pd.isna(candidate):
        return np.nan
    difference = abs(int(reference) - int(candidate))
    return float(min(difference, 12 - difference))


def compare_annual_signatures(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    """Compare annual signatures on common usable years."""
    metrics = [
        "mean_flow", "median_flow", "minimum_flow", "maximum_flow",
        "flashiness_index", "baseflow_index", "seasonality_index",
        "lag1_autocorrelation", "wettest_month", "driest_month",
    ]
    available = [
        metric for metric in metrics
        if metric in reference.columns and metric in candidate.columns
    ]
    merged = reference[["year", *available]].merge(
        candidate[["year", *available]], on="year", how="inner",
        suffixes=("_reference", "_candidate"),
    )
    rows: list[dict[str, Any]] = []
    for _, year_row in merged.iterrows():
        for metric in available:
            reference_value = year_row[f"{metric}_reference"]
            candidate_value = year_row[f"{metric}_candidate"]
            circular = metric in {"wettest_month", "driest_month"}
            difference = (
                _circular_month_distance(reference_value, candidate_value)
                if circular
                else candidate_value - reference_value
                if pd.notna(reference_value) and pd.notna(candidate_value)
                else np.nan
            )
            rows.append({
                "year": int(year_row["year"]),
                "metric": metric,
                "reference_value": reference_value,
                "candidate_value": candidate_value,
                "difference": difference,
                "difference_semantics": (
                    "circular month distance" if circular else "candidate - reference"
                ),
            })
    return pd.DataFrame(rows)


def compare_diagnostic_summaries(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    """Compare whole-record Layer 2 medians and envelopes."""
    merged = reference.merge(
        candidate, on="diagnostic", how="outer",
        suffixes=("_reference", "_candidate"),
    )
    merged["median_difference"] = (
        merged["median_candidate"] - merged["median_reference"]
    )
    return merged


def compare_signature_summaries(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    """Compare typical annual signatures even when years do not overlap."""
    metrics = [
        "mean_flow", "median_flow", "minimum_flow", "maximum_flow",
        "flashiness_index", "baseflow_index", "seasonality_index",
        "lag1_autocorrelation", "wettest_month", "driest_month",
    ]
    rows = []
    for metric in metrics:
        if metric not in reference or metric not in candidate:
            continue
        reference_values = pd.to_numeric(reference[metric], errors="coerce").dropna()
        candidate_values = pd.to_numeric(candidate[metric], errors="coerce").dropna()
        circular = metric in {"wettest_month", "driest_month"}
        if circular:
            reference_value = reference_values.mode().iloc[0] if not reference_values.empty else np.nan
            candidate_value = candidate_values.mode().iloc[0] if not candidate_values.empty else np.nan
            difference = _circular_month_distance(reference_value, candidate_value)
        else:
            reference_value = reference_values.median() if not reference_values.empty else np.nan
            candidate_value = candidate_values.median() if not candidate_values.empty else np.nan
            difference = (
                candidate_value - reference_value
                if pd.notna(reference_value) and pd.notna(candidate_value)
                else np.nan
            )
        rows.append({
            "metric": metric,
            "reference_typical_value": reference_value,
            "candidate_typical_value": candidate_value,
            "difference": difference,
            "reference_year_count": int(len(reference_values)),
            "candidate_year_count": int(len(candidate_values)),
            "summary_statistic": "mode" if circular else "median",
            "difference_semantics": (
                "circular month distance" if circular else "candidate - reference"
            ),
        })
    return pd.DataFrame(rows)


def _layer2_inputs(prepared: PreparedComparison) -> tuple[pd.Series, pd.Series]:
    return prepared.reference_selected, prepared.candidate_selected


def run_generic_comparison(
    prepared: PreparedComparison,
    *,
    layer1_config: Mapping[str, Mapping[str, Any]] | None = None,
    layer2_config: Mapping[str, Any] | None = None,
    comparison_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Layer 1 evidence and Layer 2 signatures for two inputs.

    Layer 1 receives each selected raw series independently so source-specific
    missingness and invalid values remain visible. For paired comparisons,
    Layer 2 receives identically pairwise-masked calendars so hydrological
    signature differences cannot be caused by a simulation-only tail.
    """
    settings = merge_config(DEFAULT_COMPARISON_CONFIG, comparison_config)
    reference_before = prepared.reference.values.copy(deep=True)
    candidate_before = prepared.candidate.values.copy(deep=True)

    # QA/QC describes each source as delivered.  The common window belongs
    # only to comparison calculations and must never truncate standalone L1.
    reference_l1_input = (
        prepared.reference.values
        if prepared.mode == "paired" else prepared.reference_selected
    )
    candidate_l1_input = (
        prepared.candidate.values
        if prepared.mode == "paired" else prepared.candidate_selected
    )
    reference_layer1 = run_layer1_diagnostics(
        reference_l1_input, config=layer1_config
    )
    candidate_layer1 = run_layer1_diagnostics(
        candidate_l1_input, config=layer1_config
    )
    reference_native_layer2 = run_layer2_diagnostics(
        reference_l1_input,
        layer1_result=reference_layer1,
        config=layer2_config,
        discharge_unit=prepared.reference.unit,
    )
    candidate_native_layer2 = run_layer2_diagnostics(
        candidate_l1_input,
        layer1_result=candidate_layer1,
        config=layer2_config,
        discharge_unit=prepared.candidate.unit,
    )
    reference_l2_input, candidate_l2_input = _layer2_inputs(prepared)
    if prepared.mode == "paired":
        reference_comparison_layer2 = run_layer2_diagnostics(
            reference_l2_input,
            config=layer2_config,
            discharge_unit=prepared.reference.unit,
        )
        candidate_comparison_layer2 = run_layer2_diagnostics(
            candidate_l2_input,
            config=layer2_config,
            discharge_unit=prepared.candidate.unit,
        )
    else:
        reference_comparison_layer2 = reference_native_layer2
        candidate_comparison_layer2 = candidate_native_layer2
    calculate_daily = bool(settings["daily_metrics"]["calculate"])
    daily = (
        calculate_daily_metrics(prepared.pairwise_values)
        if calculate_daily
        and prepared.mode == "paired"
        and prepared.pairwise_values is not None
        else pd.DataFrame(columns=["metric", "value", "unit"])
    )
    provided_metrics = {}
    if bool(settings["provided_metrics"]["include"]):
        for label, descriptor in (
            ("reference", prepared.reference),
            ("candidate", prepared.candidate),
        ):
            supplied = descriptor.metadata.get("stored_performance_metrics")
            if supplied:
                provided_metrics[label] = dict(supplied)
    annual = compare_annual_signatures(
        reference_comparison_layer2["annual_signatures"],
        candidate_comparison_layer2["annual_signatures"],
    )
    diagnostics = compare_diagnostic_summaries(
        reference_comparison_layer2["diagnostic_summary"],
        candidate_comparison_layer2["diagnostic_summary"],
    )
    signature_summary = compare_signature_summaries(
        reference_comparison_layer2["annual_signatures"],
        candidate_comparison_layer2["annual_signatures"],
    )
    reference_layer1_composite = score_layer1(
        reference_l1_input, reference_layer1, reference_native_layer2,
        config=layer1_config,
    )
    candidate_layer1_composite = score_layer1(
        candidate_l1_input, candidate_layer1, candidate_native_layer2,
        config=layer1_config,
    )
    layer2_composite = score_layer2_comparison(
        reference_l2_input, candidate_l2_input,
        reference_comparison_layer2, candidate_comparison_layer2,
        config=layer2_config,
    )
    composite_summary = layer2_composite["summary"].iloc[0]
    summary_metrics = {
        "mode": prepared.mode,
        "reference_name": prepared.reference.name,
        "candidate_name": prepared.candidate.name,
        "common_start_date": prepared.coverage.get("common_window_start"),
        "common_end_date": prepared.coverage.get("common_window_end"),
        "common_timestamp_count": prepared.coverage.get("common_calendar_count"),
        "pairwise_valid_count": prepared.coverage.get("pairwise_valid_count"),
        "layer2_comparison_score": composite_summary.get("layer2_score"),
        "layer2_comparison_class": composite_summary.get("layer2_class"),
        "assessable_component_count": composite_summary.get("assessable_components"),
        "assessment_incomplete": composite_summary.get("assessment_incomplete"),
        "unavailable_components": composite_summary.get("unavailable_components"),
    }
    thresholds_used = {
        f"layer2_comparison_{name}": value
        for name, value in layer2_composite.get("configuration_used", {}).items()
        if not isinstance(value, dict)
    }
    evidence_frames = []
    for evidence_type, table in (
        ("component_score", layer2_composite["components"]),
        ("annual_signature_comparison", annual),
        ("signature_summary_comparison", signature_summary),
        ("diagnostic_comparison", diagnostics),
        ("daily_metric", daily),
    ):
        if table is None or table.empty:
            continue
        frame = table.copy()
        frame.insert(0, "evidence_type", evidence_type)
        evidence_frames.append(frame)

    pd.testing.assert_series_equal(prepared.reference.values, reference_before)
    pd.testing.assert_series_equal(prepared.candidate.values, candidate_before)
    return {
        "mode": prepared.mode,
        "reference_name": prepared.reference.name,
        "candidate_name": prepared.candidate.name,
        "station_id": prepared.reference.station_id,
        "unit": prepared.reference.unit,
        "coverage": dict(prepared.coverage),
        "comparison_rules": prepared.comparison_rules,
        "overlay_data": pd.concat(
            [
                reference_l2_input.rename("reference"),
                candidate_l2_input.rename("candidate"),
            ],
            axis=1,
        ),
        "daily_metrics": daily,
        "daily_metrics_calculated": calculate_daily,
        "provided_metrics": provided_metrics,
        "annual_signature_comparison": annual,
        "signature_summary_comparison": signature_summary,
        "diagnostic_comparison": diagnostics,
        "reference_layer1": reference_layer1,
        "candidate_layer1": candidate_layer1,
        "reference_layer2": reference_native_layer2,
        "candidate_layer2": candidate_native_layer2,
        "reference_native_layer2": reference_native_layer2,
        "candidate_native_layer2": candidate_native_layer2,
        "reference_comparison_layer2": reference_comparison_layer2,
        "candidate_comparison_layer2": candidate_comparison_layer2,
        "reference_layer1_composite": reference_layer1_composite,
        "candidate_layer1_composite": candidate_layer1_composite,
        "layer2_composite": layer2_composite,
        "summary_metrics": summary_metrics,
        "thresholds_used": thresholds_used,
        "evidence": (
            pd.concat(evidence_frames, ignore_index=True, sort=False)
            if evidence_frames else pd.DataFrame()
        ),
    }


__all__ = [
    "calculate_daily_metrics",
    "compare_annual_signatures",
    "compare_diagnostic_summaries",
    "compare_signature_summaries",
    "run_generic_comparison",
]
