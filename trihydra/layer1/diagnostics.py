"""Run Layer 1 checks and assemble reusable diagnostic tables."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from trihydra.settings.defaults import DEFAULT_LAYER1_CONFIG, merge_config
from trihydra.formatting import field, metric, section

from trihydra.layer1.checks import run_layer1_checks

LAYER1_CHECK_LABELS = {
    "missing_values": "Missing values",
    "long_gaps": "Long gaps",
    "negative_discharge": "Negative discharge",
    "duplicate_timestamps": "Duplicate timestamps",
    "timestep_consistency": "Timestep consistency",
    "zero_flow_regime": "Zero-flow regime",
    "low_variability": "Non-zero plateau / flatline",
    "spike_dip": "Spike / dip",
    "step_shift": "Level/regime shift",
    "epoch_drift": "Epoch drift",
}


def eda_summary_table(series: pd.Series, series_type: str) -> pd.DataFrame:
    """Summarise valid values without altering the supplied series."""
    valid = series.dropna()

    if valid.empty:
        row = {
            stat: np.nan
            for stat in ["count", "min", "q05", "median", "mean", "q95", "max"]
        }
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

    columns = [
        "series_type", "start_date", "end_date", "count", "min", "q05",
        "median", "mean", "q95", "max",
    ]
    return pd.DataFrame([row], columns=columns)


def _describe_check_value(check_result: dict, record_length: int) -> str:
    """Translate a check-specific value into a readable summary."""
    check = check_result["check"]
    value = check_result.get("value")

    if value is None:
        return "Not calculated"

    def pct_of_record(count: float) -> str:
        if not record_length:
            return ""
        return f" ({count / record_length * 100:.2f}% of available time steps)"

    if check == "missing_values":
        percentage = check_result.get("internal_missing_percentage")
        suffix = (
            "" if percentage is None
            else f" ({float(percentage):.2f}% of internal record)"
        )
        return f"{value:.0f} missing day(s){suffix}"
    if check == "long_gaps":
        return f"longest internal gap = {value:.0f} day(s)"
    if check == "negative_discharge":
        return f"{value:.0f} negative day(s)" + pct_of_record(value)
    if check == "duplicate_timestamps":
        return f"{value:.0f} unique duplicated date(s)"
    if check == "timestep_consistency":
        return f"{value:.0f} irregular time step(s)"
    if check == "zero_flow_regime":
        return f"zero-flow ratio = {value:.3f} ({value * 100:.1f}% of days)"
    if check == "low_variability":
        periods = check_result.get("plateau_periods", []) or []
        return (
            f"{len(periods)} non-zero plateau candidate(s); "
            f"{value:.0f} repeated observation(s)"
        )
    if check == "spike_dip":
        return f"{value:.0f} candidate spike/dip day(s)"
    if check == "step_shift":
        return (
            f"{check_result.get('tier_1_count', 0)} Tier 1 / "
            f"{check_result.get('tier_2_count', 0)} Tier 2 boundary(ies)"
        )
    if check == "epoch_drift":
        tier = check_result.get("tier", "Not assessed")
        diagnosis = check_result.get("diagnosis", "Not assessed")
        return (
            f"{diagnosis} ({tier}); stable for "
            f"{100 * float(value):.1f}% of assessed years"
        )

    return str(value)


def _layer1_summary_rows(results: list[dict], record_length: int) -> list[dict]:
    """Build one concise summary row per executed check."""
    return [
        {
            "series_type": r["series_type"],
            "check": r["check"],
            "check_label": LAYER1_CHECK_LABELS.get(r["check"], r["check"]),
            "check_group": r.get("check_group", "basic"),
            "status": r.get("status", r.get("execution_status", "completed")),
            "execution_status": r.get("execution_status", "completed"),
            "finding_status": r.get("finding_status"),
            "reason_skipped": r.get("reason_skipped"),
            "flag": r["flag"],
            "value_description": _describe_check_value(r, record_length),
            "message": r.get("message", ""),
        }
        for r in results
    ]


def _layer1_detail_tables(raw_results: dict) -> dict[str, pd.DataFrame]:
    """Flatten structured check evidence into human-readable tables."""
    rows_by_table: dict[str, list[dict]] = {
        "missing_intervals": [],
        "long_gaps": [],
        "spike_dip_candidates": [],
        "spike_dip_rejected_coherent_patterns": [],
        "step_shift_boundaries": [],
        "step_shift_regimes": [],
        "epoch_drift_annual_evidence": [],
        "epoch_drift_five_year_epochs": [],
        "epoch_drift_overview_slopes": [],
        "epoch_drift_diagnosis": [],
        "zero_flow_spells": [],
        "nonzero_plateau_periods": [],
    }
    mapping = {
        "missing_values": [("internal_intervals", "missing_intervals")],
        "long_gaps": [("long_gap_intervals", "long_gaps")],
        "spike_dip": [
            ("candidate_details", "spike_dip_candidates"),
            (
                "rejected_coherent_patterns",
                "spike_dip_rejected_coherent_patterns",
            ),
        ],
        "step_shift": [
            ("regime_boundaries", "step_shift_boundaries"),
            ("regime_summary", "step_shift_regimes"),
        ],
        "epoch_drift": [
            ("annual_level_evidence", "epoch_drift_annual_evidence"),
            ("five_year_epochs", "epoch_drift_five_year_epochs"),
            (
                "consolidated_overview_slopes",
                "epoch_drift_overview_slopes",
            ),
            ("overview_diagnosis", "epoch_drift_diagnosis"),
        ],
        "zero_flow_regime": [("zero_flow_spells", "zero_flow_spells")],
        "low_variability": [("plateau_periods", "nonzero_plateau_periods")],
    }
    for series_type, results in raw_results.items():
        for result in results:
            for field, table_name in mapping.get(result["check"], []):
                for item in result.get(field, []) or []:
                    row = {"series_type": series_type}
                    row.update(item)
                    if table_name == "step_shift_boundaries":
                        row = {
                            "series_type": series_type,
                            "boundary": item.get("boundary_timestamp"),
                            "before_median": item.get("before_median"),
                            "after_median": item.get("after_median"),
                            "absolute_median_change": item.get(
                                "absolute_median_change"
                            ),
                            "fdc_q95_percentile_q05_threshold": item.get(
                                "fdc_q95_percentile_q05_threshold"
                            ),
                            "fdc_q75_percentile_q25_threshold": item.get(
                                "fdc_q75_percentile_q25_threshold"
                            ),
                            "diagnosis": item.get("diagnosis"),
                        }
                    rows_by_table[table_name].append(row)
    return {
        name: pd.DataFrame(rows)
        for name, rows in rows_by_table.items()
    }


def run_layer1_diagnostics(
    obs_series: pd.Series,
    sim_series: pd.Series | None = None,
    model_name: str = "model",
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    """Run Layer 1 independently on a primary and optional secondary series."""
    eda_rows = [eda_summary_table(obs_series, "obs")]
    raw_results = {
        "obs": run_layer1_checks(obs_series, "obs", config=config)
    }
    summary_rows = _layer1_summary_rows(
        raw_results["obs"], int(obs_series.notna().sum())
    )

    if sim_series is not None:
        eda_rows.append(eda_summary_table(sim_series, model_name))
        raw_results[model_name] = run_layer1_checks(
            sim_series, model_name, config=config
        )
        summary_rows.extend(
            _layer1_summary_rows(
                raw_results[model_name], int(sim_series.notna().sum())
            )
        )

    summary_all = pd.DataFrame(summary_rows)
    summary_flagged = summary_all[summary_all["flag"]].reset_index(drop=True)
    detail_tables = _layer1_detail_tables(raw_results)

    return {
        "eda_summary": pd.concat(eda_rows, ignore_index=True),
        "summary_all": summary_all,
        "summary_flagged": summary_flagged,
        "raw_results": raw_results,
        "detail_tables": detail_tables,
        "summary_metrics": {},
        "thresholds_used": {},
        "evidence": pd.DataFrame(),
    }


def finalise_layer1_contract(
    series: pd.Series,
    result: dict,
    composite: dict,
    config: Mapping[str, Any] | None = None,
) -> dict:
    """Attach user-facing metrics, thresholds, and one evidence table."""
    settings = merge_config(DEFAULT_LAYER1_CONFIG, config)
    checks = {item["check"]: item for item in result["raw_results"]["obs"]}
    get = lambda name: checks.get(name, {})
    missing, gaps = get("missing_values"), get("long_gaps")
    negative, duplicates = get("negative_discharge"), get("duplicate_timestamps")
    timestep, zero = get("timestep_consistency"), get("zero_flow_regime")
    plateau, spike = get("low_variability"), get("spike_dip")
    step, epoch = get("step_shift"), get("epoch_drift")
    candidates = spike.get("candidate_details", []) or []
    spike_count = sum(item.get("type") == "spike" for item in candidates)
    dip_count = sum(item.get("type") == "dip" for item in candidates)
    assessment = composite["summary"].iloc[0]
    concerns = composite["components"].loc[
        composite["components"]["contribution"] > 0, "check"
    ].tolist()
    result["summary_metrics"] = {
        "missing_day_count": int(missing.get("internal_nan_count") or 0),
        "missing_percentage": missing.get("internal_missing_percentage"),
        "long_gap_count": len(gaps.get("long_gap_intervals", []) or []),
        "longest_gap_days": gaps.get("value"),
        "duplicate_timestamp_count": duplicates.get("value"),
        "duplicate_extra_row_count": duplicates.get("extra_duplicate_rows"),
        "irregular_timestep_count": timestep.get("irregular_spacing_count"),
        "timestamps_out_of_order": timestep.get("out_of_order"),
        "negative_discharge_count": negative.get("value"),
        "most_negative_discharge_magnitude": negative.get("maximum_negative_magnitude"),
        "zero_flow_percentage": None if zero.get("zero_ratio") is None else 100 * zero["zero_ratio"],
        "zero_flow_spell_count": zero.get("zero_spell_count"),
        "longest_zero_flow_spell_days": zero.get("longest_zero_spell_days"),
        "nonzero_plateau_count": plateau.get("plateau_count"),
        "longest_nonzero_plateau_days": plateau.get("longest_plateau_days"),
        "spike_count": spike_count,
        "dip_count": dip_count,
        "unresolved_spike_dip_count": len(candidates),
        "step_shift_candidate_count": step.get("retained_candidate_count"),
        "step_shift_score": step.get("step_shift_score"),
        "step_shift_tier": step.get("composite_tier"),
        "epoch_regime": epoch.get("dominant_behaviour"),
        "epoch_stable_fraction": epoch.get("stable_year_fraction"),
        "epoch_tier": epoch.get("tier"),
        "layer1_score": int(assessment["layer1_score"]),
        "layer1_score_percent": assessment.get("layer1_score_percent"),
        "layer1_maximum_assessable_score": assessment.get(
            "maximum_assessable_score"
        ),
        "layer1_assessment_scope": assessment.get("assessment_scope"),
        "layer1_scope_conclusion": assessment.get("scope_conclusion"),
        "layer1_enabled_check_count": assessment.get("enabled_check_count"),
        "layer1_assessable_check_count": assessment.get("assessable_checks"),
        "layer1_total_composite_check_count": assessment.get(
            "total_composite_check_count"
        ),
        "layer1_evidence_coverage_percent": assessment.get(
            "evidence_coverage_percent"
        ),
        "layer1_enabled_checks": assessment.get("enabled_checks"),
        "layer1_disabled_checks": assessment.get("disabled_checks"),
        "layer1_class": str(assessment["layer1_class"]),
        "layer1_primary_concerns": "; ".join(concerns) if concerns else "None",
    }
    comp = settings["composite"]
    step_boundaries = step.get("regime_boundaries", []) or []
    result["thresholds_used"] = {
        "missing_tier2_threshold_percent": comp["missing_values"]["tier_2_minimum_percent"],
        "missing_tier1_threshold_percent": comp["missing_values"]["tier_1_above_percent"],
        "long_gap_definition_days": comp["long_gaps"]["long_gap_definition_days"],
        "long_gap_tier2_days": comp["long_gaps"]["tier_2_minimum_days"],
        "long_gap_tier1_days": comp["long_gaps"]["tier_1_minimum_days"],
        "negative_discharge_tolerance": settings["negative_discharge"]["tolerance"],
        "plateau_detection_days": settings["low_variability"]["minimum_plateau_days"],
        "plateau_tier1_days": comp["low_variability"]["tier_1_minimum_days"],
        "spike_minimum_recovery": settings["spike_dip"]["minimum_recovery"],
        "spike_minimum_score": settings["spike_dip"]["minimum_score"],
        "spike_robust_cutoff_used": spike.get("raw_robust_cutoff"),
        "step_shift_fdc_q95_used": next((x.get("fdc_q95_percentile_q05_threshold") for x in step_boundaries), None),
        "step_shift_fdc_q75_used": next((x.get("fdc_q75_percentile_q25_threshold") for x in step_boundaries), None),
        "epoch_block_years": settings["epoch_drift"]["epoch_years"],
        "epoch_tier2_stable_fraction": settings["epoch_drift"]["tier_2_minimum_stable_fraction"],
        "epoch_tier3_stable_fraction": settings["epoch_drift"]["tier_3_minimum_stable_fraction"],
        "layer1_minor_concerns_percent": comp["classification"]["minor_concerns_minimum_percent"],
        "layer1_needs_review_percent": comp["classification"]["needs_review_minimum_percent"],
    }
    frames = []
    for evidence_type, table in result.get("detail_tables", {}).items():
        if table.empty:
            continue
        frame = table.copy()
        frame.insert(0, "evidence_type", evidence_type)
        frames.append(frame)
    components = composite.get("components", pd.DataFrame())
    if not components.empty:
        frame = components.copy()
        frame.insert(0, "evidence_type", "layer1_composite_component")
        frames.append(frame)
    result["evidence"] = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return result


def render_layer1_summary(row: pd.Series, unit: str) -> list[str]:
    """Render the principal values produced by Layer 1."""
    return section("LAYER 1  -  INTRINSIC DATA QUALITY") + [
        "  Missing data",
        field("Missing observations", f"{metric(row, 'layer1_missing_day_count')}  ({metric(row, 'layer1_missing_percentage', decimals=3, suffix='%')})"),
        field("Longest gap", metric(row, "layer1_longest_gap_days", suffix=" days")),
        "", "  Timestamp integrity",
        field("Duplicated dates", metric(row, "layer1_duplicate_timestamp_count")),
        field("Irregular timestep intervals", metric(row, "layer1_irregular_timestep_count")),
        field("Chronological order", "No" if row.get("layer1_timestamps_out_of_order") else "Yes"),
        "", "  Negative discharge",
        field("Material negative observations", metric(row, "layer1_negative_discharge_count")),
        field("Largest negative magnitude", metric(row, "layer1_most_negative_discharge_magnitude", decimals=3, suffix=f" {unit}")),
        "", "  Zero-flow behaviour",
        field("Zero-flow spells", metric(row, "layer1_zero_flow_spell_count")),
        field("Longest spell", metric(row, "layer1_longest_zero_flow_spell_days", suffix=" days")),
        field("Share of valid record at zero flow", metric(row, "layer1_zero_flow_percentage", decimals=2, suffix="%")),
        "", "  Plateaus",
        field("Retained non-zero plateaus", metric(row, "layer1_nonzero_plateau_count")),
        field("Longest retained plateau", metric(row, "layer1_longest_nonzero_plateau_days", suffix=" days")),
        "", "  Spikes and dips",
        field("Spikes", metric(row, "layer1_spike_count")),
        field("Dips", metric(row, "layer1_dip_count")),
        field("Unresolved candidates", metric(row, "layer1_unresolved_spike_dip_count")),
        "", "  Step shifts",
        field("Retained candidates", metric(row, "layer1_step_shift_candidate_count")),
        field("Step-shift score", metric(row, "layer1_step_shift_score", decimals=3)),
        field("Classification", metric(row, "layer1_step_shift_tier")),
        "", "  Long-term behaviour",
        field("Dominant regime", metric(row, "layer1_epoch_regime")),
        field("Classification", metric(row, "layer1_epoch_tier")),
        field("Stable share of assessed years", metric(row, "layer1_epoch_stable_fraction", decimals=2)),
    ]


__all__ = [
    "LAYER1_CHECK_LABELS", "finalise_layer1_contract",
    "render_layer1_summary", "run_layer1_diagnostics",
]
