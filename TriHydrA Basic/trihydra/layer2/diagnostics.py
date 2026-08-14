"""Layer 2 observed-only diagnostic orchestration."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from trihydra.formatting import date, field, metric, month, section

from trihydra.layer2.annual_signatures import (
    build_diagnostic_summary,
    build_seasonality_profile,
    calculate_annual_signatures,
)
from trihydra.layer2.hydrograph_information import (
    calculate_high_flow_events,
    select_representative_event,
)
from trihydra.layer2.peak_outlier_crosscheck import (
    crosscheck_peak_outliers,
    mark_representative_eligibility,
)


def _spike_dip_details(layer1_result: dict | None) -> list[dict]:
    if not layer1_result:
        return []
    if "candidate_details" in layer1_result:
        return list(layer1_result.get("candidate_details", []) or [])
    sources = layer1_result.get("raw_results", {})
    preferred = sources.get("obs", sources.get("observation"))
    result_sets = [preferred] if preferred is not None else list(sources.values())
    for results in result_sets:
        result = next(
            (item for item in results if item.get("check") == "spike_dip"),
            None,
        )
        if result is not None:
            return list(result.get("candidate_details", []) or [])
    return []


def run_layer2_diagnostics(
    obs_series: pd.Series,
    *,
    layer1_result: dict | None = None,
    config: Mapping[str, Any] | None = None,
    discharge_unit: str = "source units",
) -> dict:
    """Calculate annual signatures and high-flow events."""
    settings = {} if config is None else dict(config)
    annual_settings = dict(settings.get("annual", {}))
    event_settings = dict(settings.get("events", {}))
    missing_before = int(obs_series.isna().sum())
    spike_dip_details = _spike_dip_details(layer1_result)
    excluded_extrema_timestamps = pd.DatetimeIndex([
        pd.Timestamp(item["timestamp"])
        for item in spike_dip_details
        if item.get("timestamp") is not None
    ])
    annual, monthly, references = calculate_annual_signatures(
        obs_series,
        excluded_extrema_timestamps=excluded_extrema_timestamps,
        **annual_settings,
    )
    trigger_percentile = float(event_settings.get("trigger_percentile", 0.95))
    boundary_percentile = float(event_settings.get("boundary_percentile", 0.90))
    raw_valid = obs_series.dropna()
    if not 0 < boundary_percentile < trigger_percentile < 1:
        raise ValueError(
            "Layer 2 event percentiles must satisfy "
            "0 < boundary_percentile < trigger_percentile < 1."
        )
    trigger = (
        float(raw_valid.quantile(trigger_percentile))
        if not raw_valid.empty else float("nan")
    )
    boundary = (
        float(raw_valid.quantile(boundary_percentile))
        if not raw_valid.empty else float("nan")
    )
    references["high_flow_trigger_percentile"] = trigger_percentile
    references["high_flow_boundary_percentile"] = boundary_percentile
    references["high_flow_trigger_value"] = trigger
    references["high_flow_boundary_value"] = boundary
    references["unit"] = discharge_unit
    events = calculate_high_flow_events(
        obs_series,
        trigger_threshold=trigger,
        boundary_threshold=boundary,
    )
    crosscheck = crosscheck_peak_outliers(
        spike_dip_details, events,
        minimum_event_duration_days=float(
            event_settings.get("spike_crosscheck_minimum_event_duration_days", 3.0)
        ),
    )
    events = mark_representative_eligibility(events, crosscheck)
    representative = select_representative_event(events)
    representative_table = (
        pd.DataFrame(columns=events.columns)
        if representative is None else pd.DataFrame([representative])
    )
    if int(obs_series.isna().sum()) != missing_before:
        raise AssertionError("Layer 2 modified the raw observation series.")
    diagnostic_summary = build_diagnostic_summary(annual, events)
    diagnostic_values = {
        str(row["diagnostic"]): row.get("median")
        for row in diagnostic_summary.to_dict("records")
    }
    screened = obs_series.drop(index=excluded_extrema_timestamps, errors="ignore").dropna()
    representative_row = (
        {} if representative_table.empty else representative_table.iloc[0].to_dict()
    )
    seasonal_profile = build_seasonality_profile(monthly)
    seasonal_valid = seasonal_profile.dropna(subset=["median"])
    summary_metrics = {
        "valid_observation_count": int(obs_series.notna().sum()),
        "mean_discharge": screened.mean() if not screened.empty else None,
        "median_discharge": screened.median() if not screened.empty else None,
        "minimum_discharge": screened.min() if not screened.empty else None,
        "maximum_discharge": screened.max() if not screened.empty else None,
        "low_flow_q05": references.get("q05_percentile_low_flow_fdc_q95"),
        "high_flow_q95": references.get("q95_percentile_high_flow_fdc_q05"),
        "annual_signature_year_count": int(len(annual)),
        "high_flow_event_count": int(len(events)),
        "median_annual_flashiness": diagnostic_values.get(
            "Median annual Richards-Baker flashiness"
        ),
        "median_annual_baseflow_index": diagnostic_values.get(
            "Median annual Lyne-Hollick BFI"
        ),
        "median_annual_seasonality_index": diagnostic_values.get(
            "Median annual Walsh-Lawler seasonality"
        ),
        "median_annual_lag1_autocorrelation": diagnostic_values.get(
            "Median annual lag-1 autocorrelation"
        ),
        "typical_wettest_month": (
            None if seasonal_valid.empty
            else int(seasonal_valid.loc[seasonal_valid["median"].idxmax(), "month"])
        ),
        "typical_driest_month": (
            None if seasonal_valid.empty
            else int(seasonal_valid.loc[seasonal_valid["median"].idxmin(), "month"])
        ),
        "representative_event_start": representative_row.get("event_start"),
        "representative_event_peak_date": representative_row.get("peak_date"),
        "representative_event_end": representative_row.get("event_end"),
        "representative_event_peak_flow": representative_row.get("peak_flow"),
        "representative_event_time_to_peak_days": representative_row.get("time_to_peak_days"),
        "representative_event_recession_days": representative_row.get("recession_days"),
        "representative_event_duration_days": representative_row.get("event_duration_days"),
        "representative_event_rising_slope": representative_row.get("rising_slope"),
        "representative_event_recession_slope": representative_row.get("recession_slope"),
        "spike_peak_overlap_count": int(events["layer1_spike_peak_overlap"].sum()),
        "excluded_flagged_extrema_count": int(len(excluded_extrema_timestamps)),
    }
    thresholds_used = {
        "high_flow_trigger_percentile": trigger_percentile,
        "high_flow_trigger_value": trigger,
        "high_flow_boundary_percentile": boundary_percentile,
        "high_flow_boundary_value": boundary,
        "spike_crosscheck_minimum_event_duration_days": float(
            event_settings.get("spike_crosscheck_minimum_event_duration_days", 3.0)
        ),
    }
    evidence_frames = []
    for evidence_type, table in (
        ("annual_signature", annual),
        ("annual_monthly_profile", monthly),
        ("seasonality_profile", seasonal_profile),
        ("high_flow_event", events),
        ("representative_event", representative_table),
        ("spike_peak_crosscheck", crosscheck),
        ("diagnostic_distribution", diagnostic_summary),
    ):
        if table is None or table.empty:
            continue
        frame = table.copy()
        frame.insert(0, "evidence_type", evidence_type)
        evidence_frames.append(frame)
    return {
        "annual_signatures": annual,
        "annual_monthly_profiles": monthly,
        "seasonality_profile": seasonal_profile,
        "hydrograph_events": events,
        "representative_event": representative_table,
        "spike_peak_crosscheck": crosscheck,
        "spike_peak_overlap_count": int(
            events["layer1_spike_peak_overlap"].sum()
        ),
        "diagnostic_summary": diagnostic_summary,
        "references": references,
        "raw_missing_count": missing_before,
        "imputation_used": False,
        "summary_metrics": summary_metrics,
        "thresholds_used": thresholds_used,
        "evidence": (
            pd.concat(evidence_frames, ignore_index=True, sort=False)
            if evidence_frames else pd.DataFrame()
        ),
    }


def render_layer2_summary(row: pd.Series, unit: str) -> list[str]:
    """Render hydrological signatures and representative-event values."""
    return section("LAYER 2  -  HYDROLOGICAL SIGNATURES") + [
        "  Whole-record flow",
        *[
            field(label, metric(row, key, decimals=3, suffix=f" {unit}"))
            for label, key in (
                ("Mean discharge", "layer2_mean_discharge"),
                ("Median discharge", "layer2_median_discharge"),
                ("Minimum discharge", "layer2_minimum_discharge"),
                ("Maximum discharge", "layer2_maximum_discharge"),
                ("Low-flow reference (Q05)", "layer2_low_flow_q05"),
                ("High-flow reference (Q95)", "layer2_high_flow_q95"),
            )
        ],
        "", "  Hydrological signatures (median annual)",
        field("Flashiness", metric(row, "layer2_median_annual_flashiness", decimals=3)),
        field("Baseflow index", metric(row, "layer2_median_annual_baseflow_index", decimals=3)),
        field("Seasonality index", metric(row, "layer2_median_annual_seasonality_index", decimals=3)),
        field("Lag-1 autocorrelation", metric(row, "layer2_median_annual_lag1_autocorrelation", decimals=3)),
        "", "  Typical seasonality",
        field("Wettest month", month(row.get("layer2_typical_wettest_month"))),
        field("Driest month", month(row.get("layer2_typical_driest_month"))),
        "", "  High-flow events",
        field("Events detected", metric(row, "layer2_high_flow_event_count")),
        field("Layer 1 candidates excluded", metric(row, "layer2_excluded_flagged_extrema_count")),
        field("Spike/peak overlaps to inspect", metric(row, "layer2_spike_peak_overlap_count")),
        "", "  Representative event",
        field("Start", date(row.get("layer2_representative_event_start"))),
        field("Peak", date(row.get("layer2_representative_event_peak_date"))),
        field("End", date(row.get("layer2_representative_event_end"))),
        field("Peak discharge", metric(row, "layer2_representative_event_peak_flow", decimals=3, suffix=f" {unit}")),
        field("Time to peak", metric(row, "layer2_representative_event_time_to_peak_days", decimals=1, suffix=" days")),
        field("Recession duration", metric(row, "layer2_representative_event_recession_days", decimals=1, suffix=" days")),
        field("Total duration", metric(row, "layer2_representative_event_duration_days", decimals=1, suffix=" days")),
        field("Rising slope", metric(row, "layer2_representative_event_rising_slope", decimals=3)),
        field("Recession slope", metric(row, "layer2_representative_event_recession_slope", decimals=3)),
    ]


__all__ = ["render_layer2_summary", "run_layer2_diagnostics"]
