"""Convert existing Layer 1 and Layer 2 evidence into review scores."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from trihydra.settings.defaults import (
    DEFAULT_LAYER1_CONFIG,
    DEFAULT_LAYER2_CONFIG,
    merge_config,
)
from trihydra.layer1.timeseries_validity import get_valid_record


LAYER2_TIER_POINTS = {"Tier 3": 0, "Tier 2": 1, "Tier 1": 2}


def _raw_checks(result: dict) -> dict[str, dict]:
    groups = result.get("raw_results", {})
    if not groups:
        return {}
    if len(groups) != 1:
        raise ValueError(
            "Layer 1 scoring requires diagnostics for exactly one series."
        )
    checks = next(iter(groups.values()))
    return {item["check"]: item for item in checks}


def _is_assessable(result: dict | None) -> bool:
    """Return whether a check exists and completed its assessment."""
    return bool(result) and result.get("execution_status") != "skipped"


def score_layer1(
    series: pd.Series,
    layer1_result: dict,
    layer2_result: dict | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert existing Layer 1 evidence into the agreed additive tiers."""
    layer1_settings = merge_config(DEFAULT_LAYER1_CONFIG, config)
    settings = layer1_settings["composite"]
    weights = settings["weights"]
    configured_points = settings["tier_points"]
    tier_points = {
        "Tier 3": int(configured_points["tier_3"]),
        "Tier 2": int(configured_points["tier_2"]),
        "Tier 1": int(configured_points["tier_1"]),
    }

    def add_row(check: str, tier: str | None, value: Any, reason: str) -> dict:
        weight = int(weights[check])
        points = tier_points.get(tier, 0)
        return {
            "check": check,
            "assessable": tier is not None,
            "raw_value": value,
            "tier": tier or "Not assessable",
            "tier_points": points if tier is not None else np.nan,
            "weight": weight,
            "contribution": weight * points if tier is not None else 0,
            "reason": reason,
        }

    checks = _raw_checks(layer1_result)
    record = get_valid_record(series)
    valid = pd.to_numeric(record, errors="coerce").dropna()
    rows: list[dict] = []

    missing = checks.get("missing_values")
    missing_assessable = _is_assessable(missing)
    missing = missing or {}
    missing_count = int(missing.get("value") or 0)
    missing_pct = (
        float(missing["internal_missing_percentage"])
        if missing_assessable
        and missing.get("internal_missing_percentage") is not None
        else np.nan
    )
    missing_settings = settings["missing_values"]
    tier2_missing = float(missing_settings["tier_2_minimum_percent"])
    tier1_missing = float(missing_settings["tier_1_above_percent"])
    missing_tier = (
        None if not missing_assessable or not np.isfinite(missing_pct) else
        "Tier 3" if missing_pct < tier2_missing
        else "Tier 2" if missing_pct <= tier1_missing else "Tier 1"
    )
    rows.append(add_row("missing_values", missing_tier, missing_pct,
                     f"Internal missingness = {missing_pct:.3f}%." if np.isfinite(missing_pct) else "No valid record."))

    gap = checks.get("long_gaps")
    gap_assessable = _is_assessable(gap)
    gap = gap or {}
    intervals = gap.get("all_missing_intervals", []) or []
    longest = int(gap.get("value") or 0)
    gap_settings = settings["long_gaps"]
    long_definition = int(gap_settings["long_gap_definition_days"])
    long_days = sum(int(item.get("missing_count", 0)) for item in intervals
                    if int(item.get("missing_count", 0)) > long_definition)
    long_share = long_days / missing_count if missing_count else 0.0
    if not gap_assessable:
        gap_tier = None
    elif (longest >= int(gap_settings["tier_1_minimum_days"]) or
            (missing_tier == "Tier 1" and long_share >= float(gap_settings["tier_1_missing_share"]))):
        gap_tier = "Tier 1"
    elif (longest >= int(gap_settings["tier_2_minimum_days"]) or
          (missing_tier == "Tier 2" and long_share >= float(gap_settings["tier_2_missing_share"]))):
        gap_tier = "Tier 2"
    else:
        gap_tier = "Tier 3"
    rows.append(add_row("long_gaps", gap_tier, longest,
                     f"Longest gap = {longest} days; {100*long_share:.1f}% of missing days are in >{long_definition}-day gaps."))

    duplicate = checks.get("duplicate_timestamps")
    duplicate_assessable = _is_assessable(duplicate)
    duplicate = duplicate or {}
    duplicate_value = int(duplicate.get("value") or 0)
    rows.append(add_row(
        "duplicate_timestamps",
        None if not duplicate_assessable else
        "Tier 1" if bool(duplicate.get("flag", duplicate_value > 0)) else "Tier 3",
        duplicate_value,
        f"{duplicate_value} unique duplicated date(s); "
        f"{int(duplicate.get('extra_duplicate_rows', 0))} extra row(s).",
    ))

    timestep = checks.get("timestep_consistency")
    timestep_assessable = _is_assessable(timestep)
    timestep = timestep or {}
    timestep_value = int(timestep.get("value") or 0)
    rows.append(add_row(
        "timestep_consistency",
        None if not timestep_assessable else
        "Tier 1" if bool(timestep.get("flag", timestep_value > 0)) else "Tier 3",
        timestep_value,
        f"{int(timestep.get('irregular_spacing_count', timestep_value))} irregular "
        f"unique-date interval(s); source out of order = "
        f"{bool(timestep.get('out_of_order', False))}.",
    ))

    negative = checks.get("negative_discharge")
    negative_assessable = _is_assessable(negative)
    negative = negative or {}
    negative_count = int(negative.get("value") or 0)
    negative_magnitude = float(negative.get("maximum_negative_magnitude") or 0.0)
    negative_settings = settings["negative_discharge"]
    low_flow_quantile = float(negative_settings["low_flow_reference_quantile"])
    reference_multiplier = float(negative_settings["tier_1_reference_multiplier"])
    fdc_q95 = (
        float(valid.clip(lower=0).quantile(low_flow_quantile))
        if not valid.empty else np.nan
    )
    tier1_threshold = fdc_q95 * reference_multiplier
    negative_tier = (
        None if not negative_assessable or not np.isfinite(fdc_q95)
        else "Tier 3" if negative_count == 0
        else "Tier 1" if negative_magnitude >= tier1_threshold else "Tier 2"
    )
    rows.append(add_row("negative_discharge", negative_tier, negative_magnitude,
                     f"Material negative observations = {negative_count}; maximum "
                     f"magnitude = {negative_magnitude:.3f}; low-flow reference = "
                     f"{fdc_q95:.3f}; Tier 1 threshold = {tier1_threshold:.3f}."))

    spike = checks.get("spike_dip")
    spike_assessable = _is_assessable(spike)
    spike = spike or {}
    unresolved = int(spike.get("value") or 0)
    if layer2_result is not None:
        crosscheck = layer2_result.get("spike_peak_crosscheck", pd.DataFrame())
        if not crosscheck.empty and "crosscheck_status" in crosscheck:
            unresolved = int(crosscheck["crosscheck_status"].ne("plausible_event_context").sum())
    spike_tier1 = int(settings["spike_dip"]["tier_1_minimum_unresolved_count"])
    spike_tier = (
        None if not spike_assessable
        else "Tier 3" if unresolved == 0
        else "Tier 2" if unresolved < spike_tier1
        else "Tier 1"
    )
    rows.append(add_row("spike_dip", spike_tier, unresolved,
                     f"{unresolved} unresolved spike/dip candidate(s); Tier 1 "
                     f"begins at {spike_tier1}."))

    plateau = checks.get("low_variability")
    plateau_assessable = _is_assessable(plateau)
    plateau = plateau or {}
    periods = plateau.get("plateau_periods", []) or []
    longest_plateau = max((int(item.get("calendar_duration_days", 0)) for item in periods), default=0)
    plateau_tier1 = int(settings["low_variability"]["tier_1_minimum_days"])
    plateau_tier = (
        None if not plateau_assessable
        else "Tier 3" if longest_plateau == 0
        else "Tier 2" if longest_plateau < plateau_tier1
        else "Tier 1"
    )
    rows.append(add_row("low_variability", plateau_tier, longest_plateau,
                     f"Longest retained non-zero plateau = {longest_plateau} days; "
                     f"Tier 1 begins at {plateau_tier1} days."))

    step = checks.get("step_shift")
    step_assessable = _is_assessable(step)
    step = step or {}
    boundaries = step.get("regime_boundaries", []) or []
    step_score = float(step.get("step_shift_score", 0.0))
    step_tier = step.get("composite_tier", "Tier 3") if step_assessable else None
    rows.append(add_row(
        "step_shift", step_tier, step_score,
        f"Composite across {len(boundaries)} retained boundary(ies) = {step_score:.3f}."
    ))

    epoch = checks.get("epoch_drift")
    epoch_assessable = _is_assessable(epoch)
    epoch = epoch or {}
    epoch_tier = epoch.get("tier") if epoch_assessable else None
    stable_fraction = epoch.get("stable_year_fraction")
    rows.append(add_row("epoch_drift", epoch_tier, stable_fraction,
                     epoch.get("message", "Epoch stability not assessed.")))

    all_component_names = [row["check"] for row in rows]
    components = pd.DataFrame(rows)
    components = components.loc[
        components["check"].map(
            lambda check: bool(
                layer1_settings.get(check, {}).get("enabled", True)
            )
        )
    ].reset_index(drop=True)
    enabled_names = components["check"].tolist()
    disabled_names = [
        check for check in all_component_names if check not in enabled_names
    ]
    enabled_count = len(enabled_names)
    total_count = len(all_component_names)
    assessment_scope = "Full" if enabled_count == total_count else "Focused"
    score = int(components["contribution"].sum())
    assessable_count = int(components["assessable"].sum())
    assessable = components.loc[components["assessable"]]
    maximum_score = int(
        (assessable["weight"] * max(tier_points.values())).sum()
    )
    score_percent = (
        100.0 * score / maximum_score if maximum_score > 0 else np.nan
    )
    classification = settings["classification"]
    overall = (
        "Not assessable" if assessable_count == 0
        else "Needs review" if score_percent >= float(
            classification["needs_review_minimum_percent"]
        )
        else "Minor concerns" if score_percent >= float(
            classification["minor_concerns_minimum_percent"]
        )
        else "No review needed"
    )
    if assessment_scope == "Full":
        scope_conclusion = f"Overall Layer 1 assessment: {overall}."
    else:
        focused_outcome = {
            "Needs review": "Review recommended within the selected checks",
            "Minor concerns": "Minor concerns within the selected checks",
            "No review needed": "No concerns detected within the selected checks",
            "Not assessable": "The selected checks could not be assessed",
        }[overall]
        scope_conclusion = f"Focused Layer 1 assessment: {focused_outcome}."
    unavailable = components.loc[~components["assessable"], "check"].tolist()
    evidence_coverage = (
        100.0 * assessable_count / enabled_count if enabled_count else np.nan
    )
    return {
        "components": components,
        "configuration_used": settings,
        "summary": pd.DataFrame([{
            "layer1_score": score, "layer1_class": overall,
            "layer1_score_percent": score_percent,
            "maximum_assessable_score": maximum_score,
            "assessment_scope": assessment_scope,
            "scope_conclusion": scope_conclusion,
            "enabled_check_count": enabled_count,
            "total_composite_check_count": total_count,
            "disabled_check_count": len(disabled_names),
            "enabled_checks": "; ".join(enabled_names),
            "disabled_checks": "; ".join(disabled_names),
            "assessable_checks": assessable_count,
            "evidence_coverage_percent": evidence_coverage,
            "assessment_incomplete": bool(unavailable),
            "unavailable_checks": "; ".join(unavailable),
        }]),
    }


def _tier_similarity(value: float, settings: Mapping[str, Any]) -> str | None:
    if not np.isfinite(value):
        return None
    return (
        "Tier 3" if value >= float(settings["similarity_tier3_minimum"])
        else "Tier 2" if value >= float(settings["similarity_tier2_minimum"])
        else "Tier 1"
    )


def _tier_days(
    value: float, kind: str, settings: Mapping[str, Any]
) -> str | None:
    if not np.isfinite(value):
        return None
    if kind == "peak":
        return (
            "Tier 3"
            if value <= float(settings["time_to_peak_tier3_max_days"])
            else "Tier 2"
            if value < float(settings["time_to_peak_tier1_min_days"])
            else "Tier 1"
        )
    return (
        "Tier 3"
        if value <= float(settings["event_duration_tier3_max_days"])
        else "Tier 2"
        if value <= float(settings["event_duration_tier2_max_days"])
        else "Tier 1"
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if len(left) < 2:
        return np.nan
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else np.nan


def _resample(values: pd.Series, size: int = 100) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(clean) < 2:
        return np.full(size, np.nan)
    return np.interp(
        np.linspace(0, 1, size),
        np.linspace(0, 1, len(clean)),
        clean,
    )


def _annual_cosine(reference: pd.DataFrame, candidate: pd.DataFrame, column: str) -> float:
    merged = reference[["year", column]].merge(candidate[["year", column]], on="year", suffixes=("_r", "_c")).dropna()
    if len(merged) >= 2:
        return _cosine(merged[f"{column}_r"].to_numpy(float), merged[f"{column}_c"].to_numpy(float))
    return _cosine(_resample(reference[column]), _resample(candidate[column]))


def _inverse_jsd(reference: pd.Series, candidate: pd.Series) -> float:
    left = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    right = pd.to_numeric(candidate, errors="coerce").dropna().to_numpy(float)
    if len(left) < 2 or len(right) < 2:
        return np.nan
    edges = np.histogram_bin_edges(np.concatenate([left, right]), bins="auto")
    if len(edges) < 3:
        return 1.0 if np.allclose(left.mean(), right.mean()) else 0.0
    p = np.histogram(left, bins=edges)[0].astype(float) + np.finfo(float).eps
    q = np.histogram(right, bins=edges)[0].astype(float) + np.finfo(float).eps
    p, q = p / p.sum(), q / q.sum()
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * np.sum(p * np.log2(p / midpoint)) + 0.5 * np.sum(q * np.log2(q / midpoint))
    return float(np.clip(1.0 - np.sqrt(divergence), 0.0, 1.0))


def _event_curve(series: pd.Series, diagnostics: dict) -> np.ndarray:
    table = diagnostics.get("representative_event", pd.DataFrame())
    if table.empty:
        return np.full(100, np.nan)
    event = table.iloc[0]
    return _resample(series.loc[event["event_start"]:event["event_end"]])


def score_layer2_comparison(
    reference_series: pd.Series,
    candidate_series: pd.Series,
    reference: dict,
    candidate: dict,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate the eight agreed equal-weight Layer 2 components."""
    supplied = {} if config is None else dict(config).get("comparison", {})
    settings = merge_config(DEFAULT_LAYER2_CONFIG["comparison"], supplied)
    weights = settings["weights"]
    enabled_components = settings["components"]
    rows: list[dict] = []
    def add(component: str, metric: str, value: float, tier: str | None, bias=np.nan):
        if not bool(enabled_components[component]):
            return
        weight = int(weights[component])
        rows.append({"component": component, "metric": metric, "value": value,
                     "median_bias": bias, "assessable": tier is not None,
                     "tier": tier or "Not assessable",
                     "tier_points": LAYER2_TIER_POINTS.get(tier, np.nan),
                     "weight": weight,
                     "contribution": weight * LAYER2_TIER_POINTS.get(tier, 0)})

    js = _inverse_jsd(reference_series, candidate_series)
    add("flow_behaviour", "Inverse JSD", js, _tier_similarity(js, settings))
    ref_annual, cand_annual = reference["annual_signatures"], candidate["annual_signatures"]
    for column, component in [("flashiness_index", "annual_flashiness_shape"), ("baseflow_index", "annual_baseflow_shape")]:
        value = _annual_cosine(ref_annual, cand_annual, column)
        ref_med, cand_med = ref_annual[column].median(), cand_annual[column].median()
        add(component, "cosine similarity", value, _tier_similarity(value, settings), cand_med - ref_med)

    ref_profile = reference["seasonality_profile"].set_index("month")["median"].reindex(range(1, 13))
    cand_profile = candidate["seasonality_profile"].set_index("month")["median"].reindex(range(1, 13))
    seasonal = _cosine(ref_profile.to_numpy(float), cand_profile.to_numpy(float))
    add("seasonal_profile_shape", "cosine similarity", seasonal, _tier_similarity(seasonal, settings), cand_profile.median() - ref_profile.median())
    if ref_profile.notna().any() and cand_profile.notna().any():
        wet = min(abs(int(ref_profile.idxmax()) - int(cand_profile.idxmax())), 12 - abs(int(ref_profile.idxmax()) - int(cand_profile.idxmax())))
        dry = min(abs(int(ref_profile.idxmin()) - int(cand_profile.idxmin())), 12 - abs(int(ref_profile.idxmin()) - int(cand_profile.idxmin())))
        timing = float(max(wet, dry))
        timing_tier = (
            "Tier 3"
            if timing <= float(settings["seasonal_timing_tier3_max_months"])
            else "Tier 2"
            if timing <= float(settings["seasonal_timing_tier2_max_months"])
            else "Tier 1"
        )
    else:
        timing, timing_tier = np.nan, None
    add("seasonal_timing", "maximum wet/dry circular month separation", timing, timing_tier)

    def event_median(frame: dict, name: str) -> float:
        events = frame["hydrograph_events"]
        return float(events[name].median()) if not events.empty else np.nan
    peak_difference = abs(event_median(candidate, "time_to_peak_days") - event_median(reference, "time_to_peak_days"))
    duration_difference = abs(event_median(candidate, "event_duration_days") - event_median(reference, "event_duration_days"))
    add("event_time_to_peak", "absolute median difference (days)", peak_difference, _tier_days(peak_difference, "peak", settings))
    add("event_duration", "absolute median difference (days)", duration_difference, _tier_days(duration_difference, "duration", settings))
    event_similarity = _cosine(_event_curve(reference_series, reference), _event_curve(candidate_series, candidate))
    add("representative_event_shape", "cosine similarity", event_similarity, _tier_similarity(event_similarity, settings))

    components = pd.DataFrame(rows)
    if components.empty:
        return {
            "components": components,
            "configuration_used": settings,
            "summary": pd.DataFrame([{
                "layer2_score": 0,
                "layer2_score_percent": np.nan,
                "maximum_assessable_score": 0,
                "layer2_class": "Not assessable",
                "assessable_components": 0,
                "assessment_incomplete": True,
                "unavailable_components": "No comparison components enabled",
            }]),
        }
    score = int(components["contribution"].sum())
    assessable_count = int(components["assessable"].sum())
    enabled_count = len(components)
    minimum_components = min(
        int(settings["minimum_assessable_components"]), enabled_count
    )
    assessable = components.loc[components["assessable"]]
    maximum_score = int(
        (assessable["weight"] * max(LAYER2_TIER_POINTS.values())).sum()
    )
    score_percent = (
        100.0 * score / maximum_score if maximum_score > 0 else np.nan
    )
    overall = (
        "Not assessable"
        if assessable_count < minimum_components
        else "Similar"
        if score_percent <= float(settings["similar_maximum_percent"])
        else "Review"
        if score_percent <= float(settings["review_maximum_percent"])
        else "Strong review"
    )
    unavailable = components.loc[~components["assessable"], "component"].tolist()
    return {
        "components": components,
        "configuration_used": settings,
        "summary": pd.DataFrame([{
            "layer2_score": score,
            "layer2_score_percent": score_percent,
            "maximum_assessable_score": maximum_score,
            "layer2_class": overall,
            "assessable_components": assessable_count,
            "assessment_incomplete": assessable_count < minimum_components,
            "unavailable_components": "; ".join(unavailable),
        }]),
    }


__all__ = ["score_layer1", "score_layer2_comparison"]
