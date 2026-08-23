"""Summarise nearby-gauge and comparable-catchment evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from trihydra.layer3.analogue_comparison import AnalogueContextResult
from trihydra.layer3.local_comparison import (
    ContextCheckResult,
    LocalContextResult,
)


@dataclass(frozen=True)
class ContextSummary:
    """One score with its evidence coverage and plain-language meaning."""

    context_type: str
    classification: str
    similarity_percent: float | None
    supported_check_count: int
    assessed_check_count: int
    total_check_count: int
    evidence_coverage_percent: float
    check_summary: pd.DataFrame
    comment: str


@dataclass(frozen=True)
class Layer3Summary:
    """Separate context summaries plus their configured combined result."""

    target_station_id: str
    local: ContextSummary
    analogues: ContextSummary
    combined_similarity_percent: float | None
    combined_classification: str
    combined_local_contribution_percent: float | None
    combined_comparable_contribution_percent: float | None
    combined_check_summary: pd.DataFrame
    local_weight: float
    comparable_weight: float
    partial_minimum_percent: float
    similar_minimum_percent: float
    report_minimum_similarity_percent: float


def _combined_context(
    local: LocalContextResult,
    local_behaviour: AnalogueContextResult | None,
    analogues: AnalogueContextResult,
    config: Mapping[str, float],
):
    """Combine distinct checks while retaining both weighted contributions."""
    local_weight = max(float(config.get("local_context_weight", 0.70)), 0.0)
    comparable_weight = max(
        float(config.get("comparable_catchment_weight", 0.30)), 0.0
    )
    total_weight = local_weight + comparable_weight
    if total_weight <= 0:
        local_weight = comparable_weight = 0.50
    else:
        local_weight /= total_weight
        comparable_weight /= total_weight

    rows = []
    local_only = [local.peak_timing, local.step_shift_timing, local.epoch_behaviour]
    for check in local_only:
        fraction = check.similarity_fraction
        rows.append({
            "check": check.check,
            "local_support_fraction": fraction,
            "comparable_support_fraction": None,
            "local_contribution_percent": None if fraction is None else 100 * fraction,
            "comparable_contribution_percent": 0.0 if fraction is not None else None,
            "combined_similarity_percent": None if fraction is None else 100 * fraction,
            "local_supporting_peer_count": check.supporting_peer_count,
            "local_assessed_peer_count": check.assessed_peer_count,
            "comparable_supporting_peer_count": 0,
            "comparable_assessed_peer_count": 0,
        })

    if local_behaviour is not None:
        local_shared = [
            local_behaviour.flashiness, local_behaviour.baseflow,
            local_behaviour.zero_flow, local_behaviour.seasonality_shape,
            local_behaviour.event_shape, local_behaviour.event_time_to_peak,
            local_behaviour.event_duration,
        ]
        comparable_shared = [
            analogues.flashiness, analogues.baseflow, analogues.zero_flow,
            analogues.seasonality_shape, analogues.event_shape,
            analogues.event_time_to_peak, analogues.event_duration,
        ]
        for nearby, comparable in zip(local_shared, comparable_shared):
            local_fraction = nearby.similarity_fraction
            comparable_fraction = comparable.similarity_fraction
            if local_fraction is None and comparable_fraction is None:
                local_contribution = comparable_contribution = combined = None
            elif comparable_fraction is None:
                # Missing comparable-catchment evidence is not disagreement.
                # Use all available weight for the nearby-gauge evidence.
                local_contribution = 100 * float(local_fraction)
                comparable_contribution = None
                combined = local_contribution
            elif local_fraction is None:
                # Distant comparable catchments cannot replace missing local
                # evidence, so their configured influence remains capped.
                local_contribution = None
                comparable_contribution = (
                    100 * float(comparable_fraction) * comparable_weight
                )
                combined = comparable_contribution
            else:
                local_contribution = 100 * float(local_fraction) * local_weight
                comparable_contribution = (
                    100 * float(comparable_fraction) * comparable_weight
                )
                combined = local_contribution + comparable_contribution
            rows.append({
                "check": nearby.check,
                "local_support_fraction": local_fraction,
                "comparable_support_fraction": comparable_fraction,
                "local_contribution_percent": local_contribution,
                "comparable_contribution_percent": comparable_contribution,
                "combined_similarity_percent": combined,
                "local_supporting_peer_count": nearby.supporting_peer_count,
                "local_assessed_peer_count": nearby.assessed_peer_count,
                "comparable_supporting_peer_count": comparable.supporting_peer_count,
                "comparable_assessed_peer_count": comparable.assessed_peer_count,
            })

    table = pd.DataFrame(rows)
    assessed = table.dropna(subset=["combined_similarity_percent"])
    if assessed.empty:
        return None, "Not assessed", None, None, table, local_weight, comparable_weight
    score = float(assessed["combined_similarity_percent"].mean())
    local_contribution = float(assessed["local_contribution_percent"].fillna(0).mean())
    comparable_contribution = float(
        assessed["comparable_contribution_percent"].fillna(0).mean()
    )
    classification = _classification(
        score,
        float(config.get("similar_minimum_percent", 75.0)),
        float(config.get("partial_minimum_percent", 40.0)),
    )
    return (
        score, classification, local_contribution, comparable_contribution,
        table, local_weight, comparable_weight,
    )


def _classification(score: float, similar_minimum: float, partial_minimum: float) -> str:
    if score >= similar_minimum:
        return "Strong agreement"
    if score >= partial_minimum:
        return "Moderate agreement"
    return "Low agreement"


def _summarise_checks(
    context_type: str,
    checks: Sequence[ContextCheckResult],
    config: Mapping[str, float],
) -> ContextSummary:
    rows = []
    for check in checks:
        included = check.status in {"supported", "not_supported"}
        rows.append({
            "check": check.check,
            "status": check.status,
            "peer_support_fraction": check.similarity_fraction,
            "supporting_peer_count": check.supporting_peer_count,
            "assessed_peer_count": check.assessed_peer_count,
            "included_in_score": included,
            "comment": check.message,
        })
    table = pd.DataFrame(rows)
    assessed = table.loc[table["included_in_score"]]
    total = len(table)
    coverage = 100.0 * len(assessed) / total if total else 0.0
    if assessed.empty:
        return ContextSummary(
            context_type, "Not assessed", None, 0, 0, total, coverage, table,
            f"{context_type.capitalize()} context was not assessed because no checks had sufficient evidence.",
        )

    supported = int((assessed["status"] == "supported").sum())
    # Preserve continuous agreement values. Passing a threshold is not the
    # same thing as perfect agreement.
    score = 100.0 * float(assessed["peer_support_fraction"].mean())
    classification = _classification(
        score,
        float(config.get("similar_minimum_percent", 75.0)),
        float(config.get("partial_minimum_percent", 40.0)),
    )
    unavailable = total - len(assessed)
    comment = (
        f"{context_type.capitalize()} context is {classification.lower()} "
        f"from the mean hydrological agreement across {len(assessed)} assessed "
        f"checks; {unavailable} check(s) were unavailable or not applicable."
    )
    return ContextSummary(
        context_type, classification, score, supported, len(assessed), total,
        coverage, table, comment,
    )


def summarise_layer3(
    local: LocalContextResult,
    analogues: AnalogueContextResult,
    config: Mapping[str, float],
    *,
    local_behaviour: AnalogueContextResult | None = None,
) -> Layer3Summary:
    """Create separate context scores and their configured combined result."""
    local_checks = [local.peak_timing, local.step_shift_timing, local.epoch_behaviour]
    if local_behaviour is not None:
        local_checks.extend([
            local_behaviour.flashiness,
            local_behaviour.baseflow,
            local_behaviour.zero_flow,
            local_behaviour.seasonality_shape,
            local_behaviour.event_shape,
            local_behaviour.event_time_to_peak,
            local_behaviour.event_duration,
        ])
    analogue_checks = [
        analogues.flashiness,
        analogues.baseflow,
        analogues.zero_flow,
        analogues.seasonality_shape,
        analogues.event_shape,
        analogues.event_time_to_peak,
        analogues.event_duration,
    ]
    combined = _combined_context(local, local_behaviour, analogues, config)
    return Layer3Summary(
        local.target_station_id,
        _summarise_checks("local", local_checks, config),
        _summarise_checks("analogue", analogue_checks, config),
        *combined,
        float(config.get("partial_minimum_percent", 40.0)),
        float(config.get("similar_minimum_percent", 75.0)),
        float(config.get("report_minimum_similarity_percent", 50.0)),
    )


__all__ = ["ContextSummary", "Layer3Summary", "summarise_layer3"]
