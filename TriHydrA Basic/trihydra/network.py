"""Independent multi-station runner used when Layer 3 context is requested."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from trihydra.layer3.evidence import build_station_context_evidence
from trihydra.layer3.orchestrator import run_layer3_context
from trihydra.pipeline import run_trihydra
from trihydra.reporting import build_station_summary
from trihydra.result import TriHydrANetworkResult
from trihydra.settings.defaults import DEFAULT_CONFIG, merge_config


def _layer3_contract(result, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Translate Layer 3 internals into user-facing contextual evidence."""
    summary = result.summary
    metrics = {
        "context_agreement_score_percent": summary.combined_similarity_percent,
        "context_agreement_class": summary.combined_classification,
        "nearby_gauge_agreement_percent": summary.local.similarity_percent,
        "nearby_gauge_agreement_class": summary.local.classification,
        "nearby_checks_supported": summary.local.supported_check_count,
        "nearby_checks_assessed": summary.local.assessed_check_count,
        "comparable_catchment_agreement_percent": summary.analogues.similarity_percent,
        "comparable_catchment_agreement_class": summary.analogues.classification,
        "comparable_checks_supported": summary.analogues.supported_check_count,
        "comparable_checks_assessed": summary.analogues.assessed_check_count,
        "nearby_gauge_count": int(len(result.peer_groups.local.peers)),
        "comparable_catchment_count": int(len(result.peer_groups.analogues.peers)),
    }
    comparison = settings["comparison"]
    thresholds = {
        "nearby_search_radius_km": settings["local_peers"]["maximum_search_radius_km"],
        "comparable_search_radius_km": settings["analogue_peers"]["maximum_search_radius_km"],
        "maximum_catchment_area_ratio": settings["analogue_peers"]["maximum_catchment_area_ratio"],
        "nearby_context_weight": summary.local_weight,
        "comparable_catchment_weight": summary.comparable_weight,
        "minimum_peer_consensus_fraction": comparison["peer_consensus_fraction"],
        "similar_minimum_percent": comparison["similar_minimum_percent"],
        "partial_minimum_percent": comparison["partial_minimum_percent"],
    }
    check_evidence = summary.combined_check_summary.copy()
    def agreement_class(value):
        if value is None or pd.isna(value):
            return "Not assessed"
        if float(value) >= float(comparison["similar_minimum_percent"]):
            return "Strong agreement"
        if float(value) >= float(comparison["partial_minimum_percent"]):
            return "Moderate agreement"
        return "Low agreement"
    check_evidence["agreement_class"] = check_evidence[
        "combined_similarity_percent"
    ].map(agreement_class)
    check_evidence = check_evidence.rename(columns={
        "local_support_fraction": "nearby_median_similarity",
        "comparable_support_fraction": "comparable_catchment_median_similarity",
        "local_contribution_percent": "nearby_weighted_contribution_percent",
        "comparable_contribution_percent": "comparable_weighted_contribution_percent",
        "combined_similarity_percent": "combined_agreement_percent",
        "local_supporting_peer_count": "nearby_peers_meeting_rule",
        "local_assessed_peer_count": "nearby_peers_assessed",
        "comparable_supporting_peer_count": "comparable_peers_meeting_rule",
        "comparable_assessed_peer_count": "comparable_peers_assessed",
    })

    def detail_frames(context_name: str, checks) -> list[pd.DataFrame]:
        """Keep the numerical evidence used for every peer-level decision."""
        detailed = []
        for check in checks:
            table = check.details
            if table is None or table.empty:
                continue
            frame = table.copy()
            frame.insert(0, "check", check.check)
            frame.insert(0, "context_group", context_name)
            frame.insert(0, "evidence_type", "peer_metric_comparison")
            detailed.append(frame)
        return detailed

    local_checks = [
        result.local_comparison.peak_timing,
        result.local_comparison.step_shift_timing,
        result.local_comparison.epoch_behaviour,
        result.local_behaviour_comparison.flashiness,
        result.local_behaviour_comparison.baseflow,
        result.local_behaviour_comparison.zero_flow,
        result.local_behaviour_comparison.seasonality_shape,
        result.local_behaviour_comparison.event_shape,
        result.local_behaviour_comparison.event_time_to_peak,
        result.local_behaviour_comparison.event_duration,
    ]
    comparable_checks = [
        result.analogue_comparison.flashiness,
        result.analogue_comparison.baseflow,
        result.analogue_comparison.zero_flow,
        result.analogue_comparison.seasonality_shape,
        result.analogue_comparison.event_shape,
        result.analogue_comparison.event_time_to_peak,
        result.analogue_comparison.event_duration,
    ]
    frames = []
    for evidence_type, table in (
        ("context_check", check_evidence),
        ("nearby_gauge", result.peer_groups.local.peers),
        ("comparable_catchment", result.peer_groups.analogues.peers),
    ):
        if table is None or table.empty:
            continue
        frame = table.copy()
        frame.insert(0, "evidence_type", evidence_type)
        frames.append(frame)
    frames.extend(detail_frames("nearby_gauge", local_checks))
    frames.extend(detail_frames("comparable_catchment", comparable_checks))
    return {
        "result": result,
        "summary_metrics": metrics,
        "thresholds_used": thresholds,
        "evidence": pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(),
    }


def run_trihydra_network(
    series_by_station: Mapping[str, pd.Series],
    *,
    context_path: str | Path,
    target_station_ids: Sequence[str] | None = None,
    unit: str = "source units",
    config: Mapping[str, Any] | None = None,
) -> TriHydrANetworkResult:
    """Run Layers 1–2 once per station, then assess Layer 3 context."""
    if len(series_by_station) < 2:
        raise ValueError("Layer 3 requires at least two station series.")
    effective = merge_config(DEFAULT_CONFIG, config)
    original = {
        str(station_id): series.copy(deep=True)
        for station_id, series in series_by_station.items()
    }
    station_results = {
        station_id: run_trihydra(
            series, station_id=station_id, unit=unit, config=effective
        )
        for station_id, series in original.items()
    }
    evidence = {
        station_id: build_station_context_evidence(
            station_id, result.series, result.layer1, result.layer2,
            layer1_config=effective["layer1"],
        )
        for station_id, result in station_results.items()
    }
    layer3_run = run_layer3_context(
        original,
        context_path,
        target_station_ids=target_station_ids,
        layer3_config=effective["layer3"],
        layer1_config=effective["layer1"],
        layer2_config=effective["layer2"],
        evidence_cache=evidence,
    )
    for station_id, layer3_result in layer3_run.station_results.items():
        contract = _layer3_contract(layer3_result, effective["layer3"])
        station_results[station_id].layer3 = contract
        station_results[station_id].summary = build_station_summary(
            station_results[station_id].station,
            layer1=station_results[station_id].layer1,
            layer2=station_results[station_id].layer2,
            comparison=station_results[station_id].comparison,
            layer3=contract,
        )
    summary = pd.concat(
        [result.summary for result in station_results.values()], ignore_index=True
    )
    for station_id, before in original.items():
        pd.testing.assert_series_equal(series_by_station[station_id], before)
    return TriHydrANetworkResult(
        station_results=station_results,
        layer3_run=layer3_run,
        summary=summary,
        series_by_station=original,
        configuration_used=effective,
    )


__all__ = ["run_trihydra_network"]
