"""Run the Layer 3 network-context assessment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from trihydra.settings.defaults import (
    DEFAULT_LAYER1_CONFIG,
    DEFAULT_LAYER2_CONFIG,
    DEFAULT_LAYER3_CONFIG,
    merge_config,
)
from trihydra.layer1.diagnostics import run_layer1_diagnostics
from trihydra.layer2.diagnostics import run_layer2_diagnostics
from trihydra.layer3.analogue_comparison import (
    AnalogueContextResult,
    compare_analogue_context,
)
from trihydra.layer3.evidence import (
    StationContextEvidence,
    build_station_context_evidence,
)
from trihydra.layer3.local_comparison import (
    LocalContextResult,
    compare_local_context,
)
from trihydra.layer3.metadata import (
    ContextValidationResult,
    attach_climate_context,
    read_context_metadata,
)
from trihydra.layer3.peers import ContextPeerGroups, select_peer_groups
from trihydra.layer3.summary import Layer3Summary, summarise_layer3


@dataclass(frozen=True)
class Layer3StationResult:
    """All inspectable Layer 3 results for one target station."""

    station_id: str
    metadata: dict[str, Any] | None
    peer_groups: ContextPeerGroups
    local_comparison: LocalContextResult
    local_behaviour_comparison: AnalogueContextResult
    analogue_comparison: AnalogueContextResult
    summary: Layer3Summary
    upstream_review_triggered: bool


@dataclass(frozen=True)
class Layer3RunResult:
    """Network results plus a compact station table."""

    metadata_validation: ContextValidationResult
    enriched_metadata: pd.DataFrame
    station_results: dict[str, Layer3StationResult]
    evidence_cache: dict[str, StationContextEvidence]
    run_summary: pd.DataFrame


def _normalise_series(series: pd.Series) -> pd.Series:
    """Ensure the adapter receives numeric values on a datetime index."""
    result = pd.to_numeric(series.copy(deep=True), errors="coerce")
    result.index = pd.to_datetime(result.index, errors="raise")
    return result.sort_index()


def _build_evidence_once(
    station_ids: set[str],
    series_by_station: Mapping[str, pd.Series],
    evidence_cache: Mapping[str, StationContextEvidence] | None,
    layer1_config: Mapping[str, Any],
    layer2_config: Mapping[str, Any],
) -> dict[str, StationContextEvidence]:
    """Reuse supplied evidence and calculate every remaining station once."""
    evidence = dict(evidence_cache or {})
    for station_id in sorted(station_ids):
        if station_id in evidence or station_id not in series_by_station:
            continue
        series = _normalise_series(series_by_station[station_id])
        before = series.copy(deep=True)
        layer1 = run_layer1_diagnostics(series, config=layer1_config)
        layer2 = run_layer2_diagnostics(
            series, layer1_result=layer1, config=layer2_config
        )
        evidence[station_id] = build_station_context_evidence(
            station_id, series, layer1, layer2, layer1_config=layer1_config
        )
        pd.testing.assert_series_equal(series, before)
    return evidence


def run_layer3_context(
    series_by_station: Mapping[str, pd.Series],
    context_path: str | Path,
    climate_raster_path: str | Path | None = None,
    climate_legend_path: str | Path | None = None,
    *,
    target_station_ids: Sequence[str] | None = None,
    layer3_config: Mapping[str, Any] | None = None,
    layer1_config: Mapping[str, Any] | None = None,
    layer2_config: Mapping[str, Any] | None = None,
    evidence_cache: Mapping[str, StationContextEvidence] | None = None,
) -> Layer3RunResult:
    """Run Layer 3 by reusing or calculating the required Layer 1/2 evidence."""
    settings = merge_config(DEFAULT_LAYER3_CONFIG, layer3_config)
    if climate_raster_path is None or climate_legend_path is None:
        from trihydra.layer3.climate import bundled_climate_paths

        bundled_raster, bundled_legend = bundled_climate_paths()
        climate_raster_path = climate_raster_path or bundled_raster
        climate_legend_path = climate_legend_path or bundled_legend
    l1_settings = merge_config(DEFAULT_LAYER1_CONFIG, layer1_config)
    l2_settings = merge_config(DEFAULT_LAYER2_CONFIG, layer2_config)

    validation = read_context_metadata(context_path, settings["metadata"])
    metadata = attach_climate_context(
        validation.stations, climate_raster_path, climate_legend_path
    )
    available_ids = set(map(str, series_by_station))
    # A station without an input series cannot be an evidence peer.
    metadata = metadata.loc[metadata.station_id.isin(available_ids)].reset_index(drop=True)

    targets = (
        list(map(str, target_station_ids))
        if target_station_ids is not None
        else metadata.station_id.astype(str).tolist()
    )
    peer_groups: dict[str, ContextPeerGroups] = {}
    required_ids = set(targets)
    for station_id in targets:
        groups = select_peer_groups(
            metadata,
            station_id,
            settings["local_peers"],
            settings["analogue_peers"],
            str(settings.get("series_type", "observation")),
        )
        peer_groups[station_id] = groups
        if groups.local.is_assessable:
            required_ids.update(groups.local.peers.station_id.astype(str))
        if groups.analogues.is_assessable:
            required_ids.update(groups.analogues.peers.station_id.astype(str))

    evidence = _build_evidence_once(
        required_ids, series_by_station, evidence_cache, l1_settings, l2_settings
    )
    results: dict[str, Layer3StationResult] = {}
    rows: list[dict[str, Any]] = []
    for station_id in targets:
        groups = peer_groups[station_id]
        if station_id not in evidence:
            rows.append({
                "station_id": station_id,
                "status": "not_assessed",
                "local_classification": "Not assessed",
                "analogue_classification": "Not assessed",
                "reason": "Target time series or diagnostic evidence is unavailable.",
            })
            continue

        local_ids = (
            groups.local.peers.station_id.astype(str).tolist()
            if groups.local.is_assessable else []
        )
        analogue_ids = (
            groups.analogues.peers.station_id.astype(str).tolist()
            if groups.analogues.is_assessable else []
        )
        local_evidence = {key: evidence[key] for key in local_ids if key in evidence}
        analogue_evidence = {key: evidence[key] for key in analogue_ids if key in evidence}
        target = evidence[station_id]
        local = compare_local_context(target, local_evidence, settings["comparison"])
        local_behaviour = compare_analogue_context(
            target, local_evidence, settings["comparison"]
        )
        analogues = compare_analogue_context(
            target, analogue_evidence, settings["comparison"]
        )
        summary = summarise_layer3(
            local,
            analogues,
            settings["comparison"],
            local_behaviour=local_behaviour,
        )
        metadata_rows = metadata.loc[metadata.station_id == station_id]
        station_metadata = (
            None if metadata_rows.empty else metadata_rows.iloc[0].to_dict()
        )
        results[station_id] = Layer3StationResult(
            station_id, station_metadata, groups, local, local_behaviour,
            analogues, summary, target.layer1_review_class == "Needs review"
        )
        rows.append({
            "station_id": station_id,
            "status": "completed",
            "local_peer_count": len(groups.local.peers),
            "local_similarity_percent": summary.local.similarity_percent,
            "local_classification": summary.local.classification,
            "local_evidence_coverage_percent": summary.local.evidence_coverage_percent,
            "analogue_peer_count": len(groups.analogues.peers),
            "analogue_similarity_percent": summary.analogues.similarity_percent,
            "analogue_classification": summary.analogues.classification,
            "analogue_evidence_coverage_percent": summary.analogues.evidence_coverage_percent,
            "layer3_composite_similarity_percent": summary.combined_similarity_percent,
            "layer3_composite_classification": summary.combined_classification,
            "upstream_review_triggered": target.layer1_review_class == "Needs review",
            "context_report_recommended": bool(
                summary.combined_similarity_percent is not None
                and summary.combined_similarity_percent
                >= summary.report_minimum_similarity_percent
                and target.layer1_review_class == "Needs review"
            ),
            "nearby_contribution_percent": summary.combined_local_contribution_percent,
            "comparable_catchment_contribution_percent": (
                summary.combined_comparable_contribution_percent
            ),
            "reason": "",
        })

    return Layer3RunResult(
        validation,
        metadata,
        results,
        evidence,
        pd.DataFrame(rows),
    )


__all__ = [
    "Layer3RunResult",
    "Layer3StationResult",
    "run_layer3_context",
]
