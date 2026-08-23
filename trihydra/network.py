"""Independent multi-station runner used when Layer 3 context is requested."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from trihydra.layer3.evidence import build_station_context_evidence
from trihydra.layer3.orchestrator import run_layer3_context
from trihydra.io.models import StationData
from trihydra.pipeline import run_trihydra, station_from_series
from trihydra.reporting import build_station_summary, station_requires_review
from trihydra.result import TriHydrABatchResult, TriHydrANetworkResult, TriHydrAResult
from trihydra.settings.defaults import DEFAULT_CONFIG, merge_config


BatchProgress = Callable[[str, Mapping[str, object]], None]


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


def _context_evidence_for_results(
    station_results: Mapping[str, TriHydrAResult],
    layer1_config: Mapping[str, Any],
) -> dict:
    """Build Layer 3 evidence from already-completed station diagnostics."""
    return {
        station_id: build_station_context_evidence(
            station_id,
            result.station.obs,
            result.layer1,
            result.layer2,
            layer1_config=layer1_config,
        )
        for station_id, result in station_results.items()
    }


def _attach_layer3_contracts(
    station_results: Mapping[str, TriHydrAResult],
    layer3_run,
    layer3_settings: Mapping[str, Any],
    *,
    update_station_metadata: bool,
) -> None:
    """Attach completed Layer 3 results and refresh station summaries."""
    for station_id, layer3_result in layer3_run.station_results.items():
        result = station_results.get(station_id)
        if result is None:
            continue
        contract = _layer3_contract(layer3_result, layer3_settings)
        result.layer3 = contract
        if update_station_metadata and layer3_result.metadata:
            result.station.metadata.update(layer3_result.metadata)
        model_name = (
            result.comparison.get("candidate_name", "model")
            if result.comparison is not None else "model"
        )
        result.summary = build_station_summary(
            result.station,
            layer1=result.layer1,
            layer2=result.layer2,
            comparison=result.comparison,
            layer3=contract,
            model_name=model_name,
        )


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
    original = {
        str(station_id): series.copy(deep=True)
        for station_id, series in series_by_station.items()
    }
    stations = [
        station_from_series(series, station_id=station_id, unit=unit)
        for station_id, series in original.items()
    ]
    batch = run_trihydra_batch(
        stations,
        config=config,
        context_path=context_path,
        target_station_ids=target_station_ids,
        continue_on_station_error=False,
    )
    if batch.network is None:
        raise RuntimeError("Network assessment produced no station results.")
    for station_id, before in original.items():
        pd.testing.assert_series_equal(series_by_station[station_id], before)
    return batch.network


def attach_layer3_to_results(
    station_results: Mapping[str, TriHydrAResult],
    *,
    context_path: str | Path,
    target_station_ids: Sequence[str] | None = None,
    config: Mapping[str, Any] | None = None,
):
    """Attach Layer 3 to completed station results without rerunning Layers 1/2."""
    if len(station_results) < 2:
        raise ValueError("Layer 3 requires at least two completed station results.")
    effective = merge_config(DEFAULT_CONFIG, config)
    ordered = {str(station_id): result for station_id, result in station_results.items()}
    series_by_station = {
        station_id: result.station.obs for station_id, result in ordered.items()
    }
    evidence = _context_evidence_for_results(ordered, effective["layer1"])
    layer3_run = run_layer3_context(
        series_by_station,
        context_path,
        target_station_ids=target_station_ids,
        layer3_config=effective["layer3"],
        layer1_config=effective["layer1"],
        layer2_config=effective["layer2"],
        evidence_cache=evidence,
    )
    _attach_layer3_contracts(
        ordered,
        layer3_run,
        effective["layer3"],
        update_station_metadata=True,
    )
    return layer3_run


def run_trihydra_batch(
    stations: Sequence[StationData] | Mapping[str, StationData],
    *,
    config: Mapping[str, Any] | None = None,
    context_path: str | Path | None = None,
    target_station_ids: Sequence[str] | None = None,
    continue_on_station_error: bool = True,
    progress: BatchProgress | None = None,
) -> TriHydrABatchResult:
    """Assess already-loaded stations without reading or writing files.

    Providing ``context_path`` requests Layer 3 after every possible station
    has completed Layers 1-2. Failures are recorded in ``manifest`` and either
    isolated or raised according to ``continue_on_station_error``.
    """
    if isinstance(stations, Mapping):
        ordered: list[StationData] = []
        for station_id, station in stations.items():
            if str(station_id) != station.station_id:
                raise ValueError(
                    f"Station mapping key {station_id!r} does not match "
                    f"StationData.station_id {station.station_id!r}."
                )
            ordered.append(station)
    else:
        ordered = list(stations)
    if not ordered:
        raise ValueError("stations cannot be empty.")
    if any(not isinstance(station, StationData) for station in ordered):
        raise TypeError("Every batch item must be a StationData object.")
    station_ids = [station.station_id for station in ordered]
    if len(set(station_ids)) != len(station_ids):
        raise ValueError("StationData.station_id values must be unique in a batch.")

    effective = merge_config(DEFAULT_CONFIG, config)
    completed: dict[str, TriHydrAResult] = {}
    rows: list[dict[str, object]] = []
    for number, station in enumerate(ordered, start=1):
        started = time.perf_counter()
        series2_status = station.metadata.get(
            "series2_status",
            "available" if station.series2 is not None else "disabled",
        )
        if progress is not None:
            progress("started", {
                "station_id": station.station_id,
                "station_number": number,
                "station_count": len(ordered),
                "series2_status": series2_status,
            })
        if len(station.series1) == 0 or not station.series1.notna().any():
            row = {
                "station_id": station.station_id,
                "status": "skipped",
                "series2_status": "not_attempted",
                "review_required": False,
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": "NoValidSeries1Values",
                "error_message": "No valid dated series1 values are available.",
            }
            rows.append(row)
            if progress is not None:
                progress("skipped", row)
            continue
        try:
            result = run_trihydra(
                station,
                config=effective,
                model_name=station.series2_name,
            )
            completed[station.station_id] = result
            row = {
                "station_id": station.station_id,
                "status": "completed",
                "series2_status": series2_status,
                "review_required": station_requires_review(result),
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": None,
                "error_message": None,
            }
            rows.append(row)
            if progress is not None:
                progress("completed", row)
        except Exception as error:
            row = {
                "station_id": station.station_id,
                "status": "failed",
                "series2_status": series2_status,
                "review_required": False,
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            rows.append(row)
            if progress is not None:
                progress("failed", row)
            if not continue_on_station_error:
                raise

    layer3_run = None
    if context_path is not None and len(completed) >= 2:
        layer3_run = attach_layer3_to_results(
            completed,
            context_path=context_path,
            target_station_ids=target_station_ids,
            config=effective,
        )

    layer3_status: dict[str, str] = {}
    if layer3_run is not None and hasattr(layer3_run, "station_results"):
        layer3_status = {
            station_id: (
                "assessed"
                if station_result.summary.combined_classification != "Not assessed"
                else "not_assessed"
            )
            for station_id, station_result in layer3_run.station_results.items()
        }
    for row in rows:
        row["layer3_status"] = layer3_status.get(
            str(row["station_id"]), "not_assessed"
        )

    network = None
    if completed:
        network = TriHydrANetworkResult(
            station_results=completed,
            layer3_run=layer3_run,
            summary=pd.concat(
                [result.summary for result in completed.values()],
                ignore_index=True,
                sort=False,
            ),
            series_by_station={
                station_id: result.station.series1
                for station_id, result in completed.items()
            },
            configuration_used=effective,
        )
    return TriHydrABatchResult(
        manifest=pd.DataFrame(rows),
        network=network,
        output_directory=None,
    )


__all__ = [
    "attach_layer3_to_results",
    "run_trihydra_batch",
    "run_trihydra_network",
]
