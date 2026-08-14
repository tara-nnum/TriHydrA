"""Build the compact, user-facing station summary."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from trihydra.io.models import StationData


def _identity(station: StationData, *, series_name: str, role: str) -> dict[str, Any]:
    metadata = station.metadata or {}
    return {
        "station_id": station.station_id,
        "series_name": series_name,
        "series_role": role,
        "unit": station.unit,
        "latitude": metadata.get("latitude", metadata.get("lat")),
        "longitude": metadata.get("longitude", metadata.get("lon")),
        "river_name": metadata.get("river_name", metadata.get("river")),
        "catchment_name": metadata.get("catchment_name", metadata.get("catchment")),
        "catchment_area_km2": metadata.get("catchment_area_km2", metadata.get("area_km2")),
        "input_dataset": metadata.get("input_dataset", metadata.get("source")),
    }


def _add_contract(row: dict[str, Any], contract: Mapping[str, Any] | None, prefix: str) -> None:
    if not contract:
        return
    for name, value in contract.get("summary_metrics", {}).items():
        row[name if str(name).startswith(prefix) else f"{prefix}{name}"] = value
    for name, value in contract.get("thresholds_used", {}).items():
        threshold_name = str(name)
        if not threshold_name.startswith(prefix):
            threshold_name = f"{prefix}{threshold_name}"
        row[f"threshold_{threshold_name}"] = value


def build_station_summary(
    station: StationData,
    *,
    layer1: Mapping[str, Any] | None,
    layer2: Mapping[str, Any] | None,
    comparison: Mapping[str, Any] | None = None,
    layer3: Mapping[str, Any] | None = None,
    model_name: str = "model",
) -> pd.DataFrame:
    """Return one readable row per assessed observation/model series."""
    observation = _identity(station, series_name="observation", role="observation")
    observation["valid_observation_count"] = int(station.obs.notna().sum())
    observation["layer1_completed"] = layer1 is not None
    observation["layer2_completed"] = layer2 is not None
    observation["comparison_completed"] = comparison is not None
    observation["layer3_status"] = "assessed" if layer3 else "not_assessed"
    _add_contract(observation, layer1, "layer1_")
    _add_contract(observation, layer2, "layer2_")
    _add_contract(observation, layer3, "layer3_")
    rows = [observation]
    if comparison:
        candidate = _identity(station, series_name=model_name, role="simulation")
        candidate["valid_observation_count"] = int(station.ml.notna().sum()) if station.ml is not None else 0
        candidate["layer1_completed"] = True
        candidate["layer2_completed"] = True
        candidate["comparison_completed"] = True
        candidate["layer3_status"] = "not_assessed"
        _add_contract(candidate, comparison.get("candidate_layer1"), "layer1_")
        _add_contract(candidate, comparison.get("candidate_native_layer2"), "layer2_")
        _add_contract(candidate, comparison, "comparison_")
        rows.append(candidate)
    return pd.DataFrame(rows)


__all__ = ["build_station_summary"]
