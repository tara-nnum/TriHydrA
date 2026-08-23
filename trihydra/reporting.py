"""Build the compact, user-facing station summary."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from trihydra.io.models import StationData
from trihydra.result import TriHydrAResult


def station_requires_review(result: TriHydrAResult) -> bool:
    """Return whether any completed assessment class requests review."""
    class_columns = [
        name for name in result.summary.columns if name.endswith("_class")
    ]
    classes = {
        str(value).strip().casefold()
        for name in class_columns
        for value in result.summary[name].dropna().tolist()
    }
    return bool(classes.intersection({"needs review", "review"}))


def _identity(
    station: StationData,
    *,
    series_name: str,
    role: str,
    metadata_key: str,
) -> dict[str, Any]:
    metadata = station.metadata or {}
    series_metadata = metadata.get(metadata_key, {}) or {}
    timespan = series_metadata.get("timespan", {}) or {}
    row = {
        "station_id": station.station_id,
        "series_name": series_name,
        "series_role": role,
        "unit": station.unit,
        "latitude": metadata.get("latitude", metadata.get("lat")),
        "longitude": metadata.get("longitude", metadata.get("lon")),
        "river_name": metadata.get("river_name", metadata.get("river")),
        "catchment_name": metadata.get("catchment_name", metadata.get("catchment")),
        "catchment_area_km2": metadata.get("catchment_area_km2", metadata.get("area_km2")),
        "climate_code": metadata.get("climate_code"),
        "climate_description": metadata.get("climate_description"),
        "input_dataset": metadata.get("input_dataset", metadata.get("source")),
    }
    row.update({
        "requested_timespan_mode": timespan.get("requested_mode"),
        "requested_start": timespan.get("requested_start"),
        "requested_end": timespan.get("requested_end"),
        "source_calendar_start": timespan.get("source_calendar_start"),
        "source_calendar_end": timespan.get("source_calendar_end"),
        "selected_calendar_start": timespan.get("selected_calendar_start"),
        "selected_calendar_end": timespan.get("selected_calendar_end"),
        "first_valid_date": timespan.get("first_valid_date"),
        "last_valid_date": timespan.get("last_valid_date"),
        "selected_row_count": timespan.get("selected_row_count"),
    })
    return row


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
    """Return one readable row per assessed primary/candidate series."""
    primary = _identity(
        station, series_name=station.series1_name, role=station.series1_role,
        metadata_key="series1_metadata",
    )
    primary["valid_observation_count"] = int(station.series1.notna().sum())
    primary["layer1_completed"] = layer1 is not None
    primary["layer2_completed"] = layer2 is not None
    primary["comparison_completed"] = comparison is not None
    layer3_class = (
        None if not layer3
        else layer3.get("summary_metrics", {}).get("context_agreement_class")
    )
    primary["layer3_status"] = (
        "assessed"
        if layer3_class not in {None, "Not assessed"}
        else "not_assessed"
    )
    _add_contract(primary, layer1, "layer1_")
    _add_contract(primary, layer2, "layer2_")
    _add_contract(primary, layer3, "layer3_")
    rows = [primary]
    if comparison:
        candidate = _identity(
            station,
            series_name=station.series2_name or model_name,
            role=station.series2_role,
            metadata_key="series2_metadata",
        )
        candidate["valid_observation_count"] = int(station.series2.notna().sum()) if station.series2 is not None else 0
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
