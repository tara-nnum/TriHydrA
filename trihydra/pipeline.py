"""Format-independent, in-memory TriHydrA pipeline for one station."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from trihydra.comparison import (
    ComparisonSeries,
    prepare_independent_comparison,
    prepare_paired_comparison,
    run_generic_comparison,
)
from trihydra.composite import score_layer1
from trihydra.settings.defaults import DEFAULT_CONFIG, merge_config
from trihydra.io.models import SourceProvenance, StationData
from trihydra.layer1.diagnostics import finalise_layer1_contract, run_layer1_diagnostics
from trihydra.layer2.diagnostics import run_layer2_diagnostics
from trihydra.reporting import build_station_summary
from trihydra.result import TriHydrAResult


def station_from_series(
    series: pd.Series,
    *,
    station_id: str,
    unit: str = "source units",
    model_series: pd.Series | None = None,
    model_name: str = "model",
    metadata: Mapping[str, Any] | None = None,
) -> StationData:
    """Wrap already-prepared pandas data in TriHydrA's input contract."""
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    if not str(station_id).strip():
        raise ValueError("station_id cannot be empty.")
    station = StationData(
        station_id=str(station_id),
        obs=series.copy(deep=True),
        unit=str(unit),
        obs_provenance=SourceProvenance.in_memory(unit=str(unit)),
        ml=None if model_series is None else model_series.copy(deep=True),
        ml_provenance=(
            None if model_series is None
            else SourceProvenance.in_memory(unit=str(unit), label=model_name)
        ),
        metadata=dict(metadata or {}),
    )
    station.validate_raw_preservation()
    return station


def run_station(
    station: StationData,
    *,
    config: Mapping[str, Any] | None = None,
    model_name: str = "model",
) -> TriHydrAResult:
    """Run one station entirely in memory and return reusable results."""
    station.validate_raw_preservation()
    observation_before = station.obs.copy(deep=True)
    model_before = None if station.ml is None else station.ml.copy(deep=True)
    effective = merge_config(DEFAULT_CONFIG, config)
    layers = effective["run"]["layers"]

    layer1 = None
    layer2 = None
    layer1_composite = None
    comparison = None
    if station.ml is not None and bool(layers["run_comparison"]):
        reference = ComparisonSeries(
            name=station.series1_name,
            values=station.series1,
            station_id=station.station_id,
            unit=station.unit,
            role=station.series1_role,
            provenance=station.series1_provenance,
            metadata=dict(station.metadata.get("series1_metadata", {})),
        )
        candidate = ComparisonSeries(
            name=station.series2_name or model_name,
            values=station.series2,
            station_id=station.station_id,
            unit=station.unit,
            role=station.series2_role,
            provenance=station.series2_provenance,
            metadata=dict(station.metadata.get("series2_metadata", {})),
        )
        comparison_mode = effective["comparison"].get("mode", "paired_overlap")
        prepared = (
            prepare_independent_comparison(reference, candidate)
            if comparison_mode == "independent_timespans"
            else prepare_paired_comparison(reference, candidate)
        )
        comparison = run_generic_comparison(
            prepared,
            layer1_config=effective["layer1"],
            layer2_config=effective["layer2"],
            comparison_config=effective["comparison"],
        )
        comparison["reference_layer1"] = finalise_layer1_contract(
            station.obs, comparison["reference_layer1"],
            comparison["reference_layer1_composite"], effective["layer1"],
        )
        comparison["candidate_layer1"] = finalise_layer1_contract(
            station.ml, comparison["candidate_layer1"],
            comparison["candidate_layer1_composite"], effective["layer1"],
        )
        layer1 = comparison["reference_layer1"]
        layer2 = comparison["reference_native_layer2"]
        layer1_composite = comparison["reference_layer1_composite"]
    else:
        if bool(layers["run_layer1"]):
            layer1 = run_layer1_diagnostics(station.obs, config=effective["layer1"])
        if bool(layers["run_layer2"]):
            layer2 = run_layer2_diagnostics(
                station.obs, layer1_result=layer1,
                config=effective["layer2"], discharge_unit=station.unit,
            )
        if layer1 is not None:
            layer1_composite = score_layer1(
                station.obs, layer1, layer2, config=effective["layer1"]
            )
            layer1 = finalise_layer1_contract(
                station.obs, layer1, layer1_composite, effective["layer1"]
            )

    pd.testing.assert_series_equal(station.obs, observation_before)
    if station.ml is not None and model_before is not None:
        pd.testing.assert_series_equal(station.ml, model_before)
    return TriHydrAResult(
        station=station,
        layer1=layer1,
        layer1_composite=layer1_composite,
        layer2=layer2,
        comparison=comparison,
        layer3=None,
        summary=build_station_summary(
            station, layer1=layer1, layer2=layer2, comparison=comparison,
            model_name=model_name,
        ),
        configuration_used=effective,
    )


def run_trihydra(
    data: StationData | pd.Series,
    *,
    station_id: str | None = None,
    unit: str = "source units",
    config: Mapping[str, Any] | None = None,
    model_series: pd.Series | None = None,
    model_name: str = "model",
    metadata: Mapping[str, Any] | None = None,
) -> TriHydrAResult:
    """Run TriHydrA from a StationData object or one pandas Series."""
    if isinstance(data, StationData):
        if station_id is not None or model_series is not None or metadata is not None:
            raise ValueError(
                "station_id, model_series, and metadata belong in StationData "
                "when data is already a StationData object."
            )
        station = data
    elif isinstance(data, pd.Series):
        if station_id is None:
            raise ValueError("station_id is required when data is a pandas Series.")
        station = station_from_series(
            data, station_id=station_id, unit=unit,
            model_series=model_series, model_name=model_name, metadata=metadata,
        )
    else:
        raise TypeError("data must be StationData or pandas.Series.")
    return run_station(station, config=config, model_name=model_name)


__all__ = ["run_station", "run_trihydra", "station_from_series"]
