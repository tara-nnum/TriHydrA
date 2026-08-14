"""Checks for attaching Layer 3 to completed station results."""

from types import SimpleNamespace

import pandas as pd

from trihydra.io.models import SourceProvenance, StationData
from trihydra.network import attach_layer3_to_results
from trihydra.result import TriHydrAResult


def _completed_result(station_id: str) -> TriHydrAResult:
    dates = pd.date_range("2000-01-01", periods=5, freq="D")
    station = StationData(
        station_id=station_id,
        obs=pd.Series([1.0, 2.0, 3.0, 2.0, 1.0], index=dates),
        unit="mm/day",
        obs_provenance=SourceProvenance.in_memory(unit="mm/day"),
    )
    return TriHydrAResult(
        station=station,
        summary=pd.DataFrame([{
            "station_id": station_id, "series_name": "observation",
            "series_role": "observation", "unit": "mm/day",
        }]),
    )


def test_layer3_attaches_to_existing_results_without_rerunning_layers(monkeypatch):
    results = {station_id: _completed_result(station_id) for station_id in ("A", "B")}
    fake_station_results = {
        station_id: SimpleNamespace(
            metadata={
                "station_id": station_id, "latitude": 1.0, "longitude": 2.0,
                "river_name": "River", "catchment_name": "Catchment",
                "catchment_area_km2": 100.0,
            },
            summary=SimpleNamespace(combined_classification="Strong agreement"),
        )
        for station_id in results
    }
    captured = {}

    def fake_layer3(series_by_station, context_path, **kwargs):
        captured["series_ids"] = list(series_by_station)
        captured["evidence_ids"] = list(kwargs["evidence_cache"])
        return SimpleNamespace(station_results=fake_station_results)

    monkeypatch.setattr("trihydra.network.run_layer3_context", fake_layer3)
    monkeypatch.setattr(
        "trihydra.network._layer3_contract",
        lambda _result, _settings: {
            "summary_metrics": {
                "context_agreement_class": "Strong agreement",
                "context_agreement_score_percent": 80.0,
            },
            "thresholds_used": {"similar_minimum_percent": 75.0},
            "evidence": pd.DataFrame(),
        },
    )

    attach_layer3_to_results(results, context_path="context.csv")

    assert captured == {"series_ids": ["A", "B"], "evidence_ids": ["A", "B"]}
    assert results["A"].layer3["summary_metrics"]["context_agreement_class"] == "Strong agreement"
    assert results["A"].summary.iloc[0]["layer3_status"] == "assessed"
    assert results["A"].summary.iloc[0]["latitude"] == 1.0

