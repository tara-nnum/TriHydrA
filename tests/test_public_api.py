"""Regression checks for the small Python/Jupyter-facing API."""

import pandas as pd
import pytest

from trihydra import (
    SourceProvenance,
    StationData,
    TriHydrABatchResult,
    TriHydrAResult,
    load_stations,
    run_batch,
    run_trihydra,
    run_trihydra_batch,
)
from trihydra.composite import score_layer1, score_layer2_comparison
from trihydra.layer1.diagnostics import run_layer1_diagnostics
from trihydra.layer2.diagnostics import run_layer2_diagnostics


def test_public_loader_uses_toml_adapters_without_running_or_writing(tmp_path):
    """A notebook can load the same selected source data used by the CLI."""
    source = tmp_path / "flows.csv"
    pd.DataFrame({
        "date": pd.date_range("2000-01-01", periods=5, freq="D"),
        "gauge_a": [1.0, 2.0, None, 4.0, 5.0],
        "gauge_b": [5.0, 4.0, 3.0, 2.0, 1.0],
    }).to_csv(source, index=False)
    config = tmp_path / "loader.toml"
    config.write_text("\n".join([
        "[run]",
        'station_ids = ["gauge_a"]',
        "all_stations = false",
        "[series1]",
        'format = "csv"',
        'path = "flows.csv"',
        'name = "reference record"',
        'role = "historical_observation"',
        'units = "mm/day"',
        '[series1.timespan]',
        'mode = "range"',
        'start_date = 2000-01-02',
        'end_date = 2000-01-04',
    ]), encoding="utf-8")

    stations = load_stations(config)

    assert len(stations) == 1
    station = stations[0]
    assert station.station_id == "gauge_a"
    assert station.series1_name == "reference record"
    assert station.series1_role == "historical_observation"
    assert station.series1.index.tolist() == list(pd.date_range(
        "2000-01-02", "2000-01-04", freq="D"
    ))
    assert station.series1.isna().sum() == 1
    assert station.series1_provenance.format == "wide CSV"
    assert not (tmp_path / "outputs").exists()


def test_public_run_trihydra_accepts_a_series_without_changing_it(clean_daily_flow):
    """A notebook user can pass one pandas Series through the public import."""
    original = clean_daily_flow.copy(deep=True)

    result = run_trihydra(
        clean_daily_flow,
        station_id="EXAMPLE_STATION",
        unit="mm/day",
    )

    assert isinstance(result, TriHydrAResult)
    assert result.station.station_id == "EXAMPLE_STATION"
    assert not result.summary.empty
    pd.testing.assert_series_equal(clean_daily_flow, original)


def test_independent_timespans_compare_without_shared_dates(clean_daily_flow):
    candidate = clean_daily_flow.copy(deep=True)
    candidate.index = pd.date_range("2010-01-01", periods=len(candidate), freq="D")
    station = StationData.from_series(
        station_id="EXAMPLE_STATION",
        series1=clean_daily_flow,
        series2=candidate,
        unit="mm/day",
        series1_provenance=SourceProvenance.in_memory(
            unit="mm/day", label="historical"
        ),
        series2_provenance=SourceProvenance.in_memory(
            unit="mm/day", label="recent"
        ),
        series1_name="historical",
        series1_role="historical_observation",
        series2_name="recent",
        series2_role="observation",
    )

    result = run_trihydra(
        station,
        config={
            "run": {"layers": {"run_comparison": True}},
            "comparison": {
                "mode": "independent_timespans",
                "daily_metrics": {"calculate": False},
            },
        },
    )

    assert result.comparison["mode"] == "independent_timespans"
    assert result.comparison["coverage"]["pairwise_valid_count"] is None
    assert result.comparison["daily_metrics"].empty


def test_python_api_exports_the_same_batch_runner_used_by_the_cli():
    assert callable(run_batch)
    assert TriHydrABatchResult.__name__ == "TriHydrABatchResult"


def test_in_memory_batch_returns_results_and_isolates_station_errors(
    clean_daily_flow, tmp_path, monkeypatch
):
    """Already-loaded stations can be processed without reading or writing."""
    monkeypatch.chdir(tmp_path)
    good = StationData.from_series(
        station_id="GOOD",
        series1=clean_daily_flow,
        unit="mm/day",
        series1_provenance=SourceProvenance.in_memory(unit="mm/day"),
    )
    bad = StationData.from_series(
        station_id="BAD",
        series1=pd.Series([1.0, 2.0], index=[0, 1]),
        unit="mm/day",
        series1_provenance=SourceProvenance.in_memory(unit="mm/day"),
    )

    batch = run_trihydra_batch([good, bad], continue_on_station_error=True)

    assert batch.output_directory is None
    assert list(batch.station_results) == ["GOOD"]
    assert batch.manifest.set_index("station_id").loc["GOOD", "status"] == "completed"
    assert batch.manifest.set_index("station_id").loc["BAD", "status"] == "failed"
    assert batch.manifest.set_index("station_id").loc["BAD", "error_type"] == "TypeError"
    assert not list(tmp_path.iterdir())


def test_in_memory_batch_can_stop_on_the_first_station_error(clean_daily_flow):
    bad = StationData.from_series(
        station_id="BAD",
        series1=pd.Series([1.0, 2.0], index=[0, 1]),
        unit="mm/day",
        series1_provenance=SourceProvenance.in_memory(unit="mm/day"),
    )

    with pytest.raises(TypeError, match="DatetimeIndex"):
        run_trihydra_batch([bad], continue_on_station_error=False)


def test_in_memory_batch_attaches_layer3_after_station_processing(
    clean_daily_flow, monkeypatch
):
    stations = [
        StationData.from_series(
            station_id=station_id,
            series1=clean_daily_flow * multiplier,
            unit="mm/day",
            series1_provenance=SourceProvenance.in_memory(unit="mm/day"),
        )
        for station_id, multiplier in (("A", 1.0), ("B", 1.1))
    ]
    marker = object()

    def fake_attach(results, **kwargs):
        assert list(results) == ["A", "B"]
        assert kwargs["context_path"] == "context.csv"
        return marker

    monkeypatch.setattr("trihydra.network.attach_layer3_to_results", fake_attach)

    batch = run_trihydra_batch(stations, context_path="context.csv")

    assert batch.network.layer3_run is marker


def test_one_enabled_layer1_check_uses_its_own_possible_score(clean_daily_flow):
    """A focused run is classified against only the check that was run."""
    config = {
        name: {"enabled": name == "duplicate_timestamps"}
        for name in (
            "missing_values", "long_gaps", "negative_discharge",
            "duplicate_timestamps", "timestep_consistency",
            "zero_flow_regime", "low_variability", "spike_dip",
            "step_shift", "epoch_drift",
        )
    }
    duplicated = pd.concat([clean_daily_flow, clean_daily_flow.iloc[[0]]])
    diagnostics = run_layer1_diagnostics(duplicated, config=config)
    composite = score_layer1(duplicated, diagnostics, config=config)
    summary = composite["summary"].iloc[0]

    assert composite["components"]["check"].tolist() == [
        "duplicate_timestamps"
    ]
    assert summary["layer1_score_percent"] == 100.0
    assert summary["layer1_class"] == "Needs review"
    assert summary["assessment_scope"] == "Focused"
    assert summary["enabled_check_count"] == 1
    assert summary["total_composite_check_count"] == 9
    assert summary["evidence_coverage_percent"] == 100.0
    assert "within the selected checks" in summary["scope_conclusion"]


def test_one_enabled_layer2_component_can_form_a_comparison(clean_daily_flow):
    """A deliberate one-component comparison is not rejected as incomplete."""
    reference = run_layer2_diagnostics(clean_daily_flow)
    candidate_series = clean_daily_flow * 1.05
    candidate = run_layer2_diagnostics(candidate_series)
    enabled = {
        name: name == "flow_behaviour"
        for name in (
            "flow_behaviour", "annual_flashiness_shape",
            "annual_baseflow_shape", "seasonal_profile_shape",
            "seasonal_timing", "event_time_to_peak", "event_duration",
            "representative_event_shape",
        )
    }
    composite = score_layer2_comparison(
        clean_daily_flow,
        candidate_series,
        reference,
        candidate,
        config={"comparison": {"components": enabled}},
    )
    summary = composite["summary"].iloc[0]

    assert composite["components"]["component"].tolist() == [
        "flow_behaviour"
    ]
    assert summary["assessable_components"] == 1
    assert summary["layer2_class"] != "Not assessable"
