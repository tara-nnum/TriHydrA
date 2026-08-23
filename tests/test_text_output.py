"""Checks for readable station-network text output."""

import pandas as pd

from trihydra.io.models import SourceProvenance, StationData
from trihydra.outputs import save_results
from trihydra.outputs.network_diagnostics import diagnostic_trigger_summary
from trihydra.outputs.reports import render_network_summary, render_station_summary
from trihydra.result import TriHydrANetworkResult, TriHydrAResult


def _result(station_id: str) -> TriHydrAResult:
    station = StationData(
        station_id=station_id,
        obs=pd.Series([1.0], index=pd.to_datetime(["2000-01-01"])),
        unit="mm/day",
        obs_provenance=SourceProvenance.in_memory(unit="mm/day"),
    )
    summary = pd.DataFrame([{
        "station_id": station_id,
        "series_name": "observation",
        "series_role": "observation",
        "unit": "mm/day",
        "layer1_score": 0,
        "layer1_class": "No review needed",
        "layer1_assessment_scope": "Focused",
        "layer1_scope_conclusion": "Focused Layer 1 assessment: No concerns detected within the selected checks.",
        "layer1_enabled_check_count": 1,
        "layer1_assessable_check_count": 1,
        "layer1_total_composite_check_count": 9,
        "layer1_evidence_coverage_percent": 100.0,
    }])
    return TriHydrAResult(station=station, summary=summary)


def test_network_save_writes_one_index_and_each_station_summary(tmp_path):
    results = {station_id: _result(station_id) for station_id in ("A", "B")}
    network = TriHydrANetworkResult(
        station_results=results,
        layer3_run=None,
        summary=pd.concat([result.summary for result in results.values()], ignore_index=True),
        series_by_station={key: value.station.obs for key, value in results.items()},
    )

    written = save_results(network, tmp_path)

    assert (tmp_path / "network_summary.txt").is_file()
    assert (tmp_path / "A" / "summary.txt").is_file()
    assert (tmp_path / "B" / "summary.txt").is_file()
    assert "A" in (tmp_path / "network_summary.txt").read_text(encoding="utf-8")
    assert "B" in (tmp_path / "network_summary.txt").read_text(encoding="utf-8")
    assert "network_summary" in written


def test_station_text_distinguishes_requested_selected_and_valid_dates():
    result = _result("A")
    result.summary["requested_timespan_mode"] = "range"
    result.summary["requested_start"] = "1999-12-01"
    result.summary["requested_end"] = "2000-12-31"
    result.summary["selected_calendar_start"] = "2000-01-01"
    result.summary["selected_calendar_end"] = "2000-01-01"

    report = render_station_summary(result)

    assert "Requested timespan" in report
    assert "1 Dec 1999 to 31 Dec 2000" in report
    assert "Selected calendar" in report
    assert "First to last valid" in report


def test_station_text_explains_focused_layer1_scope():
    report = render_station_summary(_result("A"))

    assert "Assessment scope" in report
    assert "Focused" in report
    assert "Enabled composite checks" in report
    assert "1/9" in report
    assert "within the selected checks" in report
    assert "No review needed (selected checks only)" in report


def test_network_summary_counts_stations_series_and_triggering_checks():
    result = _result("A")
    result.layer1_composite = {
        "components": pd.DataFrame([
            {
                "check": "missing_values", "assessable": True,
                "tier": "Tier 2", "contribution": 2,
            },
            {
                "check": "step_shift", "assessable": False,
                "tier": "Not assessable", "contribution": 0,
            },
        ])
    }
    result.summary["layer1_class"] = "Minor concerns"
    triggers = diagnostic_trigger_summary([result])

    report = render_network_summary(result.summary, triggers)

    assert "Unique stations" in report
    assert "Station-series assessments" in report
    assert "Minor concerns only" in report
    assert "Layer 1 checks causing concerns" in report
    assert "Assessable" in report
    assert "Triggered" in report
    assert "MISSING VALUES" in report
    assert "100.0%" in report
