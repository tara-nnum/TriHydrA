"""Readable checks for the consolidated NetCDF summary writer."""

from pathlib import Path

import pandas as pd
import xarray as xr

from trihydra.io.models import SourceProvenance, StationData
from trihydra.outputs.netcdf import build_netcdf_dataset, write_netcdf_results
from trihydra.result import TriHydrAResult


def _example_result() -> TriHydrAResult:
    dates = pd.date_range("2000-01-01", periods=5, freq="D")
    station = StationData(
        station_id="EXAMPLE_001",
        obs=pd.Series([1.0, 2.0, 3.0, 2.0, 1.0], index=dates),
        unit="mm/day",
        obs_provenance=SourceProvenance(
            path=Path("example.nc"), format="NetCDF", variable="streamflow",
            station_coordinate="station", time_coordinate="date", unit="mm/day",
        ),
    )
    components = pd.DataFrame([
        {
            "check": "missing_values", "assessable": True, "raw_value": 0.0,
            "tier": "Tier 3", "tier_points": 0, "weight": 2,
            "contribution": 0, "reason": "Internal missingness = 0.000%.",
        },
        {
            "check": "long_gaps", "assessable": True, "raw_value": 18,
            "tier": "Tier 2", "tier_points": 1, "weight": 2,
            "contribution": 2, "reason": "Longest gap = 18 days.",
        },
    ])
    composite = {
        "components": components,
        "summary": pd.DataFrame([{"layer1_score": 2, "layer1_class": "No review needed"}]),
    }
    summary = pd.DataFrame([{
        "station_id": station.station_id,
        "series_name": "observation",
        "series_role": "observation",
        "unit": station.unit,
        "layer1_score": 2,
        "layer1_class": "No review needed",
        "layer1_missing_percentage": 0.0,
        "threshold_layer1_missing_tier2_threshold_percent": 5.0,
    }])
    return TriHydrAResult(
        station=station, layer1_composite=composite, summary=summary,
        configuration_used={"run": {"layers": {"run_layer1": True}}},
    )


def test_dataset_contains_summary_provenance_and_only_flagged_checks():
    result = _example_result()
    manifest = pd.DataFrame([{
        "station_id": result.station_id, "status": "completed",
        "review_required": False, "elapsed_seconds": 0.1,
    }])

    dataset = build_netcdf_dataset([result], manifest, configuration={"example": True})

    assert dataset.sizes == {"station": 1, "series": 1, "flag_record": 1}
    assert dataset["units"].item() == "mm/day"
    assert dataset["input_dataset"].item() == "example.nc"
    assert dataset["diagnostic_status"].item() == "Concerns found"
    assert dataset["flagged_diagnostic_count"].item() == 1
    assert dataset["flagged_diagnostic_name"].item() == "long_gaps"
    assert "missing_values" not in dataset["flagged_diagnostic_name"].values
    assert dataset.attrs["completed_station_count"] == 1
    assert "threshold_definition" in dataset[
        "threshold_layer1_missing_tier2_threshold_percent"
    ].attrs
    assert dataset[
        "threshold_layer1_missing_tier2_threshold_percent"
    ].attrs["units"] == "%"


def test_context_agreement_percent_is_not_labelled_as_points():
    result = _example_result()
    result.summary["layer3_context_agreement_score_percent"] = 82.5
    manifest = pd.DataFrame([{
        "station_id": result.station_id, "status": "completed",
        "review_required": False, "elapsed_seconds": 0.1,
    }])

    dataset = build_netcdf_dataset([result], manifest)

    assert dataset["layer3_context_agreement_score_percent"].attrs["units"] == "%"
    assert "Weighted contextual agreement" in dataset[
        "layer3_context_agreement_score_percent"
    ].attrs["description"]


def test_percentile_fraction_is_distinct_from_percentage_units():
    result = _example_result()
    result.summary["threshold_layer2_high_flow_trigger_percentile"] = 0.95
    manifest = pd.DataFrame([{
        "station_id": result.station_id, "status": "completed",
        "review_required": False, "elapsed_seconds": 0.1,
    }])

    dataset = build_netcdf_dataset([result], manifest)

    assert dataset[
        "threshold_layer2_high_flow_trigger_percentile"
    ].attrs["units"] == "1"
    assert dataset["elapsed_seconds"].attrs["units"] == "s"


def test_main_user_facing_variables_have_explanatory_metadata():
    result = _example_result()
    manifest = pd.DataFrame([{
        "station_id": result.station_id, "status": "completed",
        "review_required": False, "elapsed_seconds": 0.1,
    }])

    dataset = build_netcdf_dataset([result], manifest)

    assert "Weighted sum" in dataset["layer1_score"].attrs["description"]
    assert "does not prove" in dataset["layer1_class"].attrs["interpretation"]
    assert "positive composite contribution" in dataset[
        "flagged_diagnostic_count"
    ].attrs["description"]


def test_written_file_reopens_as_one_root_dataset(tmp_path):
    result = _example_result()
    manifest = pd.DataFrame([{
        "station_id": result.station_id, "status": "completed",
        "review_required": False, "elapsed_seconds": 0.1,
    }])
    path = write_netcdf_results([result], manifest, tmp_path / "trihydra_results.nc")

    with xr.open_dataset(path) as reopened:
        assert reopened.station.item() == "EXAMPLE_001"
        assert reopened["layer1_score"].item() == 2.0
        assert reopened["flagged_diagnostic_summary"].item() == "Longest gap = 18 days."
