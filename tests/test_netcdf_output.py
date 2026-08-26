"""Readable checks for the user-facing grouped NetCDF result files."""

from pathlib import Path

import netCDF4
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
        station=station,
        layer1_composite={
            "components": components,
            "summary": pd.DataFrame([{
                "layer1_score": 2, "layer1_class": "No review needed",
            }]),
        },
        summary=summary,
        configuration_used={"run": {"layers": {"run_layer1": True}}},
    )


def _manifest(result: TriHydrAResult) -> pd.DataFrame:
    return pd.DataFrame([{
        "station_id": result.station_id,
        "status": "completed",
        "review_required": False,
        "elapsed_seconds": 0.1,
    }])


def test_network_index_is_compact_and_points_to_station_file():
    result = _example_result()
    dataset = build_netcdf_dataset([result], _manifest(result))

    assert dataset.sizes == {"station": 1, "diagnostic": 2, "attention_record": 1}
    assert dataset.station.item() == "EXAMPLE_001"
    assert dataset["processing_status"].item() == "completed"
    assert dataset["primary_series_name"].item() == "observation"
    assert dataset["flagged_diagnostic_count"].item() == 1
    assert dataset["station_result_file"].item() == "stations/EXAMPLE_001.nc"
    assert "layer1_missing_percentage" not in dataset
    assert set(dataset.diagnostic.values.tolist()) == {"missing_values", "long_gaps"}
    assert dataset["diagnostic_concern_series_count"].sel(diagnostic="long_gaps").item() == 1
    assert dataset["diagnostic_trigger_rate_percent"].sel(diagnostic="long_gaps").item() == 100.0
    assert dataset["attention_station_id"].item() == "EXAMPLE_001"
    assert dataset["attention_series_name"].item() == "observation"
    assert dataset["attention_rank"].item() == 0
    assert dataset.attrs["station_series_assessment_count"] == 1
    assert dataset.attrs["layer1_no_review_series_count"] == 1


def test_writer_creates_one_grouped_file_per_station(tmp_path):
    result = _example_result()
    network_path = write_netcdf_results(
        [result], _manifest(result), tmp_path / "trihydra_network_summary.nc",
        configuration={"example": True},
    )
    station_path = tmp_path / "stations" / "EXAMPLE_001.nc"

    assert network_path.is_file()
    assert station_path.is_file()
    with netCDF4.Dataset(station_path) as file:
        assert set(file.groups) == {
            "metadata", "layer1", "layer2", "comparison", "layer3",
            "flags", "thresholds", "evidence", "configuration",
        }

    with xr.open_dataset(station_path) as root:
        assert "station" not in root.dims
        assert root.series_slot.item() == "series1"
        assert root["series_name"].item() == "observation"
        assert root["series_role"].item() == "observation"
        assert root.attrs["station_id"] == "EXAMPLE_001"
        assert "layer1_missing_percentage" not in root

    with xr.open_dataset(station_path, group="metadata") as metadata:
        assert metadata["units"].item() == "mm/day"
        assert metadata["input_dataset"].item() == "example.nc"

    with xr.open_dataset(station_path, group="layer1") as layer1:
        assert layer1["layer1_missing_percentage"].item() == 0.0
        assert "threshold_layer1_missing_tier2_threshold_percent" not in layer1

    with xr.open_dataset(station_path, group="thresholds") as thresholds:
        variable = thresholds["threshold_layer1_missing_tier2_threshold_percent"]
        assert variable.item() == 5.0
        assert variable.attrs["units"] == "%"
        assert "threshold_definition" in variable.attrs

    with xr.open_dataset(station_path, group="flags") as flags:
        assert flags.sizes == {"diagnostic_record": 1}
        assert flags["flagged_diagnostic_name"].item() == "long_gaps"
        assert flags["flagged_diagnostic_summary"].item() == "Longest gap = 18 days."

    with xr.open_dataset(station_path, group="comparison") as comparison:
        assert comparison.attrs["comparison_status"] == "not_assessed"


def test_requested_and_selected_timespans_live_in_metadata_group(tmp_path):
    result = _example_result()
    result.summary["requested_timespan_mode"] = "range"
    result.summary["requested_start"] = "2000-01-01"
    result.summary["requested_end"] = "2000-01-10"
    result.summary["selected_calendar_start"] = "2000-01-02"
    result.summary["selected_calendar_end"] = "2000-01-05"
    result.summary["first_valid_date"] = "2000-01-03"
    result.summary["last_valid_date"] = "2000-01-05"
    write_netcdf_results(
        [result], _manifest(result), tmp_path / "trihydra_network_summary.nc",
    )

    station_path = tmp_path / "stations" / "EXAMPLE_001.nc"
    with xr.open_dataset(station_path, group="metadata") as metadata:
        assert metadata["requested_start"].item() == "2000-01-01"
        assert metadata["selected_calendar_start"].item() == "2000-01-02"
        assert metadata["first_valid_date"].item() == "2000-01-03"
        assert "Inclusive start date" in metadata["requested_start"].attrs["description"]
