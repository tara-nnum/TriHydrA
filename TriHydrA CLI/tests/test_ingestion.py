"""Synthetic tests for the canonical input data structures."""

import pandas as pd
import xarray as xr

from trihydra.io.availability import series_availability
from trihydra.io.ingestion import iter_netcdf_stations
from trihydra.io.models import SourceProvenance, StationData


def test_station_data_keeps_identity_units_and_source_information():
    dates = pd.date_range("2000-01-01", periods=3, freq="D")
    observation = pd.Series([1.0, 2.0, 3.0], index=dates)
    source = SourceProvenance(
        path="example.nc",
        format="netcdf",
        variable="discharge",
        station_coordinate="station",
        time_coordinate="time",
        unit="mm/day",
    )

    station = StationData(
        station_id="station_1",
        obs=observation,
        unit="mm/day",
        obs_provenance=source,
    )

    assert station.station_id == "station_1"
    assert station.unit == "mm/day"
    assert str(station.obs_provenance.path) == "example.nc"


def test_availability_reports_missing_values_without_changing_the_series():
    dates = pd.date_range("2000-01-01", periods=3, freq="D")
    flow = pd.Series([1.0, None, 3.0], index=dates)
    original = flow.copy(deep=True)

    summary = series_availability(flow, "observation")

    pd.testing.assert_series_equal(flow, original)
    assert summary["observation_calendar_count"] == 3
    assert summary["observation_missing_count"] == 1


def test_raw_duplicate_and_unsorted_dates_reach_layer1_unchanged():
    dates = pd.DatetimeIndex(["2000-01-02", "2000-01-01", "2000-01-01"])
    observation = pd.Series([2.0, 1.0, 1.1], index=dates)
    source = SourceProvenance(
        path="example.nc",
        format="netcdf",
        variable="discharge",
        station_coordinate="station",
        time_coordinate="time",
        unit="mm/day",
    )
    station = StationData(
        station_id="station_1",
        obs=observation.copy(deep=True),
        unit="mm/day",
        obs_provenance=source,
    )

    station.validate_raw_preservation()

    pd.testing.assert_series_equal(station.obs, observation)
    assert station.obs.index.has_duplicates
    assert not station.obs.index.is_monotonic_increasing


def test_batch_style_ingestion_can_retain_an_empty_station_for_reporting(tmp_path):
    path = tmp_path / "observations.nc"
    dataset = xr.Dataset(
        {
            "streamflow": (
                ("station", "date"),
                [[1.0, 2.0], [float("nan"), float("nan")]],
            )
        },
        coords={
            "station": ["valid", "empty"],
            "date": pd.date_range("2000-01-01", periods=2, freq="D"),
        },
    )
    dataset.to_netcdf(path, engine="netcdf4")

    stations = list(iter_netcdf_stations(
        path,
        "all",
        variable="streamflow",
        station_coordinate="station",
        time_coordinate="date",
        include_empty=True,
    ))

    assert [station.station_id for station in stations] == ["valid", "empty"]
    assert stations[1].obs.isna().all()
