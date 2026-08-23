"""Synthetic tests for the canonical input data structures."""

import pandas as pd
import xarray as xr

from trihydra.io.availability import series_availability
from trihydra.io.series_sources import open_series_source
from trihydra.io.models import SourceProvenance, StationData
from trihydra.settings.models import Series1Config


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

    config = Series1Config(
        format="netcdf",
        path=path,
        units="mm/day",
        variable="streamflow",
        station_coordinate="station",
        time_coordinate="date",
    )
    with open_series_source(config) as (station_ids, load):
        stations = [load(station_id) for station_id in station_ids]

    assert [station.station_id for station in stations] == ["valid", "empty"]
    assert stations[1].values.isna().all()


def test_station_data_exposes_neutral_series_names(clean_daily_flow):
    """Series roles do not assume that the primary input is an observation."""
    candidate = clean_daily_flow * 1.1
    station = StationData.from_series(
        station_id="A",
        series1=clean_daily_flow,
        unit="mm/day",
        series1_provenance=SourceProvenance.in_memory(
            unit="mm/day", label="model A"
        ),
        series2=candidate,
        series2_provenance=SourceProvenance.in_memory(
            unit="mm/day", label="model B"
        ),
        series1_name="Model A",
        series1_role="simulation",
        series2_name="Model B",
        series2_role="simulation",
    )

    assert station.series1 is station.obs
    assert station.series2 is station.ml
    assert station.series1_role == "simulation"
    assert station.series2_role == "simulation"
    station.validate_raw_preservation()
