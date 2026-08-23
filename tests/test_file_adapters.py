"""Small real-file checks for the supported ECMWF input adapters."""

import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from trihydra.io.pickle_results import load_model_result_pickle
from trihydra.io.series_sources import open_series_source
from trihydra.settings.models import Series1Config


def test_netcdf_adapter_detects_streamflow_time_and_station_coordinates(tmp_path):
    path = tmp_path / "observations.nc"
    dataset = xr.Dataset(
        {
            "streamflow": (
                ("basin", "date"),
                [[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]],
            )
        },
        coords={
            "basin": ["A", "B"],
            "date": pd.date_range("2000-01-01", periods=3, freq="D"),
        },
    )
    dataset.to_netcdf(path, engine="netcdf4")

    config = Series1Config(
        format="netcdf",
        path=path,
        units="mm/day",
        variable="streamflow",
        station_coordinate="basin",
        time_coordinate="date",
    )
    with open_series_source(config) as (station_ids, load):
        record = load("A")

    assert station_ids == ("A", "B")
    assert record.station_id == "A"
    assert record.values.tolist()[0] == 1.0
    assert pd.isna(record.values.iloc[1])
    assert record.provenance.variable == "streamflow"
    assert record.provenance.station_coordinate == "basin"
    assert record.provenance.time_coordinate == "date"


def test_aifl_pickle_adapter_selects_only_contemporaneous_time_step(tmp_path):
    path = tmp_path / "A_results.p"
    dates = pd.date_range("2000-01-01", periods=3, freq="D")
    dataset = xr.Dataset(
        {
            "streamflow_obs": (
                ("date", "time_step"),
                [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]],
            ),
            "streamflow_sim": (
                ("date", "time_step"),
                [[11.0, 1.1], [21.0, 2.1], [31.0, 3.1]],
            ),
        },
        coords={"date": dates, "time_step": [-1, 0]},
    )
    payload = {"A": {"1D": {"xr": dataset, "bias": 0.1}}}
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    station = load_model_result_pickle(path, station_id="A", unit="mm/day")

    assert station.obs.tolist() == [1.0, 2.0, 3.0]
    assert station.ml.tolist() == [1.1, 2.1, 3.1]
    assert station.metadata["stored_performance_metrics"] == {"bias": 0.1}
    assert station.ml_provenance.details["available_time_steps"] == (-1, 0)


def test_aifl_pickle_adapter_rejects_filename_station_mismatch(tmp_path):
    path = tmp_path / "wrong_results.p"
    with path.open("wb") as handle:
        pickle.dump({"A": {}}, handle)

    with pytest.raises(ValueError, match="filename identifies"):
        load_model_result_pickle(path, unit="mm/day")


def test_wide_csv_source_preserves_dates_missing_values_and_station_columns(tmp_path):
    path = tmp_path / "flows.csv"
    pd.DataFrame(
        {
            "date": ["2000-01-02", "2000-01-01", "2000-01-01"],
            "A": [2.0, np.nan, 1.0],
            "B": [4.0, 5.0, 6.0],
        }
    ).to_csv(path, index=False)
    config = Series1Config(
        format="csv",
        path=path,
        name="reference",
        role="observation",
        units="mm/day",
        date_column="date",
    )

    with open_series_source(config) as (station_ids, load):
        station = load("A")

    assert station_ids == ("A", "B")
    assert station.values.index.has_duplicates
    assert not station.values.index.is_monotonic_increasing
    assert station.values.isna().sum() == 1
    assert station.provenance.format == "wide CSV"


def test_configured_timespan_is_inclusive_and_does_not_fill_values(tmp_path):
    path = tmp_path / "flows.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=5, freq="D"),
            "A": [1.0, np.nan, 3.0, 4.0, 5.0],
        }
    ).to_csv(path, index=False)
    config = Series1Config.model_validate(
        {
            "format": "csv",
            "path": path,
            "units": "mm/day",
            "timespan": {
                "mode": "range",
                "start_date": "2000-01-02",
                "end_date": "2000-01-04",
            },
        }
    )

    with open_series_source(config) as (_station_ids, load):
        record = load("A")
        values = record.values

    assert values.index.tolist() == list(pd.date_range("2000-01-02", periods=3))
    assert values.isna().sum() == 1
    assert record.metadata["timespan"] == {
        "requested_mode": "range",
        "requested_start": "2000-01-02",
        "requested_end": "2000-01-04",
        "source_calendar_start": "2000-01-01",
        "source_calendar_end": "2000-01-05",
        "selected_calendar_start": "2000-01-02",
        "selected_calendar_end": "2000-01-04",
        "first_valid_date": "2000-01-03",
        "last_valid_date": "2000-01-04",
        "selected_row_count": 3,
        "selected_valid_count": 2,
    }


def test_aifl_source_role_selects_requested_stored_series(tmp_path):
    path = tmp_path / "A_results.p"
    dates = pd.date_range("2000-01-01", periods=2, freq="D")
    dataset = xr.Dataset(
        {
            "streamflow_obs": (("date", "time_step"), [[1.0], [2.0]]),
            "streamflow_sim": (("date", "time_step"), [[10.0], [20.0]]),
        },
        coords={"date": dates, "time_step": [0]},
    )
    with path.open("wb") as handle:
        pickle.dump({"A": {"1D": {"xr": dataset}}}, handle)
    config = Series1Config(
        format="aifl_pickle",
        path=path,
        role="simulation",
        units="mm/day",
        trusted=True,
    )

    with open_series_source(config) as (_station_ids, load):
        values = load("A").values

    assert values.tolist() == [10.0, 20.0]
