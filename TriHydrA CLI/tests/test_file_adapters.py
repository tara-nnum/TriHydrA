"""Small real-file checks for the supported ECMWF input adapters."""

import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from trihydra.io import NetCDFStationSource, load_model_result_pickle


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

    with NetCDFStationSource(path, unit="mm/day") as source:
        station = source.load_station("A")

    assert source.dataset is None
    assert station.station_id == "A"
    assert station.obs.tolist()[0] == 1.0
    assert pd.isna(station.obs.iloc[1])
    assert station.obs_provenance.variable == "streamflow"
    assert station.obs_provenance.station_coordinate == "basin"
    assert station.obs_provenance.time_coordinate == "date"


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
