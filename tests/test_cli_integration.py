"""One small end-to-end test of the installed-style command-line workflow."""

import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr

from trihydra.batch import run_batch
from trihydra.result import TriHydrABatchResult


def test_cli_reads_toml_runs_station_and_reopens_netcdf(tmp_path):
    input_path = tmp_path / "observations.nc"
    output_path = tmp_path / "results"
    config_path = tmp_path / "trihydra.toml"
    dates = pd.date_range("2000-01-01", periods=730, freq="D")
    flow = 2.0 + np.sin(np.arange(len(dates)) * 2 * np.pi / 365.25)
    xr.Dataset(
        {"streamflow": (("basin", "date"), [flow])},
        coords={"basin": ["A"], "date": dates},
    ).to_netcdf(input_path, engine="netcdf4")
    config_path.write_text(
        "\n".join([
            "[run]",
            'station_ids = ["A"]',
            "all_stations = false",
            "continue_on_station_error = true",
            "",
            "[layers]",
            "layer1 = true",
            "layer2 = true",
            "layer3 = false",
            "comparison = false",
            "",
            "[series1]",
            'format = "netcdf"',
            f'path = "{input_path.as_posix()}"',
            'name = "reference"',
            'role = "observation"',
            'variable = "streamflow"',
            'station_coordinate = "basin"',
            'time_coordinate = "date"',
            'units = "mm/day"',
            "",
            "[output]",
            f'directory = "{output_path.as_posix()}"',
            'html_mode = "none"',
            'non_interactive_html_mode = "none"',
            "show_figures = false",
            "write_text = true",
            "write_netcdf = true",
            "write_log = true",
        ]),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-B", "-m", "trihydra.cli", "run", "--config", str(config_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "completed station=A" in completed.stdout
    assert "TriHydrA finished" in completed.stdout
    assert (output_path / "A" / "summary.txt").is_file()
    assert (output_path / "network_summary.txt").is_file()
    assert (output_path / "trihydra_run.log").is_file()
    netcdf_path = output_path / "trihydra_network_summary.nc"
    station_path = output_path / "stations" / "A.nc"
    assert netcdf_path.is_file()
    assert station_path.is_file()
    with xr.open_dataset(netcdf_path) as result:
        assert result.station.item() == "A"
        assert result["processing_status"].item() == "completed"
        assert result.attrs["completed_station_count"] == 1
    with xr.open_dataset(station_path) as result:
        assert "station" not in result.dims
        assert result.series_slot.item() == "series1"
        assert result["series_name"].item() == "reference"


def test_batch_compares_two_wide_csv_series_from_the_same_toml(tmp_path):
    dates = pd.date_range("2000-01-01", periods=730, freq="D")
    reference = 2.0 + np.sin(np.arange(len(dates)) * 2 * np.pi / 365.25)
    candidate = reference * 1.05
    first_path = tmp_path / "reference.csv"
    second_path = tmp_path / "candidate.csv"
    pd.DataFrame({"date": dates, "A": reference}).to_csv(first_path, index=False)
    pd.DataFrame({"date": dates, "A": candidate}).to_csv(second_path, index=False)
    config_path = tmp_path / "csv_comparison.toml"
    config_path.write_text(
        "\n".join([
            "[run]",
            'station_ids = ["A"]',
            "all_stations = false",
            "",
            "[layers]",
            "layer1 = true",
            "layer2 = true",
            "layer3 = false",
            "comparison = true",
            "",
            "[series1]",
            'format = "csv"',
            f'path = "{first_path.as_posix()}"',
            'name = "reference"',
            'role = "observation"',
            'units = "mm/day"',
            "",
            "[series2]",
            "enabled = true",
            'format = "csv"',
            f'path = "{second_path.as_posix()}"',
            'name = "candidate"',
            'role = "simulation"',
            'units = "mm/day"',
            "",
            "[output]",
            f'directory = "{(tmp_path / "results_csv").as_posix()}"',
            'html_mode = "none"',
            "write_text = false",
            "write_netcdf = false",
            "write_log = false",
        ]),
        encoding="utf-8",
    )

    batch = run_batch(config_path)
    manifest = batch.manifest

    assert isinstance(batch, TriHydrABatchResult)
    assert manifest.loc[0, "status"] == "completed"
    assert manifest.loc[0, "series2_status"] == "available"
    assert list(batch.station_results) == ["A"]
    assert not batch.summary.empty
    assert batch.output_directory == (tmp_path / "results_csv").resolve()
