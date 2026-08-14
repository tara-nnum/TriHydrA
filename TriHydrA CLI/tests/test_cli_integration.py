"""One small end-to-end test of the installed-style command-line workflow."""

import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr


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
            "[observation]",
            'format = "netcdf"',
            f'path = "{input_path.as_posix()}"',
            'name = "observation"',
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
            "display_decimals = 3",
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
    netcdf_path = output_path / "trihydra_results.nc"
    assert netcdf_path.is_file()
    with xr.open_dataset(netcdf_path) as result:
        assert result.station.item() == "A"
        assert result["processing_status"].item() == "completed"
        assert result.attrs["completed_station_count"] == 1
