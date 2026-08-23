"""Focused end-to-end checks for the unified public workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trihydra import load_stations, run_batch, run_trihydra_batch
from trihydra.settings import build_runtime_config, load_toml_config


def _wide_csv(path: Path, stations: tuple[str, ...] = ("A", "B")) -> None:
    dates = pd.date_range("2000-01-01", periods=730, freq="D")
    phase = np.arange(len(dates)) * 2 * np.pi / 365.25
    data: dict[str, object] = {"date": dates}
    for offset, station in enumerate(stations):
        data[station] = 2.0 + offset + np.sin(phase + offset * 0.1)
    pd.DataFrame(data).to_csv(path, index=False)


def _toml(
    path: Path,
    source: Path,
    *,
    selection: str,
    output: Path,
    series1_role: str = "observation",
    extra: str = "",
    layer2: bool = True,
) -> None:
    path.write_text(
        "\n".join([
            "[run]",
            selection,
            "all_stations = false" if "all_stations" not in selection else "",
            "continue_on_station_error = true",
            "",
            "[layers]",
            "layer1 = true",
            f"layer2 = {str(layer2).lower()}",
            "layer3 = false",
            f"comparison = {str('[series2]' in extra).lower()}",
            "",
            "[series1]",
            'format = "csv"',
            f'path = "{source.as_posix()}"',
            'name = "primary"',
            f'role = "{series1_role}"',
            'units = "mm/day"',
            "",
            "[output]",
            f'directory = "{output.as_posix()}"',
            'html_mode = "none"',
            'non_interactive_html_mode = "none"',
            "show_figures = false",
            "write_text = false",
            "write_netcdf = false",
            "write_log = false",
            "",
            extra,
        ]),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ('station_ids = ["B"]', ["B"]),
        ('station_file = "stations.txt"', ["B", "A"]),
        ("all_stations = true", ["A", "B"]),
    ],
)
def test_every_station_selection_mode_reaches_the_public_loader(
    tmp_path, selection, expected
):
    source = tmp_path / "flows.csv"
    _wide_csv(source)
    (tmp_path / "stations.txt").write_text("B\n# comment\nA\n", encoding="utf-8")
    config = tmp_path / "selection.toml"
    _toml(config, source, selection=selection, output=tmp_path / "unused")

    stations = load_stations(config)

    assert [station.station_id for station in stations] == expected


@pytest.mark.parametrize("role", ["observation", "simulation", "historical_observation"])
def test_primary_series_role_describes_but_does_not_block_calculation(tmp_path, role):
    source = tmp_path / f"{role}.csv"
    _wide_csv(source, ("A",))
    config = tmp_path / f"{role}.toml"
    _toml(
        config, source, selection='station_ids = ["A"]',
        output=tmp_path / f"out_{role}", series1_role=role,
    )

    batch = run_batch(config)

    assert batch.manifest.loc[0, "status"] == "completed"
    assert batch.station_results["A"].station.series1_role == role
    assert batch.station_results["A"].layer1 is not None
    assert batch.station_results["A"].layer2 is not None


def test_missing_optional_series2_continues_primary_assessment(tmp_path):
    primary = tmp_path / "primary.csv"
    secondary = tmp_path / "secondary.csv"
    _wide_csv(primary, ("A", "B"))
    _wide_csv(secondary, ("A",))
    config = tmp_path / "missing_secondary.toml"
    _toml(
        config, primary, selection='station_ids = ["B"]',
        output=tmp_path / "missing_secondary_output",
        extra="\n".join([
            "[series2]",
            "enabled = true",
            'format = "csv"',
            f'path = "{secondary.as_posix()}"',
            'name = "candidate"',
            'role = "simulation"',
            'units = "mm/day"',
        ]),
    )

    batch = run_batch(config)
    result = batch.station_results["B"]

    assert batch.manifest.loc[0, "status"] == "completed"
    assert batch.manifest.loc[0, "series2_status"] == "not_available"
    assert result.layer1 is not None and result.layer2 is not None
    assert result.comparison is None


def test_same_source_supports_independent_historical_timespans(tmp_path):
    source = tmp_path / "long_record.csv"
    dates = pd.date_range("1990-01-01", "2009-12-31", freq="D")
    values = 2.0 + np.sin(np.arange(len(dates)) * 2 * np.pi / 365.25)
    pd.DataFrame({"date": dates, "A": values}).to_csv(source, index=False)
    config = tmp_path / "historical.toml"
    config.write_text("\n".join([
        "[run]", 'station_ids = ["A"]', "all_stations = false", "",
        "[layers]", "layer1 = true", "layer2 = true", "layer3 = false",
        "comparison = true", "",
        "[series1]", 'format = "csv"', f'path = "{source.as_posix()}"',
        'name = "historical"', 'role = "historical_observation"',
        'units = "mm/day"', "[series1.timespan]", 'mode = "range"',
        "start_date = 1990-01-01", "end_date = 1999-12-31", "",
        "[series2]", "enabled = true", 'format = "csv"',
        f'path = "{source.as_posix()}"', 'name = "recent"',
        'role = "observation"', 'units = "mm/day"',
        "[series2.timespan]", 'mode = "range"',
        "start_date = 2000-01-01", "end_date = 2009-12-31", "",
        "[comparison]", 'mode = "independent_timespans"',
        "calculate_daily_metrics = false", "", "[output]",
        f'directory = "{(tmp_path / "historical_output").as_posix()}"',
        'html_mode = "none"', "write_text = false", "write_netcdf = false",
        "write_log = false",
    ]), encoding="utf-8")

    result = run_batch(config).station_results["A"]

    assert result.comparison["mode"] == "independent_timespans"
    assert result.comparison["daily_metrics"].empty
    assert result.station.series1.index.max().year == 1999
    assert result.station.series2.index.min().year == 2000


def test_direct_and_configured_runs_match_and_output_toggles_are_honoured(tmp_path):
    source = tmp_path / "flows.csv"
    _wide_csv(source, ("A",))
    config = tmp_path / "equivalence.toml"
    output = tmp_path / "equivalence_output"
    _toml(
        config, source, selection='station_ids = ["A"]', output=output,
        layer2=False,
    )
    text = config.read_text(encoding="utf-8").replace(
        'html_mode = "none"', 'html_mode = "all"', 1
    )
    config.write_text(text, encoding="utf-8")

    public = load_toml_config(config)
    stations = load_stations(config)
    direct = run_trihydra_batch(
        stations,
        config=build_runtime_config(public, config.parent),
        continue_on_station_error=True,
    )
    configured = run_batch(config)

    pd.testing.assert_frame_equal(
        direct.station_results["A"].summary.reset_index(drop=True),
        configured.station_results["A"].summary.reset_index(drop=True),
        check_dtype=False,
    )
    assert (output / "A" / "layer1.html").is_file()
    assert not (output / "A" / "summary.txt").exists()
    assert not (output / "trihydra_network_summary.nc").exists()
    assert not (output / "trihydra_run.log").exists()
