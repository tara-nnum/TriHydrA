"""Readable tests for the public TriHydrA TOML contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from trihydra.settings.models import TriHydrAConfig


def minimal_configuration() -> dict:
    """Return one small valid NetCDF-only configuration."""
    return {
        "run": {"station_ids": ["gauge_1"], "all_stations": False},
        "layers": {
            "layer1": True,
            "layer2": True,
            "layer3": False,
            "comparison": False,
        },
        "observation": {
            "format": "netcdf",
            "path": "observations.nc",
            "units": "mm/day",
        },
    }


def test_one_or_all_station_selection_is_required():
    bad = minimal_configuration()
    bad["run"] = {"station_ids": [], "all_stations": False}

    with pytest.raises(ValidationError, match="Choose exactly one station mode"):
        TriHydrAConfig.model_validate(bad)


def test_station_file_is_a_third_selection_mode(tmp_path):
    station_file = tmp_path / "stations.txt"
    station_file.write_text("gauge_2\n# ignored comment\ngauge_1\ngauge_2\n", encoding="utf-8")
    data = minimal_configuration()
    data["run"] = {
        "station_ids": [],
        "station_file": station_file,
        "all_stations": False,
    }

    config = TriHydrAConfig.model_validate(data)

    assert config.station_selection() == ["gauge_2", "gauge_1"]


def test_unknown_configuration_names_are_rejected():
    bad = minimal_configuration()
    bad["layer2"] = {"events": {"trigger_percentil": 0.95}}

    with pytest.raises(ValidationError, match="trigger_percentil"):
        TriHydrAConfig.model_validate(bad)


def test_enabled_pickle_requires_explicit_trust():
    bad = minimal_configuration()
    bad["simulation"] = {
        "enabled": True,
        "format": "aifl_pickle",
        "path": "longtermruns",
        "trusted": False,
    }

    with pytest.raises(ValidationError, match="trusted=true"):
        TriHydrAConfig.model_validate(bad)


def test_comparison_requires_an_enabled_simulation():
    bad = minimal_configuration()
    bad["layers"]["comparison"] = True

    with pytest.raises(ValidationError, match="simulation.enabled=true"):
        TriHydrAConfig.model_validate(bad)


def test_public_input_and_output_settings_stay_in_the_public_configuration():
    data = minimal_configuration()
    data["output"] = {"html_mode": "ask"}
    config = TriHydrAConfig.model_validate(data)

    runtime = config.runtime_overrides(Path.cwd())

    assert config.output.html_mode == "ask"
    assert config.output.write_text is True
    assert config.observation.units == "mm/day"
    assert set(runtime["run"]) == {"layers"}
