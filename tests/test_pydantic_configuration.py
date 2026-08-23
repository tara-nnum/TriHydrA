"""Readable tests for the public TriHydrA TOML contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from trihydra.settings.loader import (
    build_runtime_config,
    load_toml_config,
    resolve_station_selection,
)
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
        "series1": {
            "format": "netcdf",
            "path": "observations.nc",
            "name": "reference",
            "role": "observation",
            "units": "mm/day",
            "variable": "streamflow",
            "station_coordinate": "station_id",
            "time_coordinate": "date",
        },
    }


def test_shipped_toml_is_a_valid_configuration_example():
    """The user-facing configuration must not drift from supported fields."""
    root = Path(__file__).resolve().parents[1]
    config = load_toml_config(root / "trihydra.toml", check_paths=False)

    assert config.series1.format == "netcdf"
    assert config.output.write_netcdf is True


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

    assert resolve_station_selection(config) == ["gauge_2", "gauge_1"]


def test_unknown_configuration_names_are_rejected():
    bad = minimal_configuration()
    bad["layer2"] = {"events": {"trigger_percentil": 0.95}}

    with pytest.raises(ValidationError, match="trigger_percentil"):
        TriHydrAConfig.model_validate(bad)


def test_enabled_pickle_requires_explicit_trust():
    bad = minimal_configuration()
    bad["series2"] = {
        "enabled": True,
        "format": "aifl_pickle",
        "path": "longtermruns",
        "trusted": False,
    }

    with pytest.raises(ValidationError, match="trusted=true"):
        TriHydrAConfig.model_validate(bad)


def test_comparison_requires_an_enabled_second_series():
    bad = minimal_configuration()
    bad["layers"]["comparison"] = True

    with pytest.raises(ValidationError, match="series2.enabled=true"):
        TriHydrAConfig.model_validate(bad)


def test_comparison_rejects_different_discharge_units():
    """TriHydrA must never compare unlike discharge units silently."""
    bad = minimal_configuration()
    bad["layers"]["comparison"] = True
    bad["series2"] = {
        "enabled": True,
        "format": "netcdf",
        "path": "simulation.nc",
        "name": "simulation",
        "role": "simulation",
        "units": "m3/s",
        "variable": "streamflow",
        "station_coordinate": "station_id",
        "time_coordinate": "date",
    }

    with pytest.raises(ValidationError, match="units must match"):
        TriHydrAConfig.model_validate(bad)


def test_public_input_and_output_settings_stay_in_the_public_configuration():
    data = minimal_configuration()
    data["output"] = {"html_mode": "ask"}
    config = TriHydrAConfig.model_validate(data)

    runtime = build_runtime_config(config, Path.cwd())

    assert config.output.html_mode == "ask"
    assert config.output.write_text is True
    assert config.series1.units == "mm/day"
    assert set(runtime["run"]) == {"layers"}


def test_range_timespan_requires_two_ordered_dates():
    incomplete = minimal_configuration()
    incomplete["series1"]["timespan"] = {
        "mode": "range",
        "start_date": "2000-01-01",
    }
    with pytest.raises(ValidationError, match="requires start_date and end_date"):
        TriHydrAConfig.model_validate(incomplete)

    reversed_dates = minimal_configuration()
    reversed_dates["series1"]["timespan"] = {
        "mode": "range",
        "start_date": "2001-01-01",
        "end_date": "2000-01-01",
    }
    with pytest.raises(ValidationError, match="must not be after"):
        TriHydrAConfig.model_validate(reversed_dates)


def test_independent_timespans_reject_paired_daily_metrics():
    data = minimal_configuration()
    data["series2"] = {
        "enabled": True,
        "format": "netcdf",
        "path": "candidate.nc",
        "name": "candidate",
        "role": "historical_observation",
        "units": "mm/day",
        "variable": "streamflow",
        "station_coordinate": "station_id",
        "time_coordinate": "date",
    }
    data["layers"]["comparison"] = True
    data["comparison"] = {
        "mode": "independent_timespans",
        "calculate_daily_metrics": True,
    }

    with pytest.raises(ValidationError, match="unavailable for independent_timespans"):
        TriHydrAConfig.model_validate(data)


def test_disabled_optional_pickle_does_not_require_trust():
    data = minimal_configuration()
    data["series2"] = {"enabled": False, "format": "aifl_pickle"}

    config = TriHydrAConfig.model_validate(data)

    assert config.series2.enabled is False


def test_enabled_netcdf_requires_variable_and_coordinate_names():
    bad = minimal_configuration()
    bad["series1"].pop("variable")

    with pytest.raises(ValidationError, match="NetCDF inputs require: variable"):
        TriHydrAConfig.model_validate(bad)


def test_layer2_percentage_boundaries_must_be_ordered():
    bad = minimal_configuration()
    bad["layer2"] = {
        "comparison": {
            "similar_maximum_percent": 60.0,
            "review_maximum_percent": 40.0,
        }
    }

    with pytest.raises(ValidationError, match="0 <= similar < review <= 100"):
        TriHydrAConfig.model_validate(bad)


def test_comparison_requires_at_least_one_enabled_component():
    bad = minimal_configuration()
    bad["series2"] = {
        "enabled": True,
        "format": "netcdf",
        "path": "candidate.nc",
        "name": "candidate",
        "role": "simulation",
        "units": "mm/day",
        "variable": "streamflow",
        "station_coordinate": "station_id",
        "time_coordinate": "date",
    }
    bad["layers"]["comparison"] = True
    bad["layer2"] = {
        "comparison": {
            "components": {
                "flow_behaviour": False,
                "annual_flashiness_shape": False,
                "annual_baseflow_shape": False,
                "seasonal_profile_shape": False,
                "seasonal_timing": False,
                "event_time_to_peak": False,
                "event_duration": False,
                "representative_event_shape": False,
            }
        }
    }

    with pytest.raises(ValidationError, match="Enable at least one"):
        TriHydrAConfig.model_validate(bad)


def test_layer2_event_percentiles_must_define_nested_thresholds():
    bad = minimal_configuration()
    bad["layer2"] = {
        "events": {"boundary_percentile": 0.97, "trigger_percentile": 0.95}
    }

    with pytest.raises(ValidationError, match="boundary_percentile < trigger_percentile"):
        TriHydrAConfig.model_validate(bad)


def test_enabled_series_names_must_be_distinct():
    bad = minimal_configuration()
    bad["series2"] = {
        "enabled": True,
        "format": "netcdf",
        "path": "candidate.nc",
        "name": "reference",
        "role": "simulation",
        "units": "mm/day",
        "variable": "streamflow",
        "station_coordinate": "station_id",
        "time_coordinate": "date",
    }

    with pytest.raises(ValidationError, match="name and series2.name must differ"):
        TriHydrAConfig.model_validate(bad)


def test_layer3_peer_limits_and_agreement_cutoffs_are_ordered():
    peer_limits = minimal_configuration()
    peer_limits["layer3"] = {
        "local_peers": {"minimum_peers": 4, "maximum_peers": 2}
    }
    with pytest.raises(ValidationError, match="peer limits"):
        TriHydrAConfig.model_validate(peer_limits)

    agreement = minimal_configuration()
    agreement["layer3"] = {
        "comparison": {
            "partial_minimum_percent": 80.0,
            "similar_minimum_percent": 75.0,
        }
    }
    with pytest.raises(ValidationError, match="agreement cutoffs"):
        TriHydrAConfig.model_validate(agreement)
