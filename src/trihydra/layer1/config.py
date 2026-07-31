"""Central runtime defaults for the current TriHydrA Layer 1-2 workflow.

These values are transparent software defaults, not universal hydrological
truths. They should eventually be represented by validated Pydantic models
loaded from TOML/YAML, after the scientific defaults and output contract have
been agreed. No setting is station-specific.

Layer 1 always receives the raw observations. Layer 2 may create a temporary
filled calculation copy, but it must never modify the caller's raw series.
Low- and high-flow thresholds are derived from valid, non-imputed observations.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


DISPLAY_DECIMALS = 3

DEFAULT_LAYER1_CONFIG: dict[str, dict[str, Any]] = {
    "missing_values": {"enabled": True},
    "long_gaps": {
        "enabled": True,
        "max_gap_days": 3,
    },
    "negative_discharge": {
        "enabled": True,
        "decimals": DISPLAY_DECIMALS,
    },
    "duplicate_timestamps": {"enabled": True},
    "timestep_consistency": {"enabled": True},
    "zero_flow_regime": {
        "enabled": True,
        "decimals": DISPLAY_DECIMALS,
    },
    "low_variability": {
        "enabled": True,
        "window": 21,
        "lower_quantile": 0.01,
        "spell_quantile": 0.99,
        "decimals": DISPLAY_DECIMALS,
    },
    # Spike/dip currently learns its robust cut-offs from each record. There
    # are no public fixed thresholds to configure without changing its method.
    "spike_dip": {"enabled": True},
    "step_shift": {
        "enabled": True,
        "evidence_window_months": 6,
        "cooldown_months": 6,
        "significance_level": 0.05,
        "minimum_standardised_effect": 0.75,
        "minimum_valid_days_per_month": 20,
        "include_flow_duration_curve": False,
    },
    "gradual_drift": {
        "enabled": True,
        "significance_level": 0.05,
        "min_daily_values_per_month": 20,
        "min_record_years": 3.0,
        "max_interpolated_gap_months": 2,
    },
}

DEFAULT_LAYER2_CONFIG: dict[str, Any] = {
    "temporary_imputation": {
        "method": "seasonal_climatology",
        "seasonal_window_days": 15,
        "minimum_seasonal_samples": 5,
        "allowed_methods": (
            "seasonal_climatology",
            "interpolate",
            "ffill",
            "none",
        ),
    },
    "signatures": {
        "zero_threshold": 1e-6,
        "low_flow_percentile": 0.05,
        "high_flow_percentile": 0.95,
        "minimum_year_coverage": 0.80,
        "minimum_month_coverage": 0.65,
        "autocorrelation_lags": (1, 2, 3, 7, 14, 30),
        "maximum_decay_lag": 90,
        "rising_tolerance": 0.0,
        "recession_tolerance": 0.0,
        "minimum_limb_length": 1,
        "peak_minimum_distance_days": 5,
        "peak_prominence": None,
        "peak_minimum_height": None,
        "event_threshold": None,
        "baseflow_alpha": 0.925,
        "baseflow_passes": 3,
    },
    "threshold_policy": {
        "source": "raw_valid_observations",
        "use_observation_thresholds_for_model": True,
        "temporary_fills_influence_thresholds": False,
    },
}

DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "layers": {"run_layer1": True, "run_layer2": True},
    "input": {
        "observation_unit": "mm/day",
        "discharge_variable": None,
        "time_coordinate": None,
        "station_coordinate": None,
    },
    "output": {
        "show_figures": False,
        "continue_on_station_error": True,
        "display_decimals": DISPLAY_DECIMALS,
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "run": DEFAULT_RUN_CONFIG,
    "layer1": DEFAULT_LAYER1_CONFIG,
    "layer2": DEFAULT_LAYER2_CONFIG,
}


def merge_config(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deep-merge user overrides without mutating the supplied dictionaries."""
    result = deepcopy(dict(defaults))
    if overrides is None:
        return result
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_default_config() -> dict[str, Any]:
    """Return an independent copy safe for one run to modify."""
    return deepcopy(DEFAULT_CONFIG)


__all__ = [
    "DISPLAY_DECIMALS",
    "DEFAULT_CONFIG",
    "DEFAULT_LAYER1_CONFIG",
    "DEFAULT_LAYER2_CONFIG",
    "DEFAULT_RUN_CONFIG",
    "get_default_config",
    "merge_config",
]
