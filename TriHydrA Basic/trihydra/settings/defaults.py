"""Central, non-station-specific defaults for the TriHydrA workflow.

These are transparent software defaults rather than universal hydrological
truths. Public TOML inputs are validated in ``configuration.py``. Layers 1 and
2 receive each source series unchanged; individual calculations may ignore
missing values but never fill or overwrite the source record.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


DISPLAY_DECIMALS = 3

DEFAULT_LAYER1_CONFIG: dict[str, dict[str, Any]] = {
    "missing_values": {"enabled": True},
    "long_gaps": {
        "enabled": True,
        "minimum_reported_gap_days": 3,
    },
    "negative_discharge": {
        "enabled": True,
        "tolerance": 0.001,
    },
    "duplicate_timestamps": {"enabled": True},
    "timestep_consistency": {"enabled": True},
    # Composite rules are deliberately configurable: organisations may apply
    # different review standards without changing the diagnostic algorithms.
    "composite": {
        "tier_points": {"tier_3": 0, "tier_2": 1, "tier_1": 2},
        "weights": {
            "duplicate_timestamps": 3,
            "timestep_consistency": 3,
            "missing_values": 2,
            "long_gaps": 2,
            "negative_discharge": 3,
            "spike_dip": 1,
            "low_variability": 1,
            "step_shift": 3,
            "epoch_drift": 2,
        },
        "missing_values": {
            "tier_2_minimum_percent": 5.0,
            "tier_1_above_percent": 15.0,
        },
        "long_gaps": {
            "long_gap_definition_days": 5,
            "tier_2_minimum_days": 6,
            "tier_1_minimum_days": 31,
            "tier_2_missing_share": 0.50,
            "tier_1_missing_share": 0.25,
        },
        "negative_discharge": {
            # Percentile Q05 is the conventional FDC Q95 low-flow reference.
            "low_flow_reference_quantile": 0.05,
            "tier_1_reference_multiplier": 1.0,
        },
        "low_variability": {
            "tier_1_minimum_days": 31,
        },
        "spike_dip": {
            "tier_1_minimum_unresolved_count": 6,
        },
        "classification": {
            "minor_concerns_minimum_score": 3,
            "needs_review_minimum_score": 8,
        },
    },
    "zero_flow_regime": {
        "enabled": True,
        "decimals": DISPLAY_DECIMALS,
    },
    "low_variability": {
        "enabled": True,
        "minimum_plateau_days": 15,
        "decimals": DISPLAY_DECIMALS,
    },
    # Spike/dip combines station-relative cut-offs with configurable guards.
    "spike_dip": {
        "enabled": True,
        "minimum_recovery": 0.80,
        "minimum_score": 8.0,
        "absolute_change_reference_quantile": 0.99,
        "score_reference_quantile": 0.995,
        "robust_mad_multiplier": 6.0,
        "minimum_outer_change_multiplier": 1.0,
    },
    "step_shift": {
        "enabled": True,
        "long_record_min_years": 12.0,
        "long_record_block_years": 4,
        "short_record_divisor": 3.0,
        "minimum_block_years": 1,
        "minimum_valid_days_per_month": 10,
        "minimum_block_coverage": 0.55,
        "minimum_calendar_months": 8,
        "structural_threshold": 3.0,
        # Magnitude tiers use low-flow thresholds from the station itself:
        # percentile Q05 = FDC Q95; percentile Q25 = FDC Q75.
        "tier_3_maximum_quantile": 0.05,
        "tier_1_minimum_quantile": 0.25,
        # Retained-boundary tiers are averaged into one station-level result.
        "tier_3_points": 0.0,
        "tier_2_points": 1.0,
        "tier_1_points": 2.0,
        "composite_tier_2_minimum_score": 1.0,
        "composite_tier_1_above_score": 1.5,
        # Boundary dating uses this fraction of the adaptive block length on
        # each side of the approximate block edge.
        "refinement_block_fraction": 0.5,
        "consolidation_max_block_widths": 2.0,
    },
    "epoch_drift": {
        "enabled": True,
        "minimum_valid_days_per_month": 10,
        "minimum_valid_months_per_year": 8,
        "epoch_years": 5,
        "minimum_valid_annual_levels": 5,
        "annual_noise_floor_log": 0.03,
        "meaningful_epoch_change_score": 1.0,
        # Final tiering uses the fraction of assessed years classified stable.
        "tier_3_minimum_stable_fraction": 0.75,
        "tier_2_minimum_stable_fraction": 0.50,
        "overview_epochs_per_segment": 4,
        "maximum_overview_slopes": 4,
    },
}

DEFAULT_LAYER2_CONFIG: dict[str, Any] = {
    "annual": {
        "minimum_valid_days_per_year": 30,
        "minimum_valid_days_per_month": 10,
        "baseflow_alpha": 0.925,
        "baseflow_passes": 3,
        "minimum_baseflow_segment_days": 3,
    },
    "events": {
        "trigger_percentile": 0.95,
        "boundary_percentile": 0.90,
        # A one-day event created by the candidate itself cannot corroborate it.
        "spike_crosscheck_minimum_event_duration_days": 3.0,
    },
    "comparison": {
        # Inverse JSD and cosine similarity use the same tier boundaries.
        "similarity_tier3_minimum": 0.80,
        "similarity_tier2_minimum": 0.50,
        # Fewer available components cannot support an overall classification.
        "minimum_assessable_components": 4,
        # Month differences use circular distance (for example, Dec-Jan = 1).
        "seasonal_timing_tier3_max_months": 1,
        "seasonal_timing_tier2_max_months": 3,
        "time_to_peak_tier3_max_days": 3.0,
        "time_to_peak_tier1_min_days": 5.0,
        "event_duration_tier3_max_days": 3.0,
        "event_duration_tier2_max_days": 7.0,
        "weights": {
            "flow_behaviour": 1,
            "annual_flashiness_shape": 1,
            "annual_baseflow_shape": 1,
            "seasonal_profile_shape": 1,
            "seasonal_timing": 1,
            "event_time_to_peak": 1,
            "event_duration": 1,
            "representative_event_shape": 1,
        },
        "similar_score_maximum": 2,
        "review_score_maximum": 7,
    },
}

DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "layers": {
        "run_layer1": True,
        "run_layer2": True,
        "run_comparison": True,
    },
    "input": {
        "observation_name": "observation",
        "observation_unit": "mm/day",
        "discharge_variable": None,
        "time_coordinate": None,
        "station_coordinate": None,
    },
    "output": {
        "show_figures": False,
        "continue_on_station_error": True,
        "display_decimals": DISPLAY_DECIMALS,
        # Disabled until the final mentor-approved NetCDF/Zarr schema exists.
        "write_netcdf": False,
        # Plot calculations are never skipped; these switches control only
        # whether interactive HTML files are written. Exactly one must be true.
        "plot_all": False,
        "plot_review_only": True,
        "plot_none": False,
        "write_csv": False,
        "write_log": True,
    },
}

DEFAULT_COMPARISON_CONFIG: dict[str, Any] = {
    "daily_metrics": {
        # Prefer metrics supplied by the producer. Recalculation is an
        # explicit fallback for sources that do not provide them.
        "calculate": False,
    },
    "provided_metrics": {
        "include": True,
    },
}

# Layer 3 evaluates a station against a small, relevant observation network.
# These are transparent starting values for experimentation, not universal
# hydrological thresholds.
DEFAULT_LAYER3_CONFIG: dict[str, Any] = {
    "enabled": False,
    "series_type": "observation",
    "metadata": {
        "context_path": "data/context.csv",
        "required_columns": [
            "station_id",
            "longitude",
            "latitude",
            "river_name",
            "catchment_name",
            "catchment_area_km2",
            "series_type",
        ],
        # None uses the climate lookup installed with the Python package.
        "climate_raster": None,
        "climate_legend": None,
    },
    "local_peers": {
        "minimum_peers": 1,
        "maximum_peers": 5,
        # Local peers support date-based event and change-point comparisons.
        "maximum_search_radius_km": 50.0,
        "prefer_same_catchment": True,
        "prefer_same_river": True,
    },
    "analogue_peers": {
        "minimum_peers": 2,
        "maximum_peers": 5,
        # Analogue selection is based primarily on climate and drainage scale.
        "require_same_climate": True,
        "maximum_catchment_area_ratio": 2.0,
        # This is only a broad safety guard against intercontinental matches.
        "maximum_search_radius_km": 1000.0,
    },
    "comparison": {
        # Shared Layer 2 context checks combine nearby and comparable gauges.
        # These weights are normalized if users provide values that do not sum to one.
        "local_context_weight": 0.70,
        "comparable_catchment_weight": 0.30,
        "peak_tolerance_days": 5,
        "step_shift_tolerance_days": 50,
        # At least half of the target dates must be corroborated for one peer
        # to support the target. These remain configurable experiment values.
        "minimum_peak_timing_similarity": 0.50,
        "minimum_step_shift_timing_similarity": 0.50,
        "minimum_epoch_overlap_years": 5.0,
        # Analogue similarities use a common 0-1 interpretation: one is most
        # similar. Event timing retains the established day tolerances.
        "analogue_similarity_minimum": 0.80,
        "minimum_profile_points": 6,
        "event_time_to_peak_tolerance_days": 5.0,
        "event_duration_tolerance_days": 7.0,
        "peer_consensus_fraction": 0.50,
        "similar_minimum_percent": 75.0,
        "partial_minimum_percent": 40.0,
        # Full context plots are useful only when the selected gauges provide
        # sufficient evidence that their behaviour is genuinely comparable.
        "report_minimum_similarity_percent": 50.0,
    },
    "plotting": {
        # recommended: gate reports; all: every assessable target; none: no HTML.
        "mode": "recommended",
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "run": DEFAULT_RUN_CONFIG,
    "layer1": DEFAULT_LAYER1_CONFIG,
    "layer2": DEFAULT_LAYER2_CONFIG,
    "layer3": DEFAULT_LAYER3_CONFIG,
    "comparison": DEFAULT_COMPARISON_CONFIG,
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
    "DEFAULT_COMPARISON_CONFIG",
    "DEFAULT_LAYER1_CONFIG",
    "DEFAULT_LAYER2_CONFIG",
    "DEFAULT_LAYER3_CONFIG",
    "DEFAULT_RUN_CONFIG",
    "get_default_config",
    "merge_config",
]
