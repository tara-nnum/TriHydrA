"""
Default configuration for TriHydrA Layer 1 QA/QC checks.

Layer 1 is metadata-independent:
- no catchment area
- no regulation information
- no aridity/climate context
- only time series behaviour
- thresholds/ranges/tolerances are not sacred, yet i.e not scientifically chosen

All checks are enabled by default.
"""

DEFAULT_LAYER1_CONFIG = {
    "missing_values": {
        "enabled": True,
        "missing_ratio_threshold": 0.05,
    },

    "long_gaps": {
        "enabled": True,
        "max_gap_days": 3,
    },

    "negative_discharge": {
        "enabled": True,
        "tolerance": -1e-6,
    },

    "duplicate_timestamps": {
        "enabled": True,
    },

    "timestep_consistency": {
        "enabled": True,
        "dominant_timestep_ratio_threshold": 0.95,
        "expected_frequency": None,
    },

    "low_variability_flow": {
        "enabled": True,
        "window_size": 7,
        "min_duration": 7,
        "variance_threshold": 1e-10,
        "cv_threshold": 0.005,
        "zero_flow_threshold": 1e-6,
        "separate_zero_flow": True,
    },

    "zero_flow_regime": {
        "enabled": True,
        "zero_flow_threshold": 1e-6,
        "min_zero_flow_spell_days": 3,
        "seasonal_month_frequency_threshold": 0.5,
        "min_years_for_seasonal_pattern": 2,
    },

    "single_point_spike_dip": {
        "enabled": True,
        "window_size": 7,
        "mad_threshold": 3.5,
        "require_reversal": True,
        "reversal_window": 2,
        "min_valid_neighbors": 5,
    },

    "step_shift": {
        "enabled": True,
        "window_size": 14,
        "relative_change_threshold": 0.5,
        "std_multiplier_threshold": 3.0,
        "min_persistence": 5,
    },

    "gradual_drift": {
        "enabled": True,
        "window_size": 30,
        "min_duration": 30,
        "slope_threshold": None,
    },
}