"""Orchestrator for the five behavioural Layer 1 checks.

Algorithms and their implementation details live in dedicated modules. This
file provides compatibility imports and the deterministic runner used by Layer
1 diagnostics; later configuration can be integrated here.
"""

import pandas as pd
from typing import Any, Mapping

from src.trihydra.layer1.config import DEFAULT_LAYER1_CONFIG, merge_config
from src.trihydra.layer1.gradual_drift import check_gradual_drift
from src.trihydra.layer1.low_variability import check_low_variability
from src.trihydra.layer1.spike_dip import check_spike_dip
from src.trihydra.layer1.step_shift import check_step_shift
from src.trihydra.layer1.zero_flow_regime import check_zero_flow_regime


def run_behavioural_checks(
    series: pd.Series,
    series_type: str = "unknown",
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> list:
    """
    Run all behavioural Layer 1 diagnostics.
    """
    settings = merge_config(DEFAULT_LAYER1_CONFIG, config)
    checks = (
        ("zero_flow_regime", check_zero_flow_regime),
        ("low_variability", check_low_variability),
        ("spike_dip", check_spike_dip),
        ("step_shift", check_step_shift),
        ("gradual_drift", check_gradual_drift),
    )
    results = []
    for name, function in checks:
        kwargs = dict(settings[name])
        if bool(kwargs.pop("enabled", True)):
            results.append(function(series, series_type=series_type, **kwargs))
    return results
