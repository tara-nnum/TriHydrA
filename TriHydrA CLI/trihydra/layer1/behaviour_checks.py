"""Run the five behavioural Layer 1 checks in a fixed order."""

from typing import Any, Mapping

import pandas as pd

from trihydra.settings.defaults import DEFAULT_LAYER1_CONFIG, merge_config
from trihydra.layer1.epoch_drift import check_epoch_drift
from trihydra.layer1.low_variability import check_low_variability
from trihydra.layer1.spike_dip import check_spike_dip
from trihydra.layer1.step_shift import check_step_shift
from trihydra.layer1.zero_flow_regime import check_zero_flow_regime


BEHAVIOURAL_CHECKS = (
    ("zero_flow_regime", check_zero_flow_regime),
    ("low_variability", check_low_variability),
    ("spike_dip", check_spike_dip),
    ("step_shift", check_step_shift),
    ("epoch_drift", check_epoch_drift),
)


def run_behavioural_checks(
    series: pd.Series,
    series_type: str = "unknown",
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict]:
    """Run every enabled behavioural check on the supplied series."""
    settings = merge_config(DEFAULT_LAYER1_CONFIG, config)
    results = []
    for name, function in BEHAVIOURAL_CHECKS:
        kwargs = dict(settings[name])
        if bool(kwargs.pop("enabled", True)):
            results.append(function(series, series_type=series_type, **kwargs))
    return results
