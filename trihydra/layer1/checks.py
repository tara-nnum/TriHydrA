"""Run the configured Layer 1 checks once in their documented order."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from trihydra.layer1.duplicate_timestamps import check_duplicate_timestamps
from trihydra.layer1.epoch_drift import check_epoch_drift
from trihydra.layer1.long_gaps import check_long_gaps
from trihydra.layer1.low_variability import check_low_variability
from trihydra.layer1.missing_values import check_missing_values
from trihydra.layer1.negative_discharge import check_negative_discharge
from trihydra.layer1.spike_dip import check_spike_dip
from trihydra.layer1.step_shift import check_step_shift
from trihydra.layer1.timestep_consistency import check_timestep_consistency
from trihydra.layer1.zero_flow_regime import check_zero_flow_regime
from trihydra.settings.defaults import DEFAULT_LAYER1_CONFIG, merge_config


BASIC_CHECKS = (
    ("missing_values", check_missing_values),
    ("long_gaps", check_long_gaps),
    ("negative_discharge", check_negative_discharge),
    ("duplicate_timestamps", check_duplicate_timestamps),
    ("timestep_consistency", check_timestep_consistency),
)

BEHAVIOURAL_CHECKS = (
    ("zero_flow_regime", check_zero_flow_regime),
    ("low_variability", check_low_variability),
    ("spike_dip", check_spike_dip),
    ("step_shift", check_step_shift),
    ("epoch_drift", check_epoch_drift),
)


def run_layer1_checks(
    series: pd.Series,
    series_type: str,
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict]:
    """Run every enabled structural and behavioural check on one series."""
    settings = merge_config(DEFAULT_LAYER1_CONFIG, config)
    results: list[dict] = []
    for group, checks in (("basic", BASIC_CHECKS), ("behavioural", BEHAVIOURAL_CHECKS)):
        for name, function in checks:
            kwargs = dict(settings[name])
            if not bool(kwargs.pop("enabled", True)):
                continue
            if group == "behavioural":
                result = function(series, series_type=series_type, **kwargs)
            else:
                result = function(series, **kwargs)
            result.setdefault("check_group", group)
            result.setdefault("execution_status", "completed")
            result.setdefault(
                "finding_status",
                "candidate_detected" if result.get("flag") else "passed",
            )
            result.setdefault("reason_skipped", None)
            result["series_type"] = series_type
            results.append(result)
    return results


__all__ = ["run_layer1_checks"]
