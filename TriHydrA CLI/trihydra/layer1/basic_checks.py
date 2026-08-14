"""Run the five structural Layer 1 checks in a fixed order."""

from typing import Any, Mapping

import pandas as pd

from trihydra.settings.defaults import DEFAULT_LAYER1_CONFIG, merge_config
from trihydra.layer1.duplicate_timestamps import check_duplicate_timestamps
from trihydra.layer1.long_gaps import check_long_gaps
from trihydra.layer1.missing_values import check_missing_values
from trihydra.layer1.negative_discharge import check_negative_discharge
from trihydra.layer1.timestep_consistency import check_timestep_consistency


BASIC_CHECKS = (
    ("missing_values", check_missing_values),
    ("long_gaps", check_long_gaps),
    ("negative_discharge", check_negative_discharge),
    ("duplicate_timestamps", check_duplicate_timestamps),
    ("timestep_consistency", check_timestep_consistency),
)


def run_basic_checks(
    series: pd.Series,
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict]:
    """Run every enabled structural check on the supplied series."""
    settings = merge_config(DEFAULT_LAYER1_CONFIG, config)
    results = []
    for name, function in BASIC_CHECKS:
        kwargs = dict(settings[name])
        if bool(kwargs.pop("enabled", True)):
            results.append(function(series, **kwargs))
    return results
