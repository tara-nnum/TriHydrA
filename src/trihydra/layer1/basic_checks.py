"""Orchestrator for the five structural Layer 1 checks.

Algorithms live in one module per check. This file intentionally owns only
compatibility imports and deterministic execution order so future configuration
can select checks and pass arguments without mixing algorithms into this file.
"""

import pandas as pd
from typing import Any, Mapping

from src.trihydra.layer1.config import DEFAULT_LAYER1_CONFIG, merge_config
from src.trihydra.layer1.duplicate_timestamps import check_duplicate_timestamps
from src.trihydra.layer1.long_gaps import check_long_gaps
from src.trihydra.layer1.missing_values import check_missing_values
from src.trihydra.layer1.negative_discharge import check_negative_discharge
from src.trihydra.layer1.timestep_consistency import check_timestep_consistency


def run_basic_checks(
    series: pd.Series,
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> list:
    """
    Run all structural QA/QC checks for daily discharge time series.
    """
    settings = merge_config(DEFAULT_LAYER1_CONFIG, config)
    checks = (
        ("missing_values", check_missing_values),
        ("long_gaps", check_long_gaps),
        ("negative_discharge", check_negative_discharge),
        ("duplicate_timestamps", check_duplicate_timestamps),
        ("timestep_consistency", check_timestep_consistency),
    )
    results = []
    for name, function in checks:
        kwargs = dict(settings[name])
        if bool(kwargs.pop("enabled", True)):
            results.append(function(series, **kwargs))
    return results
