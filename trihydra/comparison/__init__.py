"""Source-neutral time-series comparison tools."""

from trihydra.comparison.preparation import (
    ComparisonSeries,
    PreparedComparison,
    prepare_historical_comparison,
    prepare_independent_comparison,
    prepare_paired_comparison,
)
from trihydra.comparison.calculations import run_generic_comparison
from trihydra.settings.defaults import DEFAULT_COMPARISON_CONFIG

__all__ = [
    "ComparisonSeries",
    "PreparedComparison",
    "prepare_historical_comparison",
    "prepare_independent_comparison",
    "prepare_paired_comparison",
    "run_generic_comparison",
    "DEFAULT_COMPARISON_CONFIG",
]
