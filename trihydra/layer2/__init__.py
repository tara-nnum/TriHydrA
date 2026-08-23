"""Hydrological signatures and event diagnostics (Layer 2)."""

from trihydra.layer2.annual_signatures import (
    build_diagnostic_summary,
    build_seasonality_profile,
    calculate_annual_signatures,
)
from trihydra.layer2.diagnostics import run_layer2_diagnostics
from trihydra.layer2.hydrograph_information import (
    calculate_high_flow_events,
    select_representative_event,
)
from trihydra.layer2.peak_outlier_crosscheck import (
    crosscheck_peak_outliers,
    mark_representative_eligibility,
)

__all__ = [
    "build_diagnostic_summary",
    "build_seasonality_profile",
    "calculate_annual_signatures",
    "calculate_high_flow_events",
    "crosscheck_peak_outliers",
    "mark_representative_eligibility",
    "run_layer2_diagnostics",
    "select_representative_event",
]
