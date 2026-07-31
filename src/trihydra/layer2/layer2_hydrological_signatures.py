"""Compatibility orchestrator for the 13 active Layer 2 signature modules.

Actual algorithms live one-per-module. This file owns ordered orchestration,
summary flattening, and the established CSV convenience entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from src.trihydra.layer2.signature_result import SignatureResult
from src.trihydra.layer2.signature_utils import load_discharge_csv, percentile_diagnostic
from src.trihydra.layer2.flow_magnitude import calculate_flow_magnitude_signatures
from src.trihydra.layer2.low_flow import calculate_low_flow_signatures
from src.trihydra.layer2.high_flow import calculate_high_flow_signatures
from src.trihydra.layer2.annual_maximum import calculate_annual_maximum_signatures
from src.trihydra.layer2.zero_flow import calculate_zero_flow_signatures
from src.trihydra.layer2.baseflow import calculate_baseflow_signatures
from src.trihydra.layer2.flashiness import calculate_flashiness_signatures
from src.trihydra.layer2.autocorrelation import calculate_autocorrelation_signatures
from src.trihydra.layer2.rising_limb import calculate_rising_limb_signatures
from src.trihydra.layer2.recession_limb import calculate_recession_limb_signatures
from src.trihydra.layer2.peaks import calculate_peak_signatures
from src.trihydra.layer2.seasonality import calculate_seasonality_signatures
from src.trihydra.layer2.threshold_event_hydrographs import calculate_threshold_event_hydrographs
def calculate_all_hydrological_signatures(
    series: pd.Series,
    zero_threshold: float = 1e-6,
    low_flow_percentile: float = 0.05,
    high_flow_percentile: float = 0.95,
    low_flow_threshold: Optional[float] = None,
    high_flow_threshold: Optional[float] = None,
    minimum_year_coverage: float = 0.80,
    minimum_month_coverage: float = 0.65,
    autocorrelation_lags: Iterable[int] = (1, 2, 3, 7, 14, 30),
    maximum_decay_lag: int = 90,
    rising_tolerance: float = 0.0,
    recession_tolerance: float = 0.0,
    minimum_limb_length: int = 1,
    peak_minimum_distance_days: int = 5,
    peak_prominence: Optional[float] = None,
    peak_minimum_height: Optional[float] = None,
    event_threshold: Optional[float] = None,
    baseflow_alpha: float = 0.925,
    baseflow_passes: int = 3,
) -> dict[str, SignatureResult]:
    """Calculate the complete current Layer 2 discharge-only signature set."""
    results: dict[str, SignatureResult] = {
        "flow_magnitude": calculate_flow_magnitude_signatures(series),
        "low_flow": calculate_low_flow_signatures(
            series=series,
            threshold=low_flow_threshold,
            percentile=low_flow_percentile,
        ),
        "high_flow": calculate_high_flow_signatures(
            series=series,
            threshold=high_flow_threshold,
            percentile=high_flow_percentile,
        ),
        "annual_maximum": calculate_annual_maximum_signatures(
            series=series,
            minimum_year_coverage=minimum_year_coverage,
        ),
        "zero_flow": calculate_zero_flow_signatures(
            series=series,
            zero_threshold=zero_threshold,
        ),
        "baseflow": calculate_baseflow_signatures(
            series=series,
            alpha=baseflow_alpha,
            passes=baseflow_passes,
            minimum_year_coverage=minimum_year_coverage,
        ),
        "seasonality": calculate_seasonality_signatures(
            series=series,
            minimum_month_coverage=minimum_month_coverage,
        ),
        "flashiness": calculate_flashiness_signatures(
            series=series,
            minimum_year_coverage=minimum_year_coverage,
        ),
        "autocorrelation": calculate_autocorrelation_signatures(
            series=series,
            lags=autocorrelation_lags,
            maximum_decay_lag=maximum_decay_lag,
        ),
        "rising_limb": calculate_rising_limb_signatures(
            series=series,
            tolerance=rising_tolerance,
            minimum_limb_length=minimum_limb_length,
        ),
        "recession_limb": calculate_recession_limb_signatures(
            series=series,
            tolerance=recession_tolerance,
            minimum_limb_length=minimum_limb_length,
        ),
        "peaks": calculate_peak_signatures(
            series=series,
            minimum_distance_days=peak_minimum_distance_days,
            prominence=peak_prominence,
            minimum_height=peak_minimum_height,
        ),
        "threshold_event_hydrographs": calculate_threshold_event_hydrographs(
            series=series,
            threshold=(
                event_threshold
                if event_threshold is not None
                else high_flow_threshold
            ),
            percentile=high_flow_percentile,
        ),
    }
    return results

def signatures_to_summary_table(
    results: dict[str, SignatureResult],
    series_name: str = "discharge",
) -> pd.DataFrame:
    """Flatten scalar signature metrics into one tidy table."""
    rows: list[dict[str, Any]] = []

    for group_name, result in results.items():
        for metric_name, value in result.metrics.items():
            rows.append(
                {
                    "series_name": series_name,
                    "signature_group": group_name,
                    "metric": metric_name,
                    "value": value,
                    "status": result.status,
                }
            )

    return pd.DataFrame(rows)

def run_hydrological_signatures_from_csv(
    csv_path: str | Path,
    date_column: Optional[str] = None,
    discharge_column: Optional[str] = None,
    latest_years: Optional[float] = None,
    series_name: str = "discharge",
    loader_kwargs: Optional[dict[str, Any]] = None,
    signature_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Load one CSV and calculate all signatures.

    Example
    -------
    output = run_hydrological_signatures_from_csv(
        "station.csv",
        date_column="Date",
        discharge_column="Obs",
    )
    """
    loader_kwargs = {} if loader_kwargs is None else dict(loader_kwargs)
    signature_kwargs = (
        {} if signature_kwargs is None else dict(signature_kwargs)
    )

    series = load_discharge_csv(
        csv_path=csv_path,
        date_column=date_column,
        discharge_column=discharge_column,
        latest_years=latest_years,
        series_name=series_name,
        **loader_kwargs,
    )

    results = calculate_all_hydrological_signatures(
        series=series,
        **signature_kwargs,
    )

    summary = signatures_to_summary_table(
        results=results,
        series_name=series_name,
    )

    return {
        "series": series,
        "results": results,
        "summary": summary,
    }
