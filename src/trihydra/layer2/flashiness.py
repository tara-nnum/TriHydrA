"""Flashiness signature calculation for TriHydrA Layer 2.

This module owns this complete single-series signature calculation. It performs
no OBS–ML comparison and writes no files. Temporary imputation and provenance
are handled before the calculator is called.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.trihydra.layer2.signature_result import SignatureResult
from src.trihydra.layer2.signature_utils import *

def calculate_flashiness_signatures(
    series: pd.Series,
    minimum_year_coverage: float = 0.80,
) -> SignatureResult:
    """
    Calculate whole-record and annual Richards-Baker Flashiness Index.

    Larger values indicate stronger day-to-day variation relative to
    total discharge volume.
    """
    if not 0 <= minimum_year_coverage <= 1:
        raise ValueError(
            "minimum_year_coverage must be between 0 and 1."
        )

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    whole_record_rbi = _richards_baker_flashiness(discharge)

    annual_coverage = _annual_coverage_table(discharge)
    annual_rows: list[dict[str, Any]] = []

    for year, yearly_values in discharge.groupby(discharge.index.year):
        year = int(year)

        coverage_row = annual_coverage.loc[
            annual_coverage["year"] == year
        ]

        if coverage_row.empty:
            coverage_fraction = np.nan
        else:
            coverage_fraction = float(
                coverage_row["coverage_fraction"].iloc[0]
            )

        retained = (
            pd.notna(coverage_fraction)
            and coverage_fraction >= minimum_year_coverage
        )

        annual_rows.append(
            {
                "year": year,
                "flashiness_index": (
                    _richards_baker_flashiness(yearly_values)
                    if retained
                    else np.nan
                ),
                "coverage_fraction": coverage_fraction,
                "retained": retained,
            }
        )

    annual_flashiness = pd.DataFrame(annual_rows)

    if annual_flashiness.empty:
        retained_rbi = pd.Series(dtype=float)
    else:
        retained_rbi = annual_flashiness.loc[
            annual_flashiness["retained"],
            "flashiness_index",
        ].dropna()

    excluded_years = (
        int((~annual_flashiness["retained"]).sum())
        if not annual_flashiness.empty
        else 0
    )

    if excluded_years > 0:
        warnings.append(
            f"{excluded_years} year(s) were excluded from annual "
            "flashiness calculation because of low coverage."
        )

    metrics = {
        "whole_record_flashiness_index": whole_record_rbi,
        "mean_annual_flashiness_index": (
            float(retained_rbi.mean())
            if not retained_rbi.empty
            else np.nan
        ),
        "median_annual_flashiness_index": (
            float(retained_rbi.median())
            if not retained_rbi.empty
            else np.nan
        ),
        "minimum_annual_flashiness_index": (
            float(retained_rbi.min())
            if not retained_rbi.empty
            else np.nan
        ),
        "maximum_annual_flashiness_index": (
            float(retained_rbi.max())
            if not retained_rbi.empty
            else np.nan
        ),
        "standard_deviation_annual_flashiness": (
            float(retained_rbi.std(ddof=1))
            if len(retained_rbi) >= 2
            else np.nan
        ),
        "cv_annual_flashiness": _calculate_cv(retained_rbi),
        "number_of_years_retained": int(len(retained_rbi)),
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=2,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "annual_flashiness": annual_flashiness,
            "annual_coverage": annual_coverage,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
