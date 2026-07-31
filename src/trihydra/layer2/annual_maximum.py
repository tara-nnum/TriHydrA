"""Annual maximum signature calculation for TriHydrA Layer 2.

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

def calculate_annual_maximum_signatures(
    series: pd.Series,
    minimum_year_coverage: float = 0.80,
) -> SignatureResult:
    """
    Calculate annual maximum-flow signatures.

    Mean annual flood is calculated as the mean of valid annual maxima.

    Parameters
    ----------
    series
        Discharge time series.

    minimum_year_coverage
        Minimum annual data coverage required for a year to be retained.

    Returns
    -------
    SignatureResult
        Includes:
        - mean annual flood,
        - median annual maximum,
        - variability of annual maxima,
        - annual maximum dates,
        - annual data coverage.
    """
    if not 0 <= minimum_year_coverage <= 1:
        raise ValueError(
            "minimum_year_coverage must be between 0 and 1."
        )

    discharge = _prepare_discharge_series(series)
    warnings: list[str] = []

    coverage_table = _annual_coverage_table(discharge)

    annual_rows: list[dict[str, Any]] = []

    for year, yearly_values in discharge.groupby(discharge.index.year):
        year = int(year)
        valid = yearly_values.dropna()

        coverage_row = coverage_table.loc[
            coverage_table["year"] == year
        ]

        if coverage_row.empty:
            coverage_fraction = np.nan
            valid_day_count = int(valid.size)
            expected_day_count = np.nan
        else:
            coverage_fraction = float(
                coverage_row["coverage_fraction"].iloc[0]
            )
            valid_day_count = int(
                coverage_row["valid_day_count"].iloc[0]
            )
            expected_day_count = int(
                coverage_row["expected_day_count"].iloc[0]
            )

        retained = (
            not valid.empty
            and pd.notna(coverage_fraction)
            and coverage_fraction >= minimum_year_coverage
        )

        if valid.empty:
            annual_maximum = np.nan
            annual_maximum_date = pd.NaT
        else:
            annual_maximum = float(valid.max())
            annual_maximum_date = valid.idxmax()

        annual_rows.append(
            {
                "year": year,
                "annual_maximum": annual_maximum,
                "annual_maximum_date": annual_maximum_date,
                "annual_maximum_day_of_year": (
                    int(annual_maximum_date.dayofyear)
                    if pd.notna(annual_maximum_date)
                    else np.nan
                ),
                "valid_day_count": valid_day_count,
                "expected_day_count": expected_day_count,
                "coverage_fraction": coverage_fraction,
                "retained": retained,
            }
        )

    annual_maxima = pd.DataFrame(annual_rows)

    if annual_maxima.empty:
        retained_maxima = pd.Series(dtype=float)
    else:
        retained_maxima = annual_maxima.loc[
            annual_maxima["retained"],
            "annual_maximum",
        ].dropna()

    excluded_years = (
        int((~annual_maxima["retained"]).sum())
        if not annual_maxima.empty
        else 0
    )

    if excluded_years > 0:
        warnings.append(
            f"{excluded_years} year(s) were excluded because annual "
            f"coverage was below {minimum_year_coverage:.0%}."
        )

    metrics = {
        "minimum_year_coverage": float(minimum_year_coverage),
        "number_of_years_total": int(len(annual_maxima)),
        "number_of_years_retained": int(len(retained_maxima)),
        "number_of_years_excluded": excluded_years,
        "mean_annual_flood": (
            float(retained_maxima.mean())
            if not retained_maxima.empty
            else np.nan
        ),
        "median_annual_maximum": (
            float(retained_maxima.median())
            if not retained_maxima.empty
            else np.nan
        ),
        "maximum_annual_maximum": (
            float(retained_maxima.max())
            if not retained_maxima.empty
            else np.nan
        ),
        "minimum_annual_maximum": (
            float(retained_maxima.min())
            if not retained_maxima.empty
            else np.nan
        ),
        "standard_deviation_annual_maximum": (
            float(retained_maxima.std(ddof=1))
            if len(retained_maxima) >= 2
            else np.nan
        ),
        "cv_annual_maximum": _calculate_cv(retained_maxima),
        "annual_maximum_q05": (
            float(retained_maxima.quantile(0.05))
            if not retained_maxima.empty
            else np.nan
        ),
        "annual_maximum_q95": (
            float(retained_maxima.quantile(0.95))
            if not retained_maxima.empty
            else np.nan
        ),
    }

    status = _result_status(
        valid_count=len(retained_maxima),
        minimum_required=2,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "annual_maxima": annual_maxima,
            "annual_coverage": coverage_table,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
