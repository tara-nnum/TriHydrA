"""Baseflow signature calculation for TriHydrA Layer 2.

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

def calculate_baseflow_signatures(
    series: pd.Series,
    alpha: float = 0.925,
    passes: int = 3,
    minimum_year_coverage: float = 0.80,
) -> SignatureResult:
    """
    Separate baseflow from quickflow using the Lyne-Hollick recursive
    digital filter, and calculate the Baseflow Index (BFI).

    BFI = total baseflow volume / total discharge volume, for the whole
    record and per retained year.

    alpha=0.925 is the value most commonly recommended for daily
    streamflow (Nathan and McMahon, 1990); passes=3 (forward, backward,
    forward) is the standard scheme used to remove single-pass phase
    lag. Both are exposed as parameters, not fixed internally.

    Note: the filter is run on the valid values only, treated as
    sequential. If the record has internal gaps, this slightly biases
    the filter across a gap; a warning is recorded when that happens.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if passes < 1:
        raise ValueError("passes must be at least 1.")
    if not 0 <= minimum_year_coverage <= 1:
        raise ValueError("minimum_year_coverage must be between 0 and 1.")

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()
    warnings: list[str] = []

    if len(valid) < 30:
        warnings.append(
            "Fewer than 30 valid discharge values are available; "
            "baseflow separation is unreliable on very short records."
        )

    if valid.empty:
        baseflow_series = pd.Series(
            dtype=float,
            index=discharge.index,
            name="baseflow",
        )
    else:
        if len(valid) < len(discharge):
            warnings.append(
                f"{int(discharge.isna().sum())} internal gap day(s) were "
                "excluded before filtering; the recursive filter treats "
                "the remaining valid values as sequential, which can "
                "slightly bias results across a gap."
            )

        raw_baseflow = _lyne_hollick_filter(
            valid.to_numpy(),
            alpha=alpha,
            passes=passes,
        )

        baseflow_series = pd.Series(
            raw_baseflow,
            index=valid.index,
            name="baseflow",
        ).reindex(discharge.index)

    total_flow = discharge.sum(skipna=True)
    total_baseflow = baseflow_series.sum(skipna=True)
    whole_record_bfi = _safe_divide(total_baseflow, total_flow)

    annual_coverage = _annual_coverage_table(discharge)
    annual_rows: list[dict[str, Any]] = []

    for year, yearly_values in discharge.groupby(discharge.index.year):
        year = int(year)

        coverage_row = annual_coverage.loc[
            annual_coverage["year"] == year
        ]
        coverage_fraction = (
            float(coverage_row["coverage_fraction"].iloc[0])
            if not coverage_row.empty else np.nan
        )
        retained = (
            pd.notna(coverage_fraction)
            and coverage_fraction >= minimum_year_coverage
        )

        yearly_baseflow = baseflow_series.reindex(yearly_values.index)
        year_bfi = (
            _safe_divide(
                yearly_baseflow.sum(skipna=True),
                yearly_values.sum(skipna=True),
            )
            if retained else np.nan
        )

        annual_rows.append(
            {
                "year": year,
                "baseflow_index": year_bfi,
                "coverage_fraction": coverage_fraction,
                "retained": retained,
            }
        )

    annual_bfi = pd.DataFrame(annual_rows)
    retained_bfi = (
        annual_bfi.loc[annual_bfi["retained"], "baseflow_index"].dropna()
        if not annual_bfi.empty else pd.Series(dtype=float)
    )

    excluded_years = (
        int((~annual_bfi["retained"]).sum())
        if not annual_bfi.empty else 0
    )
    if excluded_years > 0:
        warnings.append(
            f"{excluded_years} year(s) were excluded from annual BFI "
            "calculation because of low coverage."
        )

    metrics = {
        "alpha": float(alpha),
        "passes": int(passes),
        "baseflow_index": whole_record_bfi,
        "mean_annual_baseflow_index": (
            float(retained_bfi.mean()) if not retained_bfi.empty else np.nan
        ),
        "median_annual_baseflow_index": (
            float(retained_bfi.median()) if not retained_bfi.empty else np.nan
        ),
        "minimum_annual_baseflow_index": (
            float(retained_bfi.min()) if not retained_bfi.empty else np.nan
        ),
        "maximum_annual_baseflow_index": (
            float(retained_bfi.max()) if not retained_bfi.empty else np.nan
        ),
        "number_of_years_retained": int(len(retained_bfi)),
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=30,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "baseflow_series": baseflow_series.to_frame(),
            "annual_baseflow_index": annual_bfi,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
