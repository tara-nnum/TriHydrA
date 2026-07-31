"""Seasonality signature calculation for TriHydrA Layer 2.

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

def calculate_seasonality_signatures(
    series: pd.Series,
    minimum_month_coverage: float = 0.65,
) -> SignatureResult:
    """
    Calculate monthly seasonality descriptors, including the Walsh-Lawler
    Seasonality Index (SI).

    SI = (1 / R) * sum_{i=1}^{12} |x_i - R/12|

    where x_i is the climatological mean flow for calendar month i
    (averaged across all years) and R = sum(x_i), the mean-based annual
    total implied by those 12 monthly means. SI ranges from 0 (perfectly
    even across the year) to a theoretical maximum of 1.83 (all volume
    concentrated in a single month); classification bands from Walsh and
    Lawler (1981) are attached as `walsh_lawler_classification`.

    A separate, simpler descriptor (max climatological monthly median /
    min climatological monthly median) is also kept, since other parts
    of this project already use it.
    """
    if not 0 <= minimum_month_coverage <= 1:
        raise ValueError("minimum_month_coverage must be between 0 and 1.")

    discharge = _prepare_discharge_series(series)
    warnings: list[str] = []

    if discharge.empty:
        warnings.append("No valid discharge values are available.")

    monthly_rows: list[dict[str, Any]] = []

    for period, values in discharge.groupby(discharge.index.to_period("M")):
        expected_days = int(period.days_in_month)
        valid = values.dropna()
        coverage = len(valid) / expected_days

        monthly_rows.append(
            {
                "month_start": period.start_time,
                "year": int(period.year),
                "month": int(period.month),
                "valid_day_count": int(len(valid)),
                "expected_day_count": expected_days,
                "coverage_fraction": float(coverage),
                "monthly_median": (
                    float(valid.median())
                    if len(valid) > 0 and coverage >= minimum_month_coverage
                    else np.nan
                ),
                "monthly_mean": (
                    float(valid.mean())
                    if len(valid) > 0 and coverage >= minimum_month_coverage
                    else np.nan
                ),
            }
        )

    monthly_values = pd.DataFrame(monthly_rows)

    if monthly_values.empty:
        climatology = pd.DataFrame(
            columns=[
                "month",
                "climatological_monthly_median",
                "number_of_valid_years",
            ]
        )
    else:
        climatology = (
            monthly_values.dropna(subset=["monthly_median"])
            .groupby("month", as_index=False)
            .agg(
                climatological_monthly_median=("monthly_median", "median"),
                number_of_valid_years=("monthly_median", "count"),
            )
            .set_index("month")
            .reindex(range(1, 13))
            .rename_axis("month")
            .reset_index()
        )

    valid_climatology = climatology.dropna(
        subset=["climatological_monthly_median"]
    )

    # Walsh-Lawler uses climatological MONTHLY MEAN, not median.
    if monthly_values.empty:
        mean_climatology = pd.DataFrame(
            columns=["month", "climatological_monthly_mean"]
        )
    else:
        mean_climatology = (
            monthly_values.dropna(subset=["monthly_mean"])
            .groupby("month", as_index=False)
            .agg(climatological_monthly_mean=("monthly_mean", "mean"))
            .set_index("month")
            .reindex(range(1, 13))
            .rename_axis("month")
            .reset_index()
        )

    valid_mean_climatology = mean_climatology.dropna(
        subset=["climatological_monthly_mean"]
    )

    if len(valid_mean_climatology) < 12:
        walsh_lawler_seasonality_index = np.nan
    else:
        monthly_means = valid_mean_climatology[
            "climatological_monthly_mean"
        ].to_numpy()
        annual_total = monthly_means.sum()

        walsh_lawler_seasonality_index = _safe_divide(
            np.abs(monthly_means - annual_total / 12).sum(),
            annual_total,
        )

    walsh_lawler_classification = _walsh_lawler_classification(
        walsh_lawler_seasonality_index
    )

    if valid_climatology.empty:
        wettest_month = None
        driest_month = None
        maximum_monthly_median = np.nan
        minimum_monthly_median = np.nan
        seasonality_index = np.nan
        seasonal_amplitude = np.nan
        normalised_seasonal_amplitude = np.nan
    else:
        wettest_row = valid_climatology.loc[
            valid_climatology["climatological_monthly_median"].idxmax()
        ]
        driest_row = valid_climatology.loc[
            valid_climatology["climatological_monthly_median"].idxmin()
        ]

        wettest_month = int(wettest_row["month"])
        driest_month = int(driest_row["month"])
        maximum_monthly_median = float(
            wettest_row["climatological_monthly_median"]
        )
        minimum_monthly_median = float(
            driest_row["climatological_monthly_median"]
        )
        seasonal_amplitude = (
            maximum_monthly_median - minimum_monthly_median
        )

        seasonality_index = _safe_divide(
            maximum_monthly_median,
            minimum_monthly_median,
        )

        climatology_mean = valid_climatology[
            "climatological_monthly_median"
        ].mean()
        normalised_seasonal_amplitude = _safe_divide(
            seasonal_amplitude,
            climatology_mean,
        )

    if len(valid_climatology) < 12:
        warnings.append(
            f"Only {len(valid_climatology)} calendar month(s) had usable "
            "climatological medians."
        )

    metrics = {
        "minimum_month_coverage": float(minimum_month_coverage),
        "seasonality_index_max_to_min": seasonality_index,
        "seasonal_amplitude": seasonal_amplitude,
        "normalised_seasonal_amplitude": normalised_seasonal_amplitude,
        "wettest_month": wettest_month,
        "driest_month": driest_month,
        "maximum_climatological_monthly_median": maximum_monthly_median,
        "minimum_climatological_monthly_median": minimum_monthly_median,
        "walsh_lawler_seasonality_index": walsh_lawler_seasonality_index,
        "walsh_lawler_classification": walsh_lawler_classification,
    }

    return SignatureResult(
        status=_result_status(
            valid_count=len(valid_climatology),
            minimum_required=6,
            warnings=warnings,
        ),
        metrics=metrics,
        tables={
            "monthly_values": monthly_values,
            "monthly_climatology": climatology,
            "monthly_climatology_mean": mean_climatology,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
