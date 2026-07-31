"""Peaks signature calculation for TriHydrA Layer 2.

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
from scipy.signal import find_peaks

def calculate_peak_signatures(
    series: pd.Series,
    minimum_distance_days: int = 5,
    prominence: Optional[float] = None,
    minimum_height: Optional[float] = None,
    default_prominence_fraction: float = 0.10,
) -> SignatureResult:
    """
    Detect streamflow peaks and calculate peak-distribution signatures.

    Parameters
    ----------
    series
        Discharge time series.

    minimum_distance_days
        Minimum separation between detected peaks.

    prominence
        Required peak prominence. If None, prominence is estimated as:

            default_prominence_fraction * IQR(Q)

        If IQR is zero, prominence is set to zero.

    minimum_height
        Optional minimum peak discharge.

    default_prominence_fraction
        Fraction of discharge IQR used as automatic prominence.

    Returns
    -------
    SignatureResult
        Includes:
        - peak count,
        - peaks per year,
        - mean and median peak magnitude,
        - peak-magnitude variability,
        - interpeak timing,
        - dominant peak month,
        - circular mean peak month,
        - peak event table,
        - monthly peak distribution.

    Notes
    -----
    This is discharge-peak timing, not rainfall-to-flow response time.
    """
    if minimum_distance_days < 1:
        raise ValueError(
            "minimum_distance_days must be at least 1."
        )

    if default_prominence_fraction < 0:
        raise ValueError(
            "default_prominence_fraction cannot be negative."
        )

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    if len(valid) < 3:
        warnings.append(
            "At least three valid discharge values are required "
            "for peak detection."
        )

    if valid.empty:
        calculated_prominence = np.nan
    elif prominence is None:
        iqr = valid.quantile(0.75) - valid.quantile(0.25)
        calculated_prominence = float(
            max(
                0.0,
                default_prominence_fraction * iqr,
            )
        )
    else:
        calculated_prominence = float(prominence)

    timestep_days = _infer_timestep_days(discharge)

    if pd.isna(timestep_days) or timestep_days <= 0:
        distance_steps = int(minimum_distance_days)
    else:
        distance_steps = max(
            1,
            int(
                round(
                    minimum_distance_days / timestep_days
                )
            ),
        )

    # SciPy does not handle NaN safely for peak detection.
    # Missing values are temporarily replaced with -inf so they cannot
    # become peaks. Their neighbouring behaviour remains separated.
    peak_input = discharge.to_numpy(dtype=float)
    peak_input_filled = np.where(
        np.isnan(peak_input),
        -np.inf,
        peak_input,
    )

    peak_kwargs: dict[str, Any] = {
        "distance": distance_steps,
    }

    if pd.notna(calculated_prominence):
        peak_kwargs["prominence"] = calculated_prominence

    if minimum_height is not None:
        peak_kwargs["height"] = float(minimum_height)

    if len(valid) >= 3:
        peak_indices, peak_properties = find_peaks(
            peak_input_filled,
            **peak_kwargs,
        )
    else:
        peak_indices = np.array([], dtype=int)
        peak_properties = {}

    rows: list[dict[str, Any]] = []

    previous_peak_date: Optional[pd.Timestamp] = None

    for position, peak_index in enumerate(peak_indices):
        peak_date = discharge.index[peak_index]
        peak_value = discharge.iloc[peak_index]

        if pd.isna(peak_value):
            continue

        if previous_peak_date is None:
            days_since_previous_peak = np.nan
        else:
            days_since_previous_peak = float(
                (peak_date - previous_peak_date)
                / pd.Timedelta(days=1)
            )

        prominence_value = np.nan

        if (
            "prominences" in peak_properties
            and position < len(
                peak_properties["prominences"]
            )
        ):
            prominence_value = float(
                peak_properties["prominences"][position]
            )

        rows.append(
            {
                "peak_id": len(rows) + 1,
                "peak_date": peak_date,
                "peak_value": float(peak_value),
                "year": int(peak_date.year),
                "month": int(peak_date.month),
                "day_of_year": int(peak_date.dayofyear),
                "days_since_previous_peak": (
                    days_since_previous_peak
                ),
                "prominence": prominence_value,
            }
        )

        previous_peak_date = peak_date

    peaks = pd.DataFrame(rows)

    if peaks.empty:
        monthly_peak_distribution = pd.DataFrame(
            {
                "month": range(1, 13),
                "peak_count": [0] * 12,
                "peak_fraction": [0.0] * 12,
            }
        )

        annual_peak_distribution = pd.DataFrame(
            columns=[
                "year",
                "peak_count",
                "mean_peak_value",
                "maximum_peak_value",
            ]
        )

        peak_values = pd.Series(dtype=float)
        interpeak_days = pd.Series(dtype=float)
        dominant_peak_month = None
        circular_mean_peak_month = np.nan
    else:
        peak_values = peaks["peak_value"].dropna()
        interpeak_days = peaks[
            "days_since_previous_peak"
        ].dropna()

        monthly_counts = (
            peaks.groupby("month")
            .size()
            .reindex(range(1, 13), fill_value=0)
        )

        monthly_peak_distribution = pd.DataFrame(
            {
                "month": range(1, 13),
                "peak_count": monthly_counts.values,
                "peak_fraction": (
                    monthly_counts.values / len(peaks)
                ),
            }
        )

        annual_peak_distribution = (
            peaks.groupby("year", as_index=False)
            .agg(
                peak_count=("peak_id", "count"),
                mean_peak_value=("peak_value", "mean"),
                maximum_peak_value=("peak_value", "max"),
            )
        )

        dominant_peak_month = int(
            monthly_counts.idxmax()
        )

        peak_angles = (
            2
            * np.pi
            * (peaks["month"].to_numpy() - 1)
            / 12
        )

        mean_sine = np.mean(np.sin(peak_angles))
        mean_cosine = np.mean(np.cos(peak_angles))

        mean_angle = np.arctan2(
            mean_sine,
            mean_cosine,
        )

        if mean_angle < 0:
            mean_angle += 2 * np.pi

        circular_mean_peak_month = (
            mean_angle * 12 / (2 * np.pi)
        ) + 1

    duration_years = _basic_metadata(discharge)[
        "duration_years"
    ]

    metrics = {
        "minimum_distance_days": int(minimum_distance_days),
        "distance_steps": int(distance_steps),
        "prominence_used": calculated_prominence,
        "minimum_height": (
            float(minimum_height)
            if minimum_height is not None
            else np.nan
        ),
        "peak_count": int(len(peaks)),
        "peak_frequency_per_year": _safe_divide(
            len(peaks),
            duration_years,
        ),
        "mean_peak_magnitude": (
            float(peak_values.mean())
            if not peak_values.empty
            else np.nan
        ),
        "median_peak_magnitude": (
            float(peak_values.median())
            if not peak_values.empty
            else np.nan
        ),
        "maximum_peak_magnitude": (
            float(peak_values.max())
            if not peak_values.empty
            else np.nan
        ),
        "minimum_peak_magnitude": (
            float(peak_values.min())
            if not peak_values.empty
            else np.nan
        ),
        "peak_magnitude_standard_deviation": (
            float(peak_values.std(ddof=1))
            if len(peak_values) >= 2
            else np.nan
        ),
        "peak_magnitude_cv": _calculate_cv(peak_values),
        "mean_interpeak_time_days": (
            float(interpeak_days.mean())
            if not interpeak_days.empty
            else np.nan
        ),
        "median_interpeak_time_days": (
            float(interpeak_days.median())
            if not interpeak_days.empty
            else np.nan
        ),
        "minimum_interpeak_time_days": (
            float(interpeak_days.min())
            if not interpeak_days.empty
            else np.nan
        ),
        "maximum_interpeak_time_days": (
            float(interpeak_days.max())
            if not interpeak_days.empty
            else np.nan
        ),
        "dominant_peak_month": dominant_peak_month,
        "circular_mean_peak_month": (
            float(circular_mean_peak_month)
            if pd.notna(circular_mean_peak_month)
            else np.nan
        ),
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=3,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "peaks": peaks,
            "monthly_peak_distribution": (
                monthly_peak_distribution
            ),
            "annual_peak_distribution": (
                annual_peak_distribution
            ),
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
