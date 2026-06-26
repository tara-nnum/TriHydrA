"""
Layer 2: OBS-ML Signature Comparison
This module compares observed discharge and ML discharge behaviour
using already-calculated hydrological signature profiles.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.trihydra.layer1.timeseries_validity import (
    to_datetime_sorted,
    get_valid_record,
)

from src.trihydra.layer1.behaviour_profile import (
    calculate_profile,
    calculate_monthly_profile,
    calculate_seasonal_profile,
    calculate_yearly_profile,
    calculate_monthly_by_year_profile,
)


# def test_imports():
#     """
#     Tiny test to confirm this file can import existing profile functions.
#     """
#     print("Layer 2 signature_comparison.py is running.")
#     print("Imported behaviour profile functions successfully.")
#
#
# if __name__ == "__main__":
#     test_imports()

def prepare_obs_ml_series(
    obs_series: pd.Series,
    ml_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Prepare OBS and ML discharge series for Layer 2 comparison.

    OBS is treated as the reference record.
    ML values outside the OBS-valid period are ignored.

    Steps:
    - convert/sort datetime index
    - trim leading/trailing unavailable periods
    - use the overlapping OBS-ML period
    - align ML to OBS dates
    """
    obs = to_datetime_sorted(obs_series)
    ml = to_datetime_sorted(ml_series)

    obs_valid = get_valid_record(obs)
    ml_valid = get_valid_record(ml)

    if obs_valid.empty or ml_valid.empty:
        raise ValueError("OBS or ML series has no valid data after trimming.")

    # OBS is the reference period.
    # ML may extend beyond OBS because forecasts can continue after the historical record.
    # Those extra ML dates are not used for OBS-ML signature comparison.
    common_start = max(obs_valid.index.min(), ml_valid.index.min())
    common_end = min(obs_valid.index.max(), ml_valid.index.max())

    if common_start > common_end:
        raise ValueError("OBS and ML series do not overlap in time.")

    obs_aligned = obs_valid.loc[common_start:common_end]

    # Align ML to OBS dates.
    # This keeps the OBS comparison calendar fixed.
    # If ML is missing on an OBS date, it remains NaN instead of silently dropping the date.
    ml_aligned = ml_valid.loc[common_start:common_end].reindex(obs_aligned.index)

    if obs_aligned.empty:
        raise ValueError("OBS and ML could not be aligned on common dates.")

    return obs_aligned.rename("obs"), ml_aligned.rename("ml")


def fill_obs_for_layer2(
    obs_aligned: pd.Series,
    method: str = "seasonal_climatology",
    window_days: int = 15,
    min_samples: int = 5,
) -> pd.Series:
    """
    Temporarily fill internal OBS gaps for Layer 2 signature calculation only.

    Temporary filling for signature calculation only. Raw OBS is not modified.

    Parameters
    ----------
    obs_aligned : pd.Series
        Raw aligned OBS series.

    method : str
        Imputation method.
        Options:
        - "seasonal_climatology": fill missing values using median flow
          from the same day-of-year window across other years.
        - "ffill": simple forward-fill. Not recommended for final use.
        - "none": no filling.

    window_days : int
        Day-of-year window used for seasonal climatology filling.
        Example: 15 means same day-of-year +/- 15 days.

    min_samples : int
        Minimum number of available observations required to fill a missing value.

    Returns
    -------
    obs_l2 : pd.Series
        Temporary OBS copy used only for Layer 2 signature calculation.
    """
    obs_l2 = obs_aligned.copy()
    obs_l2 = to_datetime_sorted(obs_l2)

    if method == "none":
        return obs_l2

    if method == "ffill":
        # Temporary filling for signature calculation only. Raw OBS is not modified.
        return obs_l2.ffill()

    if method != "seasonal_climatology":
        raise ValueError(
            "method must be one of: 'seasonal_climatology', 'ffill', or 'none'."
        )

    valid_obs = obs_l2.dropna()

    if valid_obs.empty:
        return obs_l2

    missing_dates = obs_l2[obs_l2.isna()].index

    for missing_date in missing_dates:
        target_dayofyear = missing_date.dayofyear

        candidate_values = []

        for date, value in valid_obs.items():
            candidate_dayofyear = date.dayofyear

            day_distance = abs(candidate_dayofyear - target_dayofyear)

            # Handle year boundary, e.g. Jan 2 close to Dec 31.
            circular_distance = min(day_distance, 366 - day_distance)

            if circular_distance <= window_days:
                candidate_values.append(value)

        if len(candidate_values) >= min_samples:
            obs_l2.loc[missing_date] = float(np.median(candidate_values))

    return obs_l2

def calculate_layer2_profiles(
    obs_l2: pd.Series,
    ml_aligned: pd.Series,
) -> dict:
    """
    Calculate OBS and ML behaviour profiles for Layer 2 comparison.

    OBS uses the temporary Layer 2 filled copy.
    ML remains unchanged.
    """
    profiles = {
        "obs_summary": calculate_profile(obs_l2, series_name="obs"),
        "ml_summary": calculate_profile(ml_aligned, series_name="ml"),

        "obs_monthly": calculate_monthly_profile(obs_l2),
        "ml_monthly": calculate_monthly_profile(ml_aligned),

        "obs_seasonal": calculate_seasonal_profile(obs_l2),
        "ml_seasonal": calculate_seasonal_profile(ml_aligned),

        "obs_yearly": calculate_yearly_profile(obs_l2),
        "ml_yearly": calculate_yearly_profile(ml_aligned),

        "obs_monthly_by_year": calculate_monthly_by_year_profile(obs_l2),
        "ml_monthly_by_year": calculate_monthly_by_year_profile(ml_aligned),
    }

    return profiles

def extract_signature_profile(
    summary_profile: dict,
    monthly_profile: pd.DataFrame,
    yearly_profile: pd.DataFrame,
) -> dict:
    """
    Extract compact Layer 2 catalogue signatures from behaviour profiles.

    Catalogue signatures:
    - low-flow behaviour: q05
    - high-flow behaviour: q95
    - excessive zero flow: zero_ratio
    - annual maximum flow: median of yearly maxima
    - seasonality index: max monthly median / min monthly median
    - wettest month: month with highest median flow
    - driest month: month with lowest median flow
    """
    low_flow_q05 = summary_profile.get("q05", np.nan)
    high_flow_q95 = summary_profile.get("q95", np.nan)
    zero_ratio = summary_profile.get("zero_ratio", np.nan)

    if yearly_profile.empty or "max" not in yearly_profile.columns:
        annual_max_median = np.nan
    else:
        annual_max_median = float(yearly_profile["max"].median())

    if monthly_profile.empty or "median" not in monthly_profile.columns:
        seasonality_index = np.nan
        wettest_month = None
        driest_month = None
    else:
        monthly = monthly_profile.copy()

        # Convert month_end to month number if available.
        if "month_end" in monthly.columns:
            monthly["month"] = pd.to_datetime(monthly["month_end"]).dt.month
        else:
            monthly["month"] = np.nan

        # Climatological median per calendar month across all years.
        monthly_climatology = (
            monthly
            .dropna(subset=["median"])
            .groupby("month")["median"]
            .median()
        )

        if monthly_climatology.empty:
            seasonality_index = np.nan
            wettest_month = None
            driest_month = None
        else:
            wettest_month = int(monthly_climatology.idxmax())
            driest_month = int(monthly_climatology.idxmin())

            min_monthly = monthly_climatology.min()
            max_monthly = monthly_climatology.max()

            if pd.isna(min_monthly) or min_monthly == 0:
                seasonality_index = np.nan
            else:
                seasonality_index = float(max_monthly / min_monthly)

    return {
        "low_flow_q05": float(low_flow_q05) if pd.notna(low_flow_q05) else np.nan,
        "high_flow_q95": float(high_flow_q95) if pd.notna(high_flow_q95) else np.nan,
        "zero_ratio": float(zero_ratio) if pd.notna(zero_ratio) else np.nan,
        "annual_max_median": annual_max_median,
        "seasonality_index": seasonality_index,
        "wettest_month": wettest_month,
        "driest_month": driest_month,
    }

def _relative_difference_percent(obs_value, ml_value) -> float:
    """
    Calculate relative difference as percentage against OBS.

    Formula:
    ((ML - OBS) / OBS) * 100
    """
    if pd.isna(obs_value) or pd.isna(ml_value):
        return np.nan

    if obs_value == 0:
        return np.nan

    return float(((ml_value - obs_value) / obs_value) * 100)


def _get_signature_diagnostic(
    signature: str,
    obs_value,
    ml_value,
) -> tuple[str, bool]:
    """
    Generate simple catalogue-based diagnostic text and flag.
    """
    if pd.isna(obs_value) or pd.isna(ml_value):
        return "Insufficient data for comparison", False

    if signature == "low_flow_q05":
        if ml_value < obs_value:
            return "ML underestimates low-flow behaviour", True
        if ml_value > obs_value:
            return "ML overestimates low-flow/baseflow behaviour", True
        return "ML matches OBS low-flow behaviour", False

    if signature == "high_flow_q95":
        if ml_value < obs_value:
            return "ML underestimates high-flow behaviour", True
        if ml_value > obs_value:
            return "ML overestimates high-flow behaviour", True
        return "ML matches OBS high-flow behaviour", False

    if signature == "annual_max_median":
        if ml_value < obs_value:
            return "ML underestimates annual extremes", True
        if ml_value > obs_value:
            return "ML overestimates annual extremes", True
        return "ML matches OBS annual extremes", False

    if signature == "seasonality_index":
        if ml_value < obs_value:
            return "ML smooths wet/dry seasonal contrast", True
        if ml_value > obs_value:
            return "ML exaggerates wet/dry seasonal contrast", True
        return "ML matches OBS seasonal contrast", False

    if signature == "zero_ratio":
        if ml_value > obs_value:
            return "ML creates more zero-flow behaviour", True
        if ml_value < obs_value:
            return "ML creates less zero-flow behaviour", True
        return "ML matches OBS zero-flow behaviour", False

    if signature == "wettest_month":
        if ml_value != obs_value:
            return "ML wettest month differs from OBS", True
        return "ML matches OBS wettest month", False

    if signature == "driest_month":
        if ml_value != obs_value:
            return "ML driest month differs from OBS", True
        return "ML matches OBS driest month", False

    return "No diagnostic rule defined", False


def compare_signature_profiles(
    basin_id: str,
    obs_signature_profile: dict,
    ml_signature_profile: dict,
) -> pd.DataFrame:
    """
    Compare compact OBS and ML Layer 2 signature profiles.

    Output columns:
    - basin_id
    - signature
    - obs_value
    - ml_value
    - difference
    - relative_difference_percent
    - diagnostic
    - flag
    """
    rows = []

    signature_names = [
        "low_flow_q05",
        "high_flow_q95",
        "zero_ratio",
        "annual_max_median",
        "seasonality_index",
        "wettest_month",
        "driest_month",
    ]

    for signature in signature_names:
        obs_value = obs_signature_profile.get(signature, np.nan)
        ml_value = ml_signature_profile.get(signature, np.nan)

        if pd.isna(obs_value) or pd.isna(ml_value):
            difference = np.nan
        else:
            difference = ml_value - obs_value

        relative_difference_percent = _relative_difference_percent(
            obs_value=obs_value,
            ml_value=ml_value,
        )

        diagnostic, flag = _get_signature_diagnostic(
            signature=signature,
            obs_value=obs_value,
            ml_value=ml_value,
        )

        rows.append({
            "basin_id": basin_id,
            "signature": signature,
            "obs_value": obs_value,
            "ml_value": ml_value,
            "difference": difference,
            "relative_difference_percent": relative_difference_percent,
            "diagnostic": diagnostic,
            "flag": flag,
        })

    return pd.DataFrame(rows)