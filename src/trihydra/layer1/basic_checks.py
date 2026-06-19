import pandas as pd

from src.trihydra.layer1.timeseries_validity import (
    get_valid_record,
    timestamps_to_strings,
    to_datetime_sorted,
)

# Check for missing values

def check_missing_values(series: pd.Series) -> dict:
    """
    Flag missing values only inside the valid record period.
    For daily data, missing value count equals missing day count.
    """
    valid_record = get_valid_record(series)

    if valid_record.empty:
        return {
            "check": "missing_values",
            "flag": True,
            "value": None,
            "flagged_timestamps": [],
            "message": "No valid data found.",
        }

    missing_mask = valid_record.isna()
    missing_count = int(missing_mask.sum())

    return {
        "check": "missing_values",
        "flag": missing_count > 0,
        "value": missing_count,
        "record_start": str(valid_record.index.min()),
        "record_end": str(valid_record.index.max()),
        "flagged_timestamps": timestamps_to_strings(valid_record.index[missing_mask]),
        "message": f"Internal missing day count = {missing_count}.",
    }


# Check for long gaps

def check_long_gaps(series: pd.Series, max_gap_days: int = 3) -> dict:
    """
    Flag internal missing gaps longer than max_gap_days.
    Assumes daily data, so consecutive missing values are counted as missing days.
    """
    valid_record = get_valid_record(series)

    if valid_record.empty:
        return {
            "check": "long_gaps",
            "flag": False,
            "value": 0,
            "flagged_timestamps": [],
            "message": "No valid data found.",
        }

    missing_mask = valid_record.isna()

    gap_id = (missing_mask != missing_mask.shift()).cumsum()
    gap_lengths = missing_mask.groupby(gap_id).sum()

    flagged_timestamps = []
    longest_gap = 0

    for group, length in gap_lengths.items():
        length = int(length)

        if length > 0:
            longest_gap = max(longest_gap, length)

        if length > max_gap_days:
            group_mask = gap_id == group
            flagged_timestamps.extend(valid_record.index[group_mask & missing_mask])

    return {
        "check": "long_gaps",
        "flag": len(flagged_timestamps) > 0,
        "value": longest_gap,
        "threshold": f">{max_gap_days} consecutive missing days",
        "flagged_timestamps": timestamps_to_strings(flagged_timestamps),
        "message": f"Longest internal missing gap = {longest_gap} day(s).",
    }


# Check for negative discharge

def check_negative_discharge(series: pd.Series, decimals: int = 3) -> dict:
    """
    Check whether meaningful negative discharge values exist.
    Tiny negative numerical artefacts are ignored after rounding.
    """
    s = get_valid_record(series)

    if s.empty:
        return {
            "check": "negative_discharge",
            "flag": False,
            "value": 0,
            "flagged_timestamps": [],
            "message": "No valid data found.",
        }

    rounded = s.round(decimals)
    negative_mask = rounded < 0
    negative_count = int(negative_mask.sum())

    return {
        "check": "negative_discharge",
        "flag": negative_count > 0,
        "value": negative_count,
        "flagged_timestamps": timestamps_to_strings(s.index[negative_mask]),
        "message": f"Negative discharge count = {negative_count}.",
    }


# Check for duplicate timestamps

def check_duplicate_timestamps(series: pd.Series) -> dict:
    """
    Check whether the time index contains duplicate timestamps.
    """
    s = to_datetime_sorted(series)

    duplicate_mask = s.index.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())

    return {
        "check": "duplicate_timestamps",
        "flag": duplicate_count > 0,
        "value": duplicate_count,
        "flagged_timestamps": timestamps_to_strings(s.index[duplicate_mask]),
        "message": f"Duplicate timestamp count = {duplicate_count}.",
    }


# Check for daily timestep consistency

def check_timestep_consistency(series: pd.Series) -> dict:
    """
    Check whether timestamps are consistently daily inside the valid record.
    """
    s = get_valid_record(series)

    if len(s) < 2:
        return {
            "check": "timestep_consistency",
            "flag": False,
            "value": 0,
            "flagged_timestamps": [],
            "message": "Not enough timestamps to check timestep consistency.",
        }

    diffs = s.index.to_series().diff().dropna()
    expected = pd.Timedelta(days=1)

    irregular_mask = diffs != expected
    irregular_count = int(irregular_mask.sum())

    return {
        "check": "timestep_consistency",
        "flag": irregular_count > 0,
        "value": irregular_count,
        "expected_timestep": "1 day",
        "flagged_timestamps": timestamps_to_strings(diffs.index[irregular_mask]),
        "message": f"Irregular daily timestep count = {irregular_count}.",
    }


# Basic checks runner

def run_basic_checks(series: pd.Series) -> list:
    """
    Run all structural QA/QC checks for daily discharge time series.
    """
    return [
        check_missing_values(series),
        check_long_gaps(series),
        check_negative_discharge(series),
        check_duplicate_timestamps(series),
        check_timestep_consistency(series),
    ]