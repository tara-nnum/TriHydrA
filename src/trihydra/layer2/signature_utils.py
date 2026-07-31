"""Shared preparation, coverage, event, and numerical utilities for Layer 2.

No 13-signature orchestration or OBS–ML comparison belongs in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.trihydra.layer2.signature_result import SignatureResult

__all__ = [
    "_detect_column",
    "load_discharge_csv",
    "_prepare_discharge_series",
    "_infer_timestep_days",
    "_basic_metadata",
    "_result_status",
    "_safe_divide",
    "_calculate_cv",
    "_identify_consecutive_events",
    "_event_duration_metrics",
    "_annual_coverage_table",
    "_monthly_coverage_table",
    "_lyne_hollick_single_pass",
    "_lyne_hollick_filter",
    "_richards_baker_flashiness",
    "_segment_directional_limbs",
    "_walsh_lawler_classification",
    "percentile_diagnostic",
]

def _detect_column(
    columns: Iterable[str],
    exact_candidates: Iterable[str],
    contains_candidates: Iterable[str],
    role: str,
) -> str:
    """Detect one column using exact names first and substring matches second."""
    original = list(columns)
    normalised = {str(column).lower().strip(): str(column) for column in original}

    for candidate in exact_candidates:
        if candidate in normalised:
            return normalised[candidate]

    matches = [
        str(column)
        for column in original
        if any(
            keyword in str(column).lower()
            for keyword in contains_candidates
        )
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise ValueError(
            f"No {role} column was detected. Available columns: {original}"
        )

    raise ValueError(
        f"Several possible {role} columns were detected: {matches}. "
        f"Pass the {role}_column argument explicitly."
    )

def load_discharge_csv(
    csv_path: str | Path,
    date_column: Optional[str] = None,
    discharge_column: Optional[str] = None,
    latest_years: Optional[float] = None,
    reindex_daily: bool = True,
    remove_negative: bool = False,
    series_name: str = "discharge",
    **read_csv_kwargs: Any,
) -> pd.Series:
    """
    Load one discharge series from a CSV without hard-coded station paths.

    Duplicate dates are collapsed using the median. Internal missing dates are
    retained when `reindex_daily=True`.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    frame = pd.read_csv(path, **read_csv_kwargs)

    if date_column is None:
        date_column = _detect_column(
            frame.columns,
            _DATE_EXACT_CANDIDATES,
            ("date", "time"),
            "date",
        )

    if discharge_column is None:
        discharge_column = _detect_column(
            frame.columns,
            _FLOW_EXACT_CANDIDATES,
            ("discharge", "streamflow", "flow", "observation", "obs"),
            "discharge",
        )

    missing_columns = {
        date_column,
        discharge_column,
    }.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required column(s): {sorted(missing_columns)}. "
            f"Available columns: {frame.columns.tolist()}"
        )

    frame = frame[[date_column, discharge_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[discharge_column] = pd.to_numeric(
        frame[discharge_column],
        errors="coerce",
    )
    frame = frame.dropna(subset=[date_column]).sort_values(date_column)

    series = (
        frame.groupby(date_column, sort=True)[discharge_column]
        .median()
        .rename(series_name)
    )

    if remove_negative:
        series = series.mask(series < 0)

    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()

    if first_valid is None or last_valid is None:
        return series.iloc[0:0]

    series = series.loc[first_valid:last_valid]

    if latest_years is not None:
        if latest_years <= 0:
            raise ValueError("latest_years must be positive.")
        start = last_valid - pd.DateOffset(
            days=int(round(float(latest_years) * 365.25))
        )
        series = series.loc[start:last_valid]

    if reindex_daily and not series.empty:
        full_index = pd.date_range(
            series.index.min(),
            series.index.max(),
            freq="D",
        )
        series = series.reindex(full_index)
        series.name = series_name

    return series.astype(float)


def _prepare_discharge_series(
    series: pd.Series,
    series_name: str = "discharge",
    remove_negative: bool = False,
) -> pd.Series:
    """
    Prepare a discharge series for signature calculation.

    Processing
    ----------
    - checks that input is a pandas Series,
    - converts index to DatetimeIndex,
    - sorts timestamps,
    - removes duplicate timestamps by keeping the first occurrence,
    - converts values to numeric,
    - optionally converts negative discharge to NaN,
    - keeps internal NaN values,
    - removes leading and trailing NaN-only periods.

    Parameters
    ----------
    series
        Input discharge series.

    series_name
        Name assigned to the returned Series.

    remove_negative
        If True, negative discharge values are converted to NaN.

    Returns
    -------
    pd.Series
        Prepared discharge series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input discharge data must be a pandas Series.")

    prepared = series.copy()

    try:
        prepared.index = pd.to_datetime(prepared.index)
    except Exception as exc:
        raise ValueError(
            "The discharge Series index could not be converted to datetime."
        ) from exc

    prepared = prepared.sort_index()

    if prepared.index.has_duplicates:
        prepared = prepared[~prepared.index.duplicated(keep="first")]

    prepared = pd.to_numeric(prepared, errors="coerce").astype(float)

    if remove_negative:
        prepared.loc[prepared < 0] = np.nan

    prepared = prepared.rename(series_name)

    first_valid = prepared.first_valid_index()
    last_valid = prepared.last_valid_index()

    if first_valid is None or last_valid is None:
        return prepared.iloc[0:0]

    return prepared.loc[first_valid:last_valid]

def _infer_timestep_days(series: pd.Series) -> float:
    """
    Infer the dominant timestep in days.

    Returns NaN when fewer than two timestamps are available.
    """
    if len(series.index) < 2:
        return np.nan

    differences = series.index.to_series().diff().dropna()

    if differences.empty:
        return np.nan

    dominant_timestep = differences.mode()

    if dominant_timestep.empty:
        timestep = differences.median()
    else:
        timestep = dominant_timestep.iloc[0]

    return float(timestep / pd.Timedelta(days=1))

def _basic_metadata(series: pd.Series) -> dict[str, Any]:
    """
    Generate common metadata for a discharge series.
    """
    if series.empty:
        return {
            "record_start": None,
            "record_end": None,
            "n_total": 0,
            "n_valid": 0,
            "n_missing": 0,
            "coverage_fraction": np.nan,
            "timestep_days": np.nan,
            "duration_years": np.nan,
        }

    n_total = int(len(series))
    n_valid = int(series.notna().sum())
    n_missing = int(series.isna().sum())

    if n_total == 0:
        coverage_fraction = np.nan
    else:
        coverage_fraction = n_valid / n_total

    duration_days = (
        series.index.max() - series.index.min()
    ) / pd.Timedelta(days=1)

    duration_years = (
        float(duration_days / 365.25)
        if pd.notna(duration_days)
        else np.nan
    )

    return {
        "record_start": series.index.min(),
        "record_end": series.index.max(),
        "n_total": n_total,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "coverage_fraction": float(coverage_fraction),
        "timestep_days": _infer_timestep_days(series),
        "duration_years": duration_years,
    }

def _result_status(
    valid_count: int,
    minimum_required: int,
    warnings: list[str],
) -> str:
    """
    Determine a standard result status.
    """
    if valid_count < minimum_required:
        return "insufficient_data"

    if warnings:
        return "warning"

    return "ok"

def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Safely divide two values."""
    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)

def _calculate_cv(values: pd.Series) -> float:
    """
    Calculate coefficient of variation as standard deviation divided by mean.
    """
    clean = values.dropna()

    if clean.empty:
        return np.nan

    mean_value = clean.mean()

    if mean_value == 0:
        return np.nan

    return float(clean.std(ddof=1) / mean_value)

def _identify_consecutive_events(
    condition: pd.Series,
    original_values: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Identify consecutive True periods in a Boolean time series.

    Missing condition values are treated as False and break an event.

    Parameters
    ----------
    condition
        Boolean Series with a DatetimeIndex.

    original_values
        Optional discharge Series used to calculate event statistics.

    Returns
    -------
    pd.DataFrame
        One row per event with:
        - event_id
        - start_date
        - end_date
        - duration_steps
        - duration_days
        - minimum_flow
        - maximum_flow
        - mean_flow
        - median_flow
    """
    if condition.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "start_date",
                "end_date",
                "duration_steps",
                "duration_days",
                "minimum_flow",
                "maximum_flow",
                "mean_flow",
                "median_flow",
            ]
        )

    boolean_condition = condition.fillna(False).astype(bool)

    event_start = boolean_condition & ~boolean_condition.shift(
        1,
        fill_value=False,
    )

    event_ids = event_start.cumsum()

    rows: list[dict[str, Any]] = []

    for event_id in event_ids[boolean_condition].unique():
        event_dates = boolean_condition.index[
            (event_ids == event_id) & boolean_condition
        ]

        if len(event_dates) == 0:
            continue

        start_date = event_dates.min()
        end_date = event_dates.max()
        duration_steps = int(len(event_dates))
        duration_days = int((end_date - start_date).days + 1)

        if original_values is not None:
            event_values = original_values.reindex(event_dates).dropna()
        else:
            event_values = pd.Series(dtype=float)

        rows.append(
            {
                "event_id": int(event_id),
                "start_date": start_date,
                "end_date": end_date,
                "duration_steps": duration_steps,
                "duration_days": duration_days,
                "minimum_flow": (
                    float(event_values.min())
                    if not event_values.empty
                    else np.nan
                ),
                "maximum_flow": (
                    float(event_values.max())
                    if not event_values.empty
                    else np.nan
                ),
                "mean_flow": (
                    float(event_values.mean())
                    if not event_values.empty
                    else np.nan
                ),
                "median_flow": (
                    float(event_values.median())
                    if not event_values.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)

def _event_duration_metrics(
    events: pd.DataFrame,
    duration_column: str = "duration_steps",
) -> dict[str, float]:
    """
    Calculate summary statistics for event durations.
    """
    if events.empty or duration_column not in events.columns:
        return {
            "event_count": 0,
            "mean_event_duration": np.nan,
            "median_event_duration": np.nan,
            "maximum_event_duration": np.nan,
            "minimum_event_duration": np.nan,
        }

    durations = pd.to_numeric(
        events[duration_column],
        errors="coerce",
    ).dropna()

    if durations.empty:
        return {
            "event_count": int(len(events)),
            "mean_event_duration": np.nan,
            "median_event_duration": np.nan,
            "maximum_event_duration": np.nan,
            "minimum_event_duration": np.nan,
        }

    return {
        "event_count": int(len(events)),
        "mean_event_duration": float(durations.mean()),
        "median_event_duration": float(durations.median()),
        "maximum_event_duration": float(durations.max()),
        "minimum_event_duration": float(durations.min()),
    }

def _annual_coverage_table(series: pd.Series) -> pd.DataFrame:
    """
    Calculate annual valid-data coverage.

    Coverage is based on the number of expected calendar days in each year.
    """
    if series.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "valid_day_count",
                "expected_day_count",
                "coverage_fraction",
            ]
        )

    rows: list[dict[str, Any]] = []

    for year, yearly_values in series.groupby(series.index.year):
        year = int(year)

        expected_days = (
            pd.Timestamp(year=year, month=12, day=31)
            - pd.Timestamp(year=year, month=1, day=1)
        ).days + 1

        valid_days = int(yearly_values.notna().sum())

        rows.append(
            {
                "year": year,
                "valid_day_count": valid_days,
                "expected_day_count": expected_days,
                "coverage_fraction": valid_days / expected_days,
            }
        )

    return pd.DataFrame(rows)

def _monthly_coverage_table(series: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly valid-data coverage.
    """
    if series.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "month",
                "valid_day_count",
                "expected_day_count",
                "coverage_fraction",
            ]
        )

    rows: list[dict[str, Any]] = []

    grouped = series.groupby(
        [
            series.index.year,
            series.index.month,
        ]
    )

    for (year, month), monthly_values in grouped:
        year = int(year)
        month = int(month)

        expected_days = pd.Period(
            f"{year}-{month:02d}",
            freq="M",
        ).days_in_month

        valid_days = int(monthly_values.notna().sum())

        rows.append(
            {
                "year": year,
                "month": month,
                "valid_day_count": valid_days,
                "expected_day_count": expected_days,
                "coverage_fraction": valid_days / expected_days,
            }
        )

    return pd.DataFrame(rows)

def _lyne_hollick_single_pass(
    values: np.ndarray,
    alpha: float,
    forward: bool,
) -> np.ndarray:
    """
    One pass of the Lyne-Hollick recursive digital filter.

    quickflow[i] = alpha * quickflow[previous] + ((1 + alpha) / 2) * (Q[i] - Q[previous])
    quickflow is clipped at 0 (it cannot be negative).
    baseflow = Q - quickflow, clipped to [0, Q].

    `previous` is i-1 when running forward, i+1 when running backward.
    Running forward then backward then forward again (the standard
    three-pass scheme) removes most of the phase lag a single pass
    introduces.
    """
    n = len(values)
    quickflow = np.zeros(n, dtype=float)
    factor = (1.0 + alpha) / 2.0

    order = range(1, n) if forward else range(n - 2, -1, -1)
    step = -1 if forward else 1

    for i in order:
        previous = i + step
        raw_quickflow = (
            alpha * quickflow[previous]
            + factor * (values[i] - values[previous])
        )
        quickflow[i] = raw_quickflow if raw_quickflow > 0 else 0.0

    baseflow = values - quickflow
    return np.clip(baseflow, 0.0, values)

def _lyne_hollick_filter(
    values: np.ndarray,
    alpha: float,
    passes: int,
) -> np.ndarray:
    """
    Run `passes` alternating forward/backward/forward/... passes of the
    Lyne-Hollick filter and return the final baseflow estimate.
    """
    current = values.astype(float)

    for pass_number in range(passes):
        forward = (pass_number % 2 == 0)
        current = _lyne_hollick_single_pass(
            current,
            alpha=alpha,
            forward=forward,
        )

    return current

def _richards_baker_flashiness(
    series: pd.Series,
) -> float:
    """
    Calculate the Richards-Baker Flashiness Index.

    RBI = sum(abs(Q[t] - Q[t-1])) / sum(Q[t])

    Differences are used only when both consecutive timestamps are valid, so a
    missing-data gap is not silently treated as one giant daily change.
    """
    discharge = pd.to_numeric(series, errors="coerce")

    if discharge.notna().sum() < 2:
        return np.nan

    denominator = discharge.sum(skipna=True)
    if denominator <= 0:
        return np.nan

    valid_pairs = discharge.notna() & discharge.shift(1).notna()
    numerator = (
        discharge.diff()
        .abs()
        .where(valid_pairs)
        .sum(skipna=True)
    )

    return float(numerator / denominator)

def _segment_directional_limbs(
    series: pd.Series,
    direction: str,
    tolerance: float = 0.0,
    minimum_length: int = 1,
) -> pd.DataFrame:
    """
    Segment consecutive rising or falling hydrograph limbs.

    Parameters
    ----------
    series
        Prepared discharge Series.

    direction
        "rising" or "falling".

    tolerance
        Minimum change required to classify a timestep.

        Rising:
            dQ > tolerance

        Falling:
            dQ < -tolerance

    minimum_length
        Minimum number of consecutive change steps required.

    Returns
    -------
    pd.DataFrame
        One row per limb.
    """
    if direction not in {"rising", "falling"}:
        raise ValueError(
            "direction must be either 'rising' or 'falling'."
        )

    if minimum_length < 1:
        raise ValueError(
            "minimum_length must be at least 1."
        )

    differences = series.diff()

    if direction == "rising":
        condition = differences > tolerance
    else:
        condition = differences < -tolerance

    condition = condition & series.notna() & series.shift(1).notna()

    events = _identify_consecutive_events(condition)

    rows: list[dict[str, Any]] = []

    for _, event in events.iterrows():
        change_start = pd.Timestamp(event["start_date"])
        change_end = pd.Timestamp(event["end_date"])

        start_position = series.index.get_loc(change_start)

        if isinstance(start_position, slice):
            start_position = start_position.start

        if start_position == 0:
            limb_start = change_start
        else:
            limb_start = series.index[start_position - 1]

        limb_end = change_end
        limb_values = series.loc[limb_start:limb_end].dropna()

        number_of_change_steps = int(
            event["duration_steps"]
        )

        if number_of_change_steps < minimum_length:
            continue

        if len(limb_values) < 2:
            continue

        start_flow = float(limb_values.iloc[0])
        end_flow = float(limb_values.iloc[-1])
        total_change = end_flow - start_flow

        duration_days = float(
            (limb_end - limb_start)
            / pd.Timedelta(days=1)
        )

        if duration_days <= 0:
            duration_days = float(number_of_change_steps)

        mean_rate = total_change / duration_days

        rows.append(
            {
                "limb_id": len(rows) + 1,
                "start_date": limb_start,
                "end_date": limb_end,
                "duration_steps": number_of_change_steps,
                "duration_days": duration_days,
                "start_flow": start_flow,
                "end_flow": end_flow,
                "total_change": float(total_change),
                "mean_rate": float(mean_rate),
                "maximum_flow": float(limb_values.max()),
                "minimum_flow": float(limb_values.min()),
            }
        )

    return pd.DataFrame(rows)

def _walsh_lawler_classification(seasonality_index: float) -> Optional[str]:
    """
    Classify a Walsh-Lawler Seasonality Index value using the bands
    published in Walsh and Lawler (1981).

    The index ranges from 0 (perfectly even across all 12 months) to a
    theoretical maximum of 1.83 (all volume in a single month). 1.20 is
    a classification breakpoint within that range, not the ceiling.
    """
    if pd.isna(seasonality_index):
        return None

    if seasonality_index < 0.20:
        return "very equable"
    if seasonality_index < 0.40:
        return "equable, with a well-defined wetter season"
    if seasonality_index < 0.60:
        return "rather seasonal, with a short drier season"
    if seasonality_index < 0.80:
        return "seasonal, with well-defined wet and dry seasons"
    if seasonality_index < 1.00:
        return "markedly seasonal, with a long drier season"
    if seasonality_index < 1.20:
        return "most flow occurs in three or fewer months"
    return "extreme; almost all flow occurs in one or two months"

def percentile_diagnostic(
    value: float,
    p05: float,
    p95: float,
    low_message: str,
    normal_message: str,
    high_message: str,
) -> tuple[str, bool]:
    """Compare one value with an externally supplied P05-P95 envelope."""
    if pd.isna(value) or pd.isna(p05) or pd.isna(p95):
        return "Insufficient data", False
    if value < p05:
        return low_message, True
    if value > p95:
        return high_message, True
    return normal_message, False
