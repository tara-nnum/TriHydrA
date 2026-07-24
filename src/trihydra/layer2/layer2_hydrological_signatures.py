"""
TriHydrA Layer 2: Hydrological Signatures
=========================================

Reusable discharge-only signature calculations for observed or modelled
streamflow. Functions accept pandas Series with a DatetimeIndex. The
`run_hydrological_signatures_from_csv` wrapper accepts a CSV path plus dynamic
column names or automatic column detection.

This module intentionally keeps diagnostic calculations separate from
OBS-versus-ML comparison. It does not claim that unusual behaviour is an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks
except ImportError as exc:
    raise ImportError(
        "scipy is required for peak detection. Install it using: pip install scipy"
    ) from exc



_DATE_EXACT_CANDIDATES = ("date", "datetime", "time", "timestamp")
_FLOW_EXACT_CANDIDATES = (
    "discharge",
    "streamflow",
    "flow",
    "observed_discharge",
    "observation",
    "observations",
    "obs",
    "q",
    "observed_m3s",
    "simulated_m3s",
)


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


@dataclass
class SignatureResult:
    """
    Standard output container for one signature calculation.

    Attributes
    ----------
    status
        "ok", "warning", or "insufficient_data".

    metrics
        Dictionary containing calculated scalar metrics.

    tables
        Dictionary containing supporting DataFrames or Series.

    metadata
        Information about record length, dates, timestep, and data coverage.

    warnings
        List of warning messages generated during calculation.
    """

    status: str
    metrics: dict[str, Any]
    tables: dict[str, Any]
    metadata: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary."""
        return {
            "status": self.status,
            "metrics": self.metrics,
            "tables": self.tables,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }


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


def calculate_flow_magnitude_signatures(
    series: pd.Series,
) -> SignatureResult:
    """
    Calculate basic discharge-magnitude signatures.

    Returns
    -------
    SignatureResult
        Metrics include:
        - mean_flow
        - median_flow
        - minimum_flow
        - maximum_flow
        - standard_deviation
        - coefficient_of_variation
        - q05
        - q10
        - q25
        - q75
        - q90
        - q95
    """
    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    if valid.empty:
        warnings.append("No valid discharge values are available.")

    metrics = {
        "mean_flow": (
            float(valid.mean()) if not valid.empty else np.nan
        ),
        "median_flow": (
            float(valid.median()) if not valid.empty else np.nan
        ),
        "minimum_flow": (
            float(valid.min()) if not valid.empty else np.nan
        ),
        "maximum_flow": (
            float(valid.max()) if not valid.empty else np.nan
        ),
        "standard_deviation": (
            float(valid.std(ddof=1))
            if len(valid) >= 2
            else np.nan
        ),
        "coefficient_of_variation": _calculate_cv(valid),
        "q05": (
            float(valid.quantile(0.05))
            if not valid.empty
            else np.nan
        ),
        "q10": (
            float(valid.quantile(0.10))
            if not valid.empty
            else np.nan
        ),
        "q25": (
            float(valid.quantile(0.25))
            if not valid.empty
            else np.nan
        ),
        "q75": (
            float(valid.quantile(0.75))
            if not valid.empty
            else np.nan
        ),
        "q90": (
            float(valid.quantile(0.90))
            if not valid.empty
            else np.nan
        ),
        "q95": (
            float(valid.quantile(0.95))
            if not valid.empty
            else np.nan
        ),
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=1,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={},
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )


def calculate_low_flow_signatures(
    series: pd.Series,
    threshold: Optional[float] = None,
    percentile: float = 0.05,
    include_equal: bool = True,
) -> SignatureResult:
    """
    Calculate low-flow magnitude, frequency, and duration.

    By default, the low-flow threshold is the fifth percentile of the
    valid discharge values.

    Parameters
    ----------
    series
        Discharge time series.

    threshold
        Optional externally supplied threshold. During later OBS-ML
        comparison, this can be the OBS-derived threshold.

    percentile
        Quantile used when threshold is not supplied. Default is 0.05.

    include_equal
        If True, low flow is Q <= threshold.
        If False, low flow is Q < threshold.

    Returns
    -------
    SignatureResult
        Includes scalar metrics and a low-flow event table.
    """
    if not 0 < percentile < 1:
        raise ValueError("percentile must be between 0 and 1.")

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    if valid.empty:
        warnings.append("No valid discharge values are available.")
        calculated_threshold = np.nan
    elif threshold is None:
        calculated_threshold = float(valid.quantile(percentile))
    else:
        calculated_threshold = float(threshold)

    if pd.isna(calculated_threshold):
        low_condition = pd.Series(
            False,
            index=discharge.index,
            dtype=bool,
        )
    elif include_equal:
        low_condition = discharge <= calculated_threshold
    else:
        low_condition = discharge < calculated_threshold

    low_condition = low_condition & discharge.notna()

    events = _identify_consecutive_events(
        condition=low_condition,
        original_values=discharge,
    )

    duration_metrics = _event_duration_metrics(events)

    low_flow_days = int(low_condition.sum())
    valid_days = int(discharge.notna().sum())

    annual_rows: list[dict[str, Any]] = []

    for year, yearly_values in discharge.groupby(discharge.index.year):
        yearly_condition = low_condition.reindex(yearly_values.index)

        year_valid_days = int(yearly_values.notna().sum())
        year_low_days = int(yearly_condition.sum())

        annual_rows.append(
            {
                "year": int(year),
                "valid_day_count": year_valid_days,
                "low_flow_days": year_low_days,
                "low_flow_frequency": _safe_divide(
                    year_low_days,
                    year_valid_days,
                ),
            }
        )

    annual_frequency = pd.DataFrame(annual_rows)

    metrics = {
        "low_flow_threshold": calculated_threshold,
        "threshold_percentile": float(percentile),
        "low_flow_days": low_flow_days,
        "low_flow_frequency": _safe_divide(
            low_flow_days,
            valid_days,
        ),
        "low_flow_event_count": duration_metrics["event_count"],
        "mean_low_flow_duration": duration_metrics[
            "mean_event_duration"
        ],
        "median_low_flow_duration": duration_metrics[
            "median_event_duration"
        ],
        "maximum_low_flow_duration": duration_metrics[
            "maximum_event_duration"
        ],
        "minimum_low_flow_duration": duration_metrics[
            "minimum_event_duration"
        ],
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=10,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "low_flow_events": events,
            "annual_low_flow_frequency": annual_frequency,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )


def calculate_high_flow_signatures(
    series: pd.Series,
    threshold: Optional[float] = None,
    percentile: float = 0.95,
    include_equal: bool = True,
) -> SignatureResult:
    """
    Calculate high-flow magnitude, frequency, and duration.

    By default, the high-flow threshold is the 95th percentile of the
    valid discharge values.

    Parameters
    ----------
    series
        Discharge time series.

    threshold
        Optional externally supplied threshold.

    percentile
        Quantile used when threshold is not supplied. Default is 0.95.

    include_equal
        If True, high flow is Q >= threshold.
        If False, high flow is Q > threshold.
    """
    if not 0 < percentile < 1:
        raise ValueError("percentile must be between 0 and 1.")

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    if valid.empty:
        warnings.append("No valid discharge values are available.")
        calculated_threshold = np.nan
    elif threshold is None:
        calculated_threshold = float(valid.quantile(percentile))
    else:
        calculated_threshold = float(threshold)

    if pd.isna(calculated_threshold):
        high_condition = pd.Series(
            False,
            index=discharge.index,
            dtype=bool,
        )
    elif include_equal:
        high_condition = discharge >= calculated_threshold
    else:
        high_condition = discharge > calculated_threshold

    high_condition = high_condition & discharge.notna()

    events = _identify_consecutive_events(
        condition=high_condition,
        original_values=discharge,
    )

    duration_metrics = _event_duration_metrics(events)

    high_flow_days = int(high_condition.sum())
    valid_days = int(discharge.notna().sum())

    annual_rows: list[dict[str, Any]] = []

    for year, yearly_values in discharge.groupby(discharge.index.year):
        yearly_condition = high_condition.reindex(yearly_values.index)

        year_valid_days = int(yearly_values.notna().sum())
        year_high_days = int(yearly_condition.sum())

        annual_rows.append(
            {
                "year": int(year),
                "valid_day_count": year_valid_days,
                "high_flow_days": year_high_days,
                "high_flow_frequency": _safe_divide(
                    year_high_days,
                    year_valid_days,
                ),
            }
        )

    annual_frequency = pd.DataFrame(annual_rows)

    metrics = {
        "high_flow_threshold": calculated_threshold,
        "threshold_percentile": float(percentile),
        "high_flow_days": high_flow_days,
        "high_flow_frequency": _safe_divide(
            high_flow_days,
            valid_days,
        ),
        "high_flow_event_count": duration_metrics["event_count"],
        "mean_high_flow_duration": duration_metrics[
            "mean_event_duration"
        ],
        "median_high_flow_duration": duration_metrics[
            "median_event_duration"
        ],
        "maximum_high_flow_duration": duration_metrics[
            "maximum_event_duration"
        ],
        "minimum_high_flow_duration": duration_metrics[
            "minimum_event_duration"
        ],
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=10,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "high_flow_events": events,
            "annual_high_flow_frequency": annual_frequency,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )


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


def calculate_zero_flow_signatures(
    series: pd.Series,
    zero_threshold: float = 1e-6,
) -> SignatureResult:
    """
    Calculate zero-flow frequency and persistence.

    A zero-flow day is defined as Q <= zero_threshold.

    Parameters
    ----------
    series
        Discharge time series.

    zero_threshold
        Numerical threshold used to identify zero or near-zero flow.
    """
    if zero_threshold < 0:
        raise ValueError("zero_threshold cannot be negative.")

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    if valid.empty:
        warnings.append("No valid discharge values are available.")

    zero_condition = (
        (discharge <= zero_threshold)
        & discharge.notna()
    )

    events = _identify_consecutive_events(
        condition=zero_condition,
        original_values=discharge,
    )

    duration_metrics = _event_duration_metrics(events)

    zero_days = int(zero_condition.sum())
    valid_days = int(discharge.notna().sum())

    annual_rows: list[dict[str, Any]] = []

    for year, yearly_values in discharge.groupby(discharge.index.year):
        yearly_zero = zero_condition.reindex(yearly_values.index)

        n_valid = int(yearly_values.notna().sum())
        n_zero = int(yearly_zero.sum())

        annual_rows.append(
            {
                "year": int(year),
                "valid_day_count": n_valid,
                "zero_flow_days": n_zero,
                "zero_flow_frequency": _safe_divide(
                    n_zero,
                    n_valid,
                ),
            }
        )

    monthly_rows: list[dict[str, Any]] = []

    grouped_monthly = discharge.groupby(
        [
            discharge.index.year,
            discharge.index.month,
        ]
    )

    for (year, month), monthly_values in grouped_monthly:
        monthly_zero = zero_condition.reindex(
            monthly_values.index
        )

        n_valid = int(monthly_values.notna().sum())
        n_zero = int(monthly_zero.sum())

        monthly_rows.append(
            {
                "year": int(year),
                "month": int(month),
                "valid_day_count": n_valid,
                "zero_flow_days": n_zero,
                "zero_flow_frequency": _safe_divide(
                    n_zero,
                    n_valid,
                ),
            }
        )

    annual_frequency = pd.DataFrame(annual_rows)
    monthly_frequency = pd.DataFrame(monthly_rows)

    if monthly_frequency.empty:
        monthly_climatology = pd.DataFrame(
            columns=[
                "month",
                "median_zero_flow_frequency",
                "mean_zero_flow_frequency",
                "maximum_zero_flow_frequency",
            ]
        )
    else:
        monthly_climatology = (
            monthly_frequency
            .groupby("month", as_index=False)
            .agg(
                median_zero_flow_frequency=(
                    "zero_flow_frequency",
                    "median",
                ),
                mean_zero_flow_frequency=(
                    "zero_flow_frequency",
                    "mean",
                ),
                maximum_zero_flow_frequency=(
                    "zero_flow_frequency",
                    "max",
                ),
            )
        )

    metrics = {
        "zero_threshold": float(zero_threshold),
        "zero_flow_days": zero_days,
        "zero_flow_ratio": _safe_divide(
            zero_days,
            valid_days,
        ),
        "zero_flow_event_count": duration_metrics["event_count"],
        "mean_zero_flow_duration": duration_metrics[
            "mean_event_duration"
        ],
        "median_zero_flow_duration": duration_metrics[
            "median_event_duration"
        ],
        "maximum_zero_flow_duration": duration_metrics[
            "maximum_event_duration"
        ],
        "minimum_zero_flow_duration": duration_metrics[
            "minimum_event_duration"
        ],
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=1,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "zero_flow_events": events,
            "annual_zero_flow_frequency": annual_frequency,
            "monthly_zero_flow_frequency": monthly_frequency,
            "monthly_zero_flow_climatology": monthly_climatology,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )


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


def calculate_autocorrelation_signatures(
    series: pd.Series,
    lags: Iterable[int] = (1, 2, 3, 7, 14, 30),
    maximum_decay_lag: int = 90,
) -> SignatureResult:
    """
    Calculate discharge autocorrelation and persistence metrics.

    Parameters
    ----------
    series
        Discharge time series.

    lags
        Specific lags for which autocorrelation should be returned.

    maximum_decay_lag
        Maximum lag examined when estimating decorrelation time.

    Returns
    -------
    SignatureResult
        Metrics include:
        - autocorrelation at requested lags,
        - decorrelation lag,
        - first non-positive autocorrelation lag,
        - integral correlation time,
        - persistence ratio AC7 / AC1.

    Notes
    -----
    The decorrelation lag is the first lag where autocorrelation is
    less than or equal to 1/e.

    Integral correlation time is calculated using the initial positive
    sequence:

        1 + 2 * sum(ACF[k])

    until the first non-positive autocorrelation.
    """
    requested_lags = sorted(
        {
            int(lag)
            for lag in lags
            if int(lag) > 0
        }
    )

    if maximum_decay_lag < 1:
        raise ValueError(
            "maximum_decay_lag must be at least 1."
        )

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    if len(valid) < 3:
        warnings.append(
            "At least three valid values are required for "
            "autocorrelation calculation."
        )

    all_lags = sorted(
        set(
            requested_lags
            + list(range(1, maximum_decay_lag + 1))
        )
    )

    acf_rows: list[dict[str, Any]] = []

    for lag in all_lags:
        if len(discharge.dropna()) <= lag:
            acf = np.nan
            valid_pair_count = 0
        else:
            current = discharge
            lagged = discharge.shift(lag)

            paired = pd.concat(
                [
                    current.rename("current"),
                    lagged.rename("lagged"),
                ],
                axis=1,
            ).dropna()

            valid_pair_count = int(len(paired))

            if valid_pair_count < 3:
                acf = np.nan
            elif (
                paired["current"].std(ddof=1) == 0
                or paired["lagged"].std(ddof=1) == 0
            ):
                acf = np.nan
            else:
                acf = float(
                    paired["current"].corr(
                        paired["lagged"]
                    )
                )

        acf_rows.append(
            {
                "lag": lag,
                "autocorrelation": acf,
                "valid_pair_count": valid_pair_count,
            }
        )

    acf_table = pd.DataFrame(acf_rows)

    requested_acf = {
        f"autocorrelation_lag_{lag}": (
            float(
                acf_table.loc[
                    acf_table["lag"] == lag,
                    "autocorrelation",
                ].iloc[0]
            )
            if (
                not acf_table.loc[
                    acf_table["lag"] == lag,
                    "autocorrelation",
                ].empty
                and pd.notna(
                    acf_table.loc[
                        acf_table["lag"] == lag,
                        "autocorrelation",
                    ].iloc[0]
                )
            )
            else np.nan
        )
        for lag in requested_lags
    }

    decay_table = acf_table.loc[
        acf_table["lag"] <= maximum_decay_lag
    ].copy()

    one_over_e = 1 / np.e

    decorrelation_candidates = decay_table.loc[
        decay_table["autocorrelation"] <= one_over_e,
        "lag",
    ]

    if decorrelation_candidates.empty:
        decorrelation_lag = np.nan
    else:
        decorrelation_lag = int(
            decorrelation_candidates.iloc[0]
        )

    non_positive_candidates = decay_table.loc[
        decay_table["autocorrelation"] <= 0,
        "lag",
    ]

    if non_positive_candidates.empty:
        first_non_positive_lag = np.nan
    else:
        first_non_positive_lag = int(
            non_positive_candidates.iloc[0]
        )

    positive_acf_values: list[float] = []

    for _, row in decay_table.sort_values("lag").iterrows():
        acf_value = row["autocorrelation"]

        if pd.isna(acf_value):
            break

        if acf_value <= 0:
            break

        positive_acf_values.append(float(acf_value))

    integral_correlation_time = (
        float(1 + 2 * np.sum(positive_acf_values))
        if positive_acf_values
        else np.nan
    )

    ac1 = requested_acf.get(
        "autocorrelation_lag_1",
        np.nan,
    )
    ac7 = requested_acf.get(
        "autocorrelation_lag_7",
        np.nan,
    )

    metrics = {
        **requested_acf,
        "decorrelation_threshold": float(one_over_e),
        "decorrelation_lag": decorrelation_lag,
        "first_non_positive_autocorrelation_lag": (
            first_non_positive_lag
        ),
        "integral_correlation_time": (
            integral_correlation_time
        ),
        "acf_7_to_acf_1_ratio": _safe_divide(
            ac7,
            ac1,
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
            "autocorrelation_function": acf_table,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )


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


def calculate_rising_limb_signatures(
    series: pd.Series,
    tolerance: float = 0.0,
    minimum_limb_length: int = 1,
) -> SignatureResult:
    """
    Calculate rising-limb signatures.

    Includes:
    - fraction of valid change steps that are rising,
    - mean and median positive dQ/dt,
    - P90 rising rate,
    - maximum rising rate,
    - normalised rising rates,
    - number of rising limbs,
    - rising-limb density,
    - rising-limb durations.

    Rising-limb density is calculated as:

        number of rising limbs / total rising timesteps

    This is equivalent to the inverse of mean rising-limb length.
    """
    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    differences = discharge.diff()

    valid_change = (
        discharge.notna()
        & discharge.shift(1).notna()
    )

    rising_condition = (
        differences > tolerance
    ) & valid_change

    rising_rates = differences.loc[rising_condition].dropna()

    limbs = _segment_directional_limbs(
        series=discharge,
        direction="rising",
        tolerance=tolerance,
        minimum_length=minimum_limb_length,
    )

    valid_change_count = int(valid_change.sum())
    rising_step_count = int(rising_condition.sum())
    rising_limb_count = int(len(limbs))

    if rising_step_count == 0:
        rising_limb_density = np.nan
    else:
        rising_limb_density = (
            rising_limb_count / rising_step_count
        )

    median_flow = valid.median() if not valid.empty else np.nan

    normalised_rates = (
        rising_rates / median_flow
        if pd.notna(median_flow) and median_flow != 0
        else pd.Series(dtype=float)
    )

    metrics = {
        "rising_tolerance": float(tolerance),
        "minimum_limb_length": int(minimum_limb_length),
        "valid_change_step_count": valid_change_count,
        "rising_step_count": rising_step_count,
        "rising_day_fraction": _safe_divide(
            rising_step_count,
            valid_change_count,
        ),
        "mean_rising_rate": (
            float(rising_rates.mean())
            if not rising_rates.empty
            else np.nan
        ),
        "median_rising_rate": (
            float(rising_rates.median())
            if not rising_rates.empty
            else np.nan
        ),
        "p90_rising_rate": (
            float(rising_rates.quantile(0.90))
            if not rising_rates.empty
            else np.nan
        ),
        "maximum_rising_rate": (
            float(rising_rates.max())
            if not rising_rates.empty
            else np.nan
        ),
        "mean_normalised_rising_rate": (
            float(normalised_rates.mean())
            if not normalised_rates.empty
            else np.nan
        ),
        "median_normalised_rising_rate": (
            float(normalised_rates.median())
            if not normalised_rates.empty
            else np.nan
        ),
        "rising_limb_count": rising_limb_count,
        "rising_limb_density": float(rising_limb_density),
        "mean_rising_limb_duration": (
            float(limbs["duration_steps"].mean())
            if not limbs.empty
            else np.nan
        ),
        "median_rising_limb_duration": (
            float(limbs["duration_steps"].median())
            if not limbs.empty
            else np.nan
        ),
        "maximum_rising_limb_duration": (
            float(limbs["duration_steps"].max())
            if not limbs.empty
            else np.nan
        ),
        "median_rising_limb_total_change": (
            float(limbs["total_change"].median())
            if not limbs.empty
            else np.nan
        ),
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
            "rising_limbs": limbs,
            "rising_rates": rising_rates.rename(
                "rising_rate"
            ).to_frame(),
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )


def calculate_recession_limb_signatures(
    series: pd.Series,
    tolerance: float = 0.0,
    minimum_limb_length: int = 1,
) -> SignatureResult:
    """
    Calculate recession-limb signatures.

    Includes ordinary discharge-decrease rates and log-flow recession
    rates.

    Log recession rate is calculated for positive consecutive flows as:

        log(Q[t]) - log(Q[t-1])

    Negative values indicate recession.
    """
    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    differences = discharge.diff()

    valid_change = (
        discharge.notna()
        & discharge.shift(1).notna()
    )

    recession_condition = (
        differences < -tolerance
    ) & valid_change

    recession_rates = differences.loc[
        recession_condition
    ].dropna()

    positive_pair = (
        (discharge > 0)
        & (discharge.shift(1) > 0)
        & valid_change
    )

    log_difference = (
        np.log(discharge)
        - np.log(discharge.shift(1))
    )

    log_recession_rates = log_difference.loc[
        recession_condition & positive_pair
    ].dropna()

    limbs = _segment_directional_limbs(
        series=discharge,
        direction="falling",
        tolerance=tolerance,
        minimum_length=minimum_limb_length,
    )

    valid_change_count = int(valid_change.sum())
    recession_step_count = int(recession_condition.sum())

    median_flow = valid.median() if not valid.empty else np.nan

    normalised_rates = (
        recession_rates / median_flow
        if pd.notna(median_flow) and median_flow != 0
        else pd.Series(dtype=float)
    )

    metrics = {
        "recession_tolerance": float(tolerance),
        "minimum_limb_length": int(minimum_limb_length),
        "valid_change_step_count": valid_change_count,
        "recession_step_count": recession_step_count,
        "recession_day_fraction": _safe_divide(
            recession_step_count,
            valid_change_count,
        ),
        "mean_recession_rate": (
            float(recession_rates.mean())
            if not recession_rates.empty
            else np.nan
        ),
        "median_recession_rate": (
            float(recession_rates.median())
            if not recession_rates.empty
            else np.nan
        ),
        "p10_recession_rate": (
            float(recession_rates.quantile(0.10))
            if not recession_rates.empty
            else np.nan
        ),
        "minimum_recession_rate": (
            float(recession_rates.min())
            if not recession_rates.empty
            else np.nan
        ),
        "mean_normalised_recession_rate": (
            float(normalised_rates.mean())
            if not normalised_rates.empty
            else np.nan
        ),
        "median_normalised_recession_rate": (
            float(normalised_rates.median())
            if not normalised_rates.empty
            else np.nan
        ),
        "mean_log_recession_rate": (
            float(log_recession_rates.mean())
            if not log_recession_rates.empty
            else np.nan
        ),
        "median_log_recession_rate": (
            float(log_recession_rates.median())
            if not log_recession_rates.empty
            else np.nan
        ),
        "recession_limb_count": int(len(limbs)),
        "mean_recession_limb_duration": (
            float(limbs["duration_steps"].mean())
            if not limbs.empty
            else np.nan
        ),
        "median_recession_limb_duration": (
            float(limbs["duration_steps"].median())
            if not limbs.empty
            else np.nan
        ),
        "maximum_recession_limb_duration": (
            float(limbs["duration_steps"].max())
            if not limbs.empty
            else np.nan
        ),
        "median_recession_limb_total_change": (
            float(limbs["total_change"].median())
            if not limbs.empty
            else np.nan
        ),
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
            "recession_limbs": limbs,
            "recession_rates": recession_rates.rename(
                "recession_rate"
            ).to_frame(),
            "log_recession_rates": log_recession_rates.rename(
                "log_recession_rate"
            ).to_frame(),
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )


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


def calculate_threshold_event_hydrographs(
    series: pd.Series,
    threshold: Optional[float] = None,
    percentile: float = 0.95,
    include_equal: bool = True,
) -> SignatureResult:
    """
    Calculate the event-hydrograph metrics used in the Layer 2 notebook.

    An event is one consecutive spell above a fixed threshold, Q95 by default.
    The event start/end are therefore threshold crossings, not complete
    catchment-response boundaries.
    """
    if not 0 < percentile < 1:
        raise ValueError("percentile must be between 0 and 1.")

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()
    warnings: list[str] = []

    if valid.empty:
        calculated_threshold = np.nan
        warnings.append("No valid discharge values are available.")
    elif threshold is None:
        calculated_threshold = float(valid.quantile(percentile))
    else:
        calculated_threshold = float(threshold)

    if pd.isna(calculated_threshold):
        condition = pd.Series(False, index=discharge.index, dtype=bool)
    elif include_equal:
        condition = discharge.ge(calculated_threshold) & discharge.notna()
    else:
        condition = discharge.gt(calculated_threshold) & discharge.notna()

    event_start = condition & ~condition.shift(1, fill_value=False)
    event_ids = event_start.cumsum()
    rows: list[dict[str, Any]] = []

    for current_event_id in event_ids[condition].unique():
        dates = discharge.index[(event_ids == current_event_id) & condition]
        if len(dates) == 0:
            continue

        start_date = dates.min()
        end_date = dates.max()
        event_flow = discharge.loc[start_date:end_date].dropna()
        if event_flow.empty:
            continue

        peak_date = event_flow.idxmax()
        peak_flow = float(event_flow.max())

        rising_flow = discharge.loc[start_date:peak_date].dropna()
        rising_changes = rising_flow.diff().dropna()
        positive_rises = rising_changes[rising_changes > 0]

        time_to_peak_days = float(
            (peak_date - start_date) / pd.Timedelta(days=1)
        )
        overall_rising_slope = (
            (peak_flow - float(rising_flow.iloc[0])) / time_to_peak_days
            if time_to_peak_days > 0
            else np.nan
        )

        recession_flow = discharge.loc[peak_date:end_date].dropna()
        recession_changes = recession_flow.diff().dropna()
        negative_recessions = recession_changes[recession_changes < 0]

        recession_duration_days = float(
            (end_date - peak_date) / pd.Timedelta(days=1)
        )
        overall_recession_slope = (
            (float(recession_flow.iloc[-1]) - peak_flow)
            / recession_duration_days
            if recession_duration_days > 0
            else np.nan
        )

        positive_recession_flow = recession_flow[recession_flow > 0]
        if len(positive_recession_flow) >= 2:
            elapsed_days = (
                positive_recession_flow.index
                - positive_recession_flow.index[0]
            ) / pd.Timedelta(days=1)
            log_recession_slope = float(
                np.polyfit(
                    elapsed_days,
                    np.log(positive_recession_flow.to_numpy()),
                    1,
                )[0]
            )
        else:
            log_recession_slope = np.nan

        rows.append(
            {
                "event_id": int(current_event_id),
                "event_start": start_date,
                "peak_date": peak_date,
                "event_end": end_date,
                "event_duration_days": int((end_date - start_date).days + 1),
                "start_flow": float(event_flow.iloc[0]),
                "peak_flow": peak_flow,
                "end_flow": float(event_flow.iloc[-1]),
                "time_to_peak_days": time_to_peak_days,
                "mean_positive_rising_rate": (
                    float(positive_rises.mean())
                    if not positive_rises.empty else np.nan
                ),
                "median_positive_rising_rate": (
                    float(positive_rises.median())
                    if not positive_rises.empty else np.nan
                ),
                "maximum_positive_rising_rate": (
                    float(positive_rises.max())
                    if not positive_rises.empty else np.nan
                ),
                "overall_rising_slope": float(overall_rising_slope)
                if pd.notna(overall_rising_slope) else np.nan,
                "mean_recession_slope": (
                    float(negative_recessions.mean())
                    if not negative_recessions.empty else np.nan
                ),
                "median_recession_slope": (
                    float(negative_recessions.median())
                    if not negative_recessions.empty else np.nan
                ),
                "steepest_recession_slope": (
                    float(negative_recessions.min())
                    if not negative_recessions.empty else np.nan
                ),
                "overall_recession_slope": float(overall_recession_slope)
                if pd.notna(overall_recession_slope) else np.nan,
                "log_recession_slope": log_recession_slope,
            }
        )

    events = pd.DataFrame(rows)

    def _metric(column: str, operation: str = "median") -> float:
        if events.empty or column not in events:
            return np.nan
        values = pd.to_numeric(events[column], errors="coerce").dropna()
        if values.empty:
            return np.nan
        return float(getattr(values, operation)())

    duration_years = _basic_metadata(discharge)["duration_years"]

    metrics = {
        "event_threshold": calculated_threshold,
        "threshold_percentile": float(percentile),
        "event_count": int(len(events)),
        "event_frequency_per_year": _safe_divide(len(events), duration_years),
        "median_event_duration_days": _metric("event_duration_days"),
        "median_peak_flow": _metric("peak_flow"),
        "median_rising_limb_rate": _metric("median_positive_rising_rate"),
        "median_overall_rising_slope": _metric("overall_rising_slope"),
        "median_recession_limb_slope": _metric("median_recession_slope"),
        "median_overall_recession_slope": _metric("overall_recession_slope"),
        "median_log_recession_slope": _metric("log_recession_slope"),
        "median_time_to_peak_days": _metric("time_to_peak_days"),
        "maximum_time_to_peak_days": _metric("time_to_peak_days", "max"),
    }

    if events.empty:
        warnings.append("No threshold-exceedance events were detected.")

    return SignatureResult(
        status=_result_status(
            valid_count=len(valid),
            minimum_required=3,
            warnings=warnings,
        ),
        metrics=metrics,
        tables={"events": events},
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )


def calculate_derivative_event_hydrographs(
    series: pd.Series,
    smoothing_window_days: int = 5,
    min_consecutive_rising_days: int = 2,
) -> SignatureResult:
    """
    Separate real event hydrographs using the SIGN of the day-to-day
    change (dQ), instead of a fixed percentile threshold.

    A fixed threshold (see calculate_threshold_event_hydrographs) misses
    real floods in dry years and flags ordinary baseflow variation in
    wet years, because it judges events by MAGNITUDE. This instead
    judges events by VELOCITY: a rolling median smooths out day-to-day
    noise, and an event starts the moment the smoothed series turns and
    keeps rising for at least min_consecutive_rising_days, then runs
    through its peak and recedes until the next genuine rise begins.
    Smoothing is used ONLY to decide where an event starts and ends;
    peak magnitude and recession rates are read back off the RAW
    (unsmoothed) discharge, so a real event's reported magnitude is
    never blunted by the smoothing.

    A second rain pulse partway through a long recession will register
    as a separate event by this method, since each new qualifying rise
    restarts the state machine -- a deliberate, cheap tradeoff, not a
    bug, and worth knowing before reading "event_count" as "flood count".
    """
    if smoothing_window_days < 1:
        raise ValueError("smoothing_window_days must be at least 1.")
    if min_consecutive_rising_days < 1:
        raise ValueError("min_consecutive_rising_days must be at least 1.")

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()
    warnings: list[str] = []

    empty_metrics = {
        "smoothing_window_days": int(smoothing_window_days),
        "min_consecutive_rising_days": int(min_consecutive_rising_days),
        "event_count": 0,
        "event_frequency_per_year": np.nan,
        "median_event_duration_days": np.nan,
        "median_peak_flow": np.nan,
        "median_rising_limb_rate": np.nan,
        "median_overall_rising_slope": np.nan,
        "median_recession_limb_slope": np.nan,
        "median_overall_recession_slope": np.nan,
        "median_log_recession_slope": np.nan,
        "median_time_to_peak_days": np.nan,
        "maximum_time_to_peak_days": np.nan,
    }

    if len(valid) < 3:
        warnings.append(
            "Not enough valid discharge values for event separation."
        )
        return SignatureResult(
            status=_result_status(
                valid_count=len(valid),
                minimum_required=3,
                warnings=warnings,
            ),
            metrics=empty_metrics,
            tables={"events": pd.DataFrame()},
            metadata=_basic_metadata(discharge),
            warnings=warnings,
        )

    smoothed = discharge.rolling(
        window=smoothing_window_days,
        center=True,
        min_periods=max(1, smoothing_window_days // 2 + 1),
    ).median()

    smoothed_diff = smoothed.diff()
    rising_condition = (smoothed_diff > 0) & smoothed_diff.notna()

    rising_runs = _identify_consecutive_events(rising_condition)
    genuine_rises = (
        rising_runs[
            rising_runs["duration_steps"] >= min_consecutive_rising_days
        ]
        .sort_values("start_date")
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    n = len(discharge)

    for i, rise in genuine_rises.iterrows():
        rise_start_position = discharge.index.get_loc(rise["start_date"])
        if isinstance(rise_start_position, slice):
            rise_start_position = rise_start_position.start

        event_start_position = max(rise_start_position - 1, 0)
        event_start = discharge.index[event_start_position]

        if i + 1 < len(genuine_rises):
            next_rise_start_position = discharge.index.get_loc(
                genuine_rises.loc[i + 1, "start_date"]
            )
            if isinstance(next_rise_start_position, slice):
                next_rise_start_position = next_rise_start_position.start
            event_end_position = max(
                next_rise_start_position - 1,
                event_start_position,
            )
        else:
            event_end_position = n - 1

        event_end = discharge.index[event_end_position]

        event_flow = discharge.loc[event_start:event_end].dropna()
        if event_flow.empty or len(event_flow) < 2:
            continue

        peak_date = event_flow.idxmax()
        peak_flow = float(event_flow.max())

        rising_flow = discharge.loc[event_start:peak_date].dropna()
        rising_changes = rising_flow.diff().dropna()
        positive_rises = rising_changes[rising_changes > 0]

        time_to_peak_days = float(
            (peak_date - event_start) / pd.Timedelta(days=1)
        )
        overall_rising_slope = (
            (peak_flow - float(rising_flow.iloc[0])) / time_to_peak_days
            if time_to_peak_days > 0 else np.nan
        )

        recession_flow = discharge.loc[peak_date:event_end].dropna()
        recession_changes = recession_flow.diff().dropna()
        negative_recessions = recession_changes[recession_changes < 0]

        recession_duration_days = float(
            (event_end - peak_date) / pd.Timedelta(days=1)
        )
        overall_recession_slope = (
            (float(recession_flow.iloc[-1]) - peak_flow)
            / recession_duration_days
            if recession_duration_days > 0 else np.nan
        )

        positive_recession_flow = recession_flow[recession_flow > 0]
        if len(positive_recession_flow) >= 2:
            elapsed_days = (
                positive_recession_flow.index
                - positive_recession_flow.index[0]
            ) / pd.Timedelta(days=1)
            log_recession_slope = float(
                np.polyfit(
                    elapsed_days,
                    np.log(positive_recession_flow.to_numpy()),
                    1,
                )[0]
            )
        else:
            log_recession_slope = np.nan

        rows.append(
            {
                "event_id": len(rows) + 1,
                "event_start": event_start,
                "peak_date": peak_date,
                "event_end": event_end,
                "event_duration_days": int(
                    (event_end - event_start).days + 1
                ),
                "start_flow": float(event_flow.iloc[0]),
                "peak_flow": peak_flow,
                "end_flow": float(event_flow.iloc[-1]),
                "time_to_peak_days": time_to_peak_days,
                "mean_positive_rising_rate": (
                    float(positive_rises.mean())
                    if not positive_rises.empty else np.nan
                ),
                "median_positive_rising_rate": (
                    float(positive_rises.median())
                    if not positive_rises.empty else np.nan
                ),
                "maximum_positive_rising_rate": (
                    float(positive_rises.max())
                    if not positive_rises.empty else np.nan
                ),
                "overall_rising_slope": float(overall_rising_slope)
                if pd.notna(overall_rising_slope) else np.nan,
                "mean_recession_slope": (
                    float(negative_recessions.mean())
                    if not negative_recessions.empty else np.nan
                ),
                "median_recession_slope": (
                    float(negative_recessions.median())
                    if not negative_recessions.empty else np.nan
                ),
                "steepest_recession_slope": (
                    float(negative_recessions.min())
                    if not negative_recessions.empty else np.nan
                ),
                "overall_recession_slope": float(overall_recession_slope)
                if pd.notna(overall_recession_slope) else np.nan,
                "log_recession_slope": log_recession_slope,
            }
        )

    events = pd.DataFrame(rows)

    def _metric(column: str, operation: str = "median") -> float:
        if events.empty or column not in events:
            return np.nan
        values = pd.to_numeric(events[column], errors="coerce").dropna()
        if values.empty:
            return np.nan
        return float(getattr(values, operation)())

    duration_years = _basic_metadata(discharge)["duration_years"]

    metrics = {
        "smoothing_window_days": int(smoothing_window_days),
        "min_consecutive_rising_days": int(min_consecutive_rising_days),
        "event_count": int(len(events)),
        "event_frequency_per_year": _safe_divide(len(events), duration_years),
        "median_event_duration_days": _metric("event_duration_days"),
        "median_peak_flow": _metric("peak_flow"),
        "median_rising_limb_rate": _metric("median_positive_rising_rate"),
        "median_overall_rising_slope": _metric("overall_rising_slope"),
        "median_recession_limb_slope": _metric("median_recession_slope"),
        "median_overall_recession_slope": _metric("overall_recession_slope"),
        "median_log_recession_slope": _metric("log_recession_slope"),
        "median_time_to_peak_days": _metric("time_to_peak_days"),
        "maximum_time_to_peak_days": _metric("time_to_peak_days", "max"),
    }

    if events.empty:
        warnings.append("No rising-then-falling events were detected.")

    return SignatureResult(
        status=_result_status(
            valid_count=len(valid),
            minimum_required=3,
            warnings=warnings,
        ),
        metrics=metrics,
        tables={"events": events},
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )


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


def calculate_flashiness_persistence_analysis(
    series: pd.Series,
    minimum_year_coverage: float = 0.80,
    rolling_window_days: Optional[int] = 90,
    rolling_minimum_valid_days: int = 70,
) -> SignatureResult:
    """
    Combine annual/rolling Richards-Baker flashiness and lag-1 persistence.

    Rolling calculations use pandas vectorised rolling operations rather than
    one Python loop per date.
    """
    if not 0 <= minimum_year_coverage <= 1:
        raise ValueError("minimum_year_coverage must be between 0 and 1.")
    if rolling_window_days is not None and rolling_window_days < 2:
        raise ValueError("rolling_window_days must be at least 2.")
    if rolling_minimum_valid_days < 2:
        raise ValueError("rolling_minimum_valid_days must be at least 2.")

    discharge = _prepare_discharge_series(series)
    warnings: list[str] = []

    whole_flashiness = _richards_baker_flashiness(discharge)

    pairs = pd.concat(
        [
            discharge.rename("current"),
            discharge.shift(1).rename("previous"),
        ],
        axis=1,
    ).dropna()
    whole_ac1 = (
        float(pairs["current"].corr(pairs["previous"]))
        if len(pairs) >= 3
        and pairs["current"].std(ddof=1) > 0
        and pairs["previous"].std(ddof=1) > 0
        else np.nan
    )

    coverage = _annual_coverage_table(discharge)
    annual_rows: list[dict[str, Any]] = []

    for year, yearly in discharge.groupby(discharge.index.year):
        coverage_row = coverage.loc[coverage["year"] == int(year)]
        coverage_fraction = (
            float(coverage_row["coverage_fraction"].iloc[0])
            if not coverage_row.empty else np.nan
        )
        retained = (
            pd.notna(coverage_fraction)
            and coverage_fraction >= minimum_year_coverage
        )

        yearly_pairs = pd.concat(
            [
                yearly.rename("current"),
                yearly.shift(1).rename("previous"),
            ],
            axis=1,
        ).dropna()

        ac1 = (
            float(yearly_pairs["current"].corr(yearly_pairs["previous"]))
            if retained
            and len(yearly_pairs) >= 3
            and yearly_pairs["current"].std(ddof=1) > 0
            and yearly_pairs["previous"].std(ddof=1) > 0
            else np.nan
        )

        annual_rows.append(
            {
                "year": int(year),
                "flashiness_index": (
                    _richards_baker_flashiness(yearly)
                    if retained else np.nan
                ),
                "lag1_autocorrelation": ac1,
                "coverage_fraction": coverage_fraction,
                "retained": retained,
            }
        )

    annual = pd.DataFrame(annual_rows)
    valid_annual = annual.dropna(
        subset=["flashiness_index", "lag1_autocorrelation"]
    )

    annual_correlation = (
        float(
            valid_annual["flashiness_index"].corr(
                valid_annual["lag1_autocorrelation"]
            )
        )
        if len(valid_annual) >= 3
        else np.nan
    )

    if len(valid_annual) >= 3:
        fi_p25 = float(valid_annual["flashiness_index"].quantile(0.25))
        fi_p75 = float(valid_annual["flashiness_index"].quantile(0.75))
        ac_p25 = float(valid_annual["lag1_autocorrelation"].quantile(0.25))
        ac_p75 = float(valid_annual["lag1_autocorrelation"].quantile(0.75))

        def classify(row: pd.Series) -> str:
            fi = row["flashiness_index"]
            ac = row["lag1_autocorrelation"]
            if pd.isna(fi) or pd.isna(ac):
                return "Insufficient data"
            if fi > fi_p75 and ac < ac_p25:
                return "Relatively flashy and weakly persistent"
            if fi < fi_p25 and ac > ac_p75:
                return "Relatively smooth and strongly persistent"
            if fi > fi_p75:
                return "Relatively flashy"
            if ac > ac_p75:
                return "Relatively persistent"
            return "Typical station behaviour"

        annual["diagnostic"] = annual.apply(classify, axis=1)
    else:
        fi_p25 = fi_p75 = ac_p25 = ac_p75 = np.nan
        annual["diagnostic"] = "Insufficient data"

    rolling = pd.DataFrame()
    rolling_correlation = np.nan

    if rolling_window_days is not None and not discharge.empty:
        q = discharge.astype(float)
        valid_pair = q.notna() & q.shift(1).notna()
        rolling_numerator = (
            q.diff().abs().where(valid_pair)
            .rolling(
                rolling_window_days,
                min_periods=rolling_minimum_valid_days - 1,
            )
            .sum()
        )
        rolling_denominator = q.rolling(
            rolling_window_days,
            min_periods=rolling_minimum_valid_days,
        ).sum()
        rolling_flashiness = (
            rolling_numerator / rolling_denominator
        ).replace([np.inf, -np.inf], np.nan)

        rolling_ac1 = q.rolling(
            rolling_window_days,
            min_periods=rolling_minimum_valid_days,
        ).corr(q.shift(1))

        rolling = pd.DataFrame(
            {
                "rolling_flashiness": rolling_flashiness,
                "rolling_ac1": rolling_ac1,
                "valid_days": q.notna().rolling(
                    rolling_window_days,
                    min_periods=1,
                ).sum(),
            }
        ).dropna(
            subset=["rolling_flashiness", "rolling_ac1"],
            how="all",
        )

        valid_rolling = rolling.dropna(
            subset=["rolling_flashiness", "rolling_ac1"]
        )
        if len(valid_rolling) >= 3:
            rolling_correlation = float(
                valid_rolling["rolling_flashiness"].corr(
                    valid_rolling["rolling_ac1"]
                )
            )

    metrics = {
        "whole_record_flashiness_index": whole_flashiness,
        "whole_record_lag1_autocorrelation": whole_ac1,
        "annual_flashiness_ac1_correlation": annual_correlation,
        "rolling_flashiness_ac1_correlation": rolling_correlation,
        "annual_flashiness_p25": fi_p25,
        "annual_flashiness_p75": fi_p75,
        "annual_ac1_p25": ac_p25,
        "annual_ac1_p75": ac_p75,
        "rolling_window_days": rolling_window_days,
    }

    if len(valid_annual) < 3:
        warnings.append(
            "Fewer than three retained years were available for the "
            "annual flashiness-autocorrelation relationship."
        )

    return SignatureResult(
        status=_result_status(
            valid_count=int(discharge.notna().sum()),
            minimum_required=3,
            warnings=warnings,
        ),
        metrics=metrics,
        tables={
            "annual_flashiness_autocorrelation": annual,
            "rolling_flashiness_autocorrelation": rolling,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )



def calculate_all_hydrological_signatures(
    series: pd.Series,
    zero_threshold: float = 1e-6,
    low_flow_percentile: float = 0.05,
    high_flow_percentile: float = 0.95,
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
    include_flashiness_persistence: bool = True,
    rolling_window_days: Optional[int] = 90,
    rolling_minimum_valid_days: int = 70,
    baseflow_alpha: float = 0.925,
    baseflow_passes: int = 3,
    derivative_smoothing_window_days: int = 5,
    derivative_min_consecutive_rising_days: int = 2,
) -> dict[str, SignatureResult]:
    """Calculate the complete current Layer 2 discharge-only signature set."""
    results: dict[str, SignatureResult] = {
        "flow_magnitude": calculate_flow_magnitude_signatures(series),
        "low_flow": calculate_low_flow_signatures(
            series=series,
            percentile=low_flow_percentile,
        ),
        "high_flow": calculate_high_flow_signatures(
            series=series,
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
            threshold=event_threshold,
            percentile=high_flow_percentile,
        ),
        "derivative_event_hydrographs": calculate_derivative_event_hydrographs(
            series=series,
            smoothing_window_days=derivative_smoothing_window_days,
            min_consecutive_rising_days=derivative_min_consecutive_rising_days,
        ),
    }

    if include_flashiness_persistence:
        results["flashiness_persistence"] = (
            calculate_flashiness_persistence_analysis(
                series=series,
                minimum_year_coverage=minimum_year_coverage,
                rolling_window_days=rolling_window_days,
                rolling_minimum_valid_days=rolling_minimum_valid_days,
            )
        )

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


__all__ = [
    "SignatureResult",
    "load_discharge_csv",
    "calculate_flow_magnitude_signatures",
    "calculate_low_flow_signatures",
    "calculate_high_flow_signatures",
    "calculate_annual_maximum_signatures",
    "calculate_zero_flow_signatures",
    "calculate_baseflow_signatures",
    "calculate_seasonality_signatures",
    "calculate_flashiness_signatures",
    "calculate_autocorrelation_signatures",
    "calculate_rising_limb_signatures",
    "calculate_recession_limb_signatures",
    "calculate_peak_signatures",
    "calculate_threshold_event_hydrographs",
    "calculate_derivative_event_hydrographs",
    "calculate_flashiness_persistence_analysis",
    "percentile_diagnostic",
    "calculate_all_hydrological_signatures",
    "signatures_to_summary_table",
    "run_hydrological_signatures_from_csv",
]
