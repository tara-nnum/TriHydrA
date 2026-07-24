"""
nc_loader.py

A minimal loader for candidate-gauge discharge, read from the network
NetCDF file (e.g. caravan_observations.nc). This is currently the only
data source available for any gauge other than the handful with their
own full OBS+AIFL CSVs, so it's kept -- but deliberately slimmed down
from the exploratory prototype:

- KEPT: lightweight auto-detection of which variable/dimension is
  discharge/time/station, since it's cheap (just inspecting names, no
  data read) and makes this resilient to minor naming differences.
- DROPPED from the default path: the heavy one-time trust-building
  step (correlating extracted NetCDF rows against known CSVs to prove
  row N really is gauge N). That's a "confirm this once when the file
  changes" step, not something to redo on every call. It's kept below
  as validate_row_order_against_known_series(), callable by hand if
  the NetCDF is ever replaced/refreshed and that trust needs rebuilding.

Only observed discharge lives in this NetCDF -- there is no simulated/
model discharge for the network's other gauges, which is exactly why
Layer 3 only ever compares OBS against OBS (see layer3.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

_PREFERRED_DISCHARGE_TERMS = ("discharge", "streamflow", "observ", "qobs", "flow", "q")


def open_discharge_dataset(nc_path: str | Path) -> xr.Dataset:
    return xr.open_dataset(nc_path)


def detect_discharge_variable_and_dims(ds: xr.Dataset) -> tuple[str, str, str]:
    """
    Auto-detect which data variable is discharge, and which of its
    dimensions is time vs. station. Raises a clear error rather than
    guessing silently wrong if the file doesn't match expectations.
    """
    names = list(ds.data_vars)

    variable_name = None
    for term in _PREFERRED_DISCHARGE_TERMS:
        matches = [name for name in names if term in name.casefold()]
        if matches:
            variable_name = matches[0]
            break
    if variable_name is None:
        if len(names) == 1:
            variable_name = names[0]
        else:
            raise ValueError(
                "Could not identify the discharge variable automatically. "
                f"Available variables: {names}"
            )

    q_data = ds[variable_name]

    time_candidates = [
        dim for dim in q_data.dims
        if "time" in dim.casefold() or "date" in dim.casefold()
    ]
    if not time_candidates:
        raise ValueError(f"No time/date dimension was recognised in {q_data.dims}.")
    time_dim = time_candidates[0]

    station_candidates = [dim for dim in q_data.dims if dim != time_dim]
    if len(station_candidates) != 1:
        raise ValueError(
            "The discharge variable does not have exactly one non-time "
            f"dimension. Set the station dimension manually. Dimensions: {q_data.dims}"
        )
    station_dim = station_candidates[0]

    return variable_name, time_dim, station_dim


def load_discharge_by_row(
    ds: xr.Dataset,
    variable_name: str,
    time_dim: str,
    station_dim: str,
    row_index: int,
) -> pd.Series:
    """Extract one station's discharge series by its row position in
    the station dimension (not by gauge ID -- this NetCDF has no
    gauge-ID coordinate, only row order, so the caller must already
    know which row corresponds to which gauge_id; see
    load_discharge_by_gauge_id for the usual, safer way to call this)."""
    series = (
        ds[variable_name]
        .isel({station_dim: int(row_index)})
        .to_series()
        .rename("discharge")
    )

    if isinstance(series.index, pd.MultiIndex):
        if time_dim not in series.index.names:
            raise ValueError(f"Unexpected NetCDF series index: {series.index.names}")
        series = series.reset_index().set_index(time_dim)["discharge"]

    series.index = pd.to_datetime(series.index)
    series = pd.to_numeric(series, errors="coerce")
    return series.sort_index()


def load_discharge_by_gauge_id(
    gauge_id: str,
    meta: pd.DataFrame,
    ds: xr.Dataset,
    variable_name: str,
    time_dim: str,
    station_dim: str,
) -> pd.Series:
    """
    The usual entry point: look up gauge_id's row in the (already-
    loaded) gauge network metadata, then extract that row's discharge.
    """
    matches = meta.loc[meta["gauge_id"].eq(gauge_id), "outlet_row_index"]
    if matches.empty:
        raise KeyError(f"{gauge_id} was not found in the gauge network metadata.")

    row_index = int(matches.iloc[0])
    return load_discharge_by_row(ds, variable_name, time_dim, station_dim, row_index)


def validate_row_order_against_known_series(
    meta: pd.DataFrame,
    ds: xr.Dataset,
    variable_name: str,
    time_dim: str,
    station_dim: str,
    known_series: dict[str, pd.Series],
    minimum_absolute_spearman: float = 0.95,
) -> pd.DataFrame:
    """
    One-time trust-building check, NOT part of the regular Layer 3
    run -- call this by hand after the NetCDF is first hooked up, or
    if it's ever replaced/refreshed, to confirm the row-order mapping
    is still trustworthy. `known_series` should map a few gauge_ids to
    their already-trusted observed series (e.g. from their own CSVs).

    Returns a table of overlap days and Spearman correlation per gauge
    checked; row-order trust is supported if every value clears
    `minimum_absolute_spearman`.
    """
    rows = []
    for gauge_id, known in known_series.items():
        nc_series = load_discharge_by_gauge_id(gauge_id, meta, ds, variable_name, time_dim, station_dim)
        joined = pd.concat(
            [known.rename("known"), nc_series.rename("netcdf")], axis=1
        ).dropna()

        correlation = (
            joined.corr(method="spearman").iloc[0, 1] if len(joined) >= 2 else np.nan
        )
        rows.append({
            "gauge_id": gauge_id,
            "overlap_days": len(joined),
            "spearman_correlation": correlation,
            "trusted": bool(pd.notna(correlation) and abs(correlation) >= minimum_absolute_spearman),
        })

    return pd.DataFrame(rows)
