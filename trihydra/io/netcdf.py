"""Schema detection and station extraction for station-by-time NetCDF data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def open_netcdf(path: str | Path, engine: str = "auto") -> xr.Dataset:
    """Open a NetCDF dataset for reading.

    Use the returned dataset as a context manager or close it after use.
    """
    kwargs = {} if engine == "auto" else {"engine": engine}
    return xr.open_dataset(Path(path).resolve(), **kwargs)


def detect_netcdf_schema(
    dataset: xr.Dataset,
    *,
    variable: str | None = None,
    time_coordinate: str | None = None,
    station_coordinate: str | None = None,
) -> tuple[str, str, str]:
    """Resolve discharge, time, and station names from a compatible dataset."""
    variables = list(dataset.data_vars)
    if variable:
        if variable not in dataset.data_vars:
            raise KeyError(f"Configured discharge variable not found: {variable}")
    else:
        preferred = [
            name for name in variables
            if any(term in name.casefold() for term in ("streamflow", "discharge", "qobs"))
        ]
        if len(preferred) == 1:
            variable = preferred[0]
        elif len(variables) == 1:
            variable = variables[0]
        else:
            raise ValueError(f"Ambiguous discharge variable: {variables}")

    array = dataset[variable]
    if time_coordinate:
        if time_coordinate not in array.dims:
            raise ValueError(
                f"Configured time coordinate {time_coordinate!r} is not "
                f"a dimension of {variable!r}."
            )
    else:
        time_dimensions = [
            dim for dim in array.dims
            if dim in dataset.coords
            and np.issubdtype(dataset[dim].dtype, np.datetime64)
        ]
        if len(time_dimensions) != 1:
            raise ValueError(
                f"Expected one datetime coordinate; found {time_dimensions}."
            )
        time_coordinate = time_dimensions[0]

    station_dimensions = [dim for dim in array.dims if dim != time_coordinate]
    if len(station_dimensions) != 1:
        raise ValueError(f"Expected one station dimension; found {station_dimensions}.")
    station_dimension = station_dimensions[0]
    candidates = [
        name for name, coordinate in dataset.coords.items()
        if coordinate.dims == (station_dimension,)
    ]
    if station_coordinate:
        if station_coordinate not in candidates:
            raise ValueError(
                f"Configured station coordinate {station_coordinate!r} "
                f"is not aligned with dimension {station_dimension!r}."
            )
    else:
        named = [
            name for name in candidates
            if any(term in name.casefold() for term in ("basin", "station", "gauge"))
        ]
        if len(named) == 1:
            station_coordinate = named[0]
        elif station_dimension in candidates:
            station_coordinate = station_dimension
        else:
            raise ValueError(f"Ambiguous station coordinate: {candidates}")
    return variable, time_coordinate, station_coordinate


def netcdf_station_ids(dataset: xr.Dataset, station_coordinate: str) -> tuple[str, ...]:
    """Return station identifiers exactly as represented by the source."""
    return tuple(map(str, dataset[station_coordinate].values))


def read_netcdf_series(
    dataset: xr.Dataset,
    station_id: str,
    *,
    variable: str,
    time_coordinate: str,
    station_coordinate: str,
) -> pd.Series:
    """Extract one raw station array without filling, sorting, or dropping NaNs."""
    identifiers = np.asarray(dataset[station_coordinate].values).astype(str)
    positions = np.flatnonzero(identifiers == str(station_id))
    if len(positions) != 1:
        raise KeyError(f"Expected one station {station_id}; found {len(positions)}.")
    station_dimension = dataset[station_coordinate].dims[0]
    values = np.asarray(
        dataset[variable].isel({station_dimension: int(positions[0])}).values
    ).copy()
    dates = pd.DatetimeIndex(
        np.asarray(dataset[time_coordinate].values).copy(), name=time_coordinate
    )
    series = pd.Series(values, index=dates, name=str(station_id), copy=False)
    if int(series.isna().sum()) != int(pd.isna(values).sum()):
        raise AssertionError("NetCDF extraction changed the NaN count.")
    return series


__all__ = [
    "detect_netcdf_schema",
    "netcdf_station_ids",
    "open_netcdf",
    "read_netcdf_series",
]
