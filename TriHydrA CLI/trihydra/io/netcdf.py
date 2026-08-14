"""Input-agnostic reader for station-by-time NetCDF datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from trihydra.io.models import SourceProvenance, StationData


class NetCDFStationSource:
    """Context-managed access to one station-by-time NetCDF variable."""

    def __init__(
        self,
        path: str | Path,
        unit: str = "mm/day",
        *,
        variable: str | None = None,
        time_coordinate: str | None = None,
        station_coordinate: str | None = None,
    ):
        self.path = Path(path).resolve()
        self.unit = unit
        self.dataset: xr.Dataset | None = None
        self.variable = "" if variable is None else str(variable)
        self.time_coordinate = "" if time_coordinate is None else str(time_coordinate)
        self.station_coordinate = "" if station_coordinate is None else str(station_coordinate)

    def __enter__(self) -> "NetCDFStationSource":
        self.dataset = xr.open_dataset(self.path)
        self._detect_schema()
        return self

    def __exit__(self, *_args) -> None:
        if self.dataset is not None:
            self.dataset.close()
        self.dataset = None

    def _detect_schema(self) -> None:
        if self.dataset is None:
            raise RuntimeError("Dataset is not open.")
        variables = list(self.dataset.data_vars)
        if self.variable:
            if self.variable not in self.dataset.data_vars:
                raise KeyError(f"Configured discharge variable not found: {self.variable}")
        else:
            preferred = [
                name for name in variables
                if any(term in name.casefold() for term in ("streamflow", "discharge", "qobs"))
            ]
            if len(preferred) == 1:
                self.variable = preferred[0]
            elif len(variables) == 1:
                self.variable = variables[0]
            else:
                raise ValueError(f"Ambiguous discharge variable: {variables}")

        array = self.dataset[self.variable]
        if self.time_coordinate:
            if self.time_coordinate not in array.dims:
                raise ValueError(
                    f"Configured time coordinate {self.time_coordinate!r} is not "
                    f"a dimension of {self.variable!r}."
                )
        else:
            time_dims = [
                dim for dim in array.dims
                if dim in self.dataset.coords
                and np.issubdtype(self.dataset[dim].dtype, np.datetime64)
            ]
            if len(time_dims) != 1:
                raise ValueError(f"Expected one datetime coordinate; found {time_dims}.")
            self.time_coordinate = time_dims[0]
        station_dims = [dim for dim in array.dims if dim != self.time_coordinate]
        if len(station_dims) != 1:
            raise ValueError(f"Expected one station dimension; found {station_dims}.")
        station_dim = station_dims[0]

        candidates = []
        for name, coordinate in self.dataset.coords.items():
            if coordinate.dims == (station_dim,):
                candidates.append(name)
        if self.station_coordinate:
            if self.station_coordinate not in candidates:
                raise ValueError(
                    f"Configured station coordinate {self.station_coordinate!r} "
                    f"is not aligned with dimension {station_dim!r}."
                )
        else:
            named = [
                name for name in candidates
                if any(term in name.casefold() for term in ("basin", "station", "gauge"))
            ]
            if len(named) == 1:
                self.station_coordinate = named[0]
            elif station_dim in candidates:
                self.station_coordinate = station_dim
            else:
                raise ValueError(f"Ambiguous station coordinate: {candidates}")

    @property
    def station_ids(self) -> tuple[str, ...]:
        """Return station identifiers exactly as represented by the source."""
        if self.dataset is None:
            raise RuntimeError("Dataset is not open.")
        return tuple(map(str, self.dataset[self.station_coordinate].values))

    def load_series(self, station_id: str) -> pd.Series:
        """Extract one raw station array without filling or dropping NaNs."""
        if self.dataset is None:
            raise RuntimeError("Dataset is not open.")
        ids = np.asarray(self.dataset[self.station_coordinate].values).astype(str)
        positions = np.flatnonzero(ids == str(station_id))
        if len(positions) != 1:
            raise KeyError(f"Expected one station {station_id}; found {len(positions)}.")
        station_dim = self.dataset[self.station_coordinate].dims[0]
        values = np.asarray(
            self.dataset[self.variable].isel({station_dim: int(positions[0])}).values
        ).copy()
        dates = pd.DatetimeIndex(
            np.asarray(self.dataset[self.time_coordinate].values).copy(),
            name=self.time_coordinate,
        )
        series = pd.Series(values, index=dates, name="obs", copy=False)
        if len(series) != len(values) or int(series.isna().sum()) != int(pd.isna(values).sum()):
            raise AssertionError("NetCDF extraction changed length or NaN count.")
        return series

    def load_station(
        self, station_id: str, metadata: dict | None = None
    ) -> StationData:
        """Return one canonical raw-observation station object."""
        station = StationData(
            station_id=station_id,
            obs=self.load_series(station_id),
            unit=self.unit,
            obs_provenance=SourceProvenance(
                path=self.path,
                format="NetCDF",
                variable=self.variable,
                station_coordinate=self.station_coordinate,
                time_coordinate=self.time_coordinate,
                unit=self.unit,
                transformations=("xarray missing-value decoding",),
            ),
            metadata={} if metadata is None else dict(metadata),
        )
        station.validate_raw_preservation()
        return station

    def load_many(self, station_ids: Iterable[str]) -> dict[str, pd.Series]:
        """Load raw OBS for several stations into a gauge-ID mapping."""
        return {station_id: self.load_series(station_id) for station_id in station_ids}


__all__ = ["NetCDFStationSource"]
