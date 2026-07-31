"""Reader for the Caravan observation NetCDF used by TriHydrA."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from src.trihydra.io.models import SourceProvenance, StationData


class CaravanObservations:
    """Context-managed, station-ID-based access to raw Caravan observations."""

    def __init__(self, path: str | Path, unit: str = "mm/day"):
        self.path = Path(path).resolve()
        self.unit = unit
        self.dataset: xr.Dataset | None = None
        self.variable = ""
        self.time_coordinate = ""
        self.station_coordinate = ""

    def __enter__(self) -> "CaravanObservations":
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
        preferred = [
            name for name in variables
            if any(term in name.casefold() for term in ("streamflow", "discharge", "qobs"))
        ]
        if len(preferred) == 1:
            self.variable = preferred[0]
        elif len(variables) == 1:
            self.variable = variables[0]
        else:
            raise ValueError(f"Ambiguous Caravan discharge variable: {variables}")

        array = self.dataset[self.variable]
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
            raise KeyError(f"Expected one Caravan station {station_id}; found {len(positions)}.")
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
            raise AssertionError("Caravan extraction changed length or NaN count.")
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


__all__ = ["CaravanObservations"]
