"""Production ingestion boundary for Caravan observation NetCDF files.

The loader accepts one station ID, several station IDs, or every station with
at least one dated streamflow observation. It preserves the complete source
calendar and every NaN value. It never fills, interpolates, clips, resamples,
sorts, aggregates, or converts discharge values.

For large files, :func:`iter_caravan_stations` yields one station at a time so
an all-station run does not keep every hydrograph in memory simultaneously.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from src.trihydra.io.models import StationData
from src.trihydra.io.readers.caravan import CaravanObservations


StationSelection = str | Sequence[str] | Literal["all"] | None


def _normalise_selection(
    requested: StationSelection,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve a single, multiple, or all-station request deterministically."""
    if requested is None or (
        isinstance(requested, str) and requested.casefold() == "all"
    ):
        return available
    if isinstance(requested, str):
        selected = (requested,)
    elif isinstance(requested, Iterable):
        selected = tuple(map(str, requested))
    else:
        raise TypeError("stations must be one ID, a sequence of IDs, or 'all'.")

    if not selected:
        raise ValueError("The station selection is empty.")
    selected = tuple(dict.fromkeys(selected))
    missing = [station for station in selected if station not in available]
    if missing:
        preview = ", ".join(available[:5])
        raise KeyError(
            f"Caravan station(s) not found: {missing}. "
            f"First available IDs: {preview}"
        )
    return selected


def station_availability(station: StationData) -> dict:
    """Summarise availability without changing the raw observation series."""
    series = station.obs
    valid = series.notna()
    first_valid = series.index[valid][0] if valid.any() else None
    last_valid = series.index[valid][-1] if valid.any() else None
    return {
        "station_id": station.station_id,
        "source_path": str(station.obs_provenance.path),
        "variable": station.obs_provenance.variable,
        "unit": station.unit,
        "calendar_start": series.index.min() if len(series) else None,
        "calendar_end": series.index.max() if len(series) else None,
        "calendar_count": int(len(series)),
        "valid_count": int(valid.sum()),
        "missing_count": int(series.isna().sum()),
        "first_valid_date": first_valid,
        "last_valid_date": last_valid,
    }


def iter_caravan_stations(
    netcdf_path: str | Path,
    stations: StationSelection = "all",
    *,
    unit: str = "mm/day",
    skip_empty_when_all: bool = True,
) -> Iterator[StationData]:
    """Yield raw Caravan observations for one, several, or all stations.

    An explicitly requested station with no valid streamflow raises an error.
    In ``"all"`` mode, entirely empty stations are skipped by default because
    there is no streamflow information for Layers 1-2 to calculate.
    """
    all_mode = stations is None or (
        isinstance(stations, str) and stations.casefold() == "all"
    )
    with CaravanObservations(netcdf_path, unit=unit) as source:
        selected = _normalise_selection(stations, source.station_ids)
        for station_id in selected:
            station = source.load_station(station_id)
            if not isinstance(station.obs.index, pd.DatetimeIndex):
                raise TypeError(
                    f"{station_id}: source time coordinate is not datetime."
                )
            if len(station.obs) == 0 or not station.obs.notna().any():
                if all_mode and skip_empty_when_all:
                    continue
                raise ValueError(
                    f"{station_id}: no dated streamflow observations are available."
                )
            station.validate_raw_preservation()
            yield station


def load_caravan_stations(
    netcdf_path: str | Path,
    stations: StationSelection = "all",
    *,
    unit: str = "mm/day",
    skip_empty_when_all: bool = True,
) -> list[StationData]:
    """Materialise selected stations; prefer iteration for full-network runs."""
    return list(
        iter_caravan_stations(
            netcdf_path,
            stations,
            unit=unit,
            skip_empty_when_all=skip_empty_when_all,
        )
    )


__all__ = [
    "StationSelection",
    "iter_caravan_stations",
    "load_caravan_stations",
    "station_availability",
]
