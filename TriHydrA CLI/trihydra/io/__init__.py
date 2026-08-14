"""Canonical input/output boundary for TriHydrA."""

from trihydra.io.ingestion import (
    StationSelection,
    iter_netcdf_stations,
    load_netcdf_stations,
    station_availability,
)
from trihydra.io.models import StationData, SourceProvenance
from trihydra.io.availability import pair_availability, series_availability
from trihydra.io.netcdf import NetCDFStationSource
from trihydra.io.pickle_results import (
    ModelResultSelection,
    iter_model_result_pickles,
    load_model_result_pickle,
    load_model_result_pickles,
)

__all__ = [
    "StationData",
    "SourceProvenance",
    "ModelResultSelection",
    "NetCDFStationSource",
    "StationSelection",
    "iter_model_result_pickles",
    "iter_netcdf_stations",
    "load_model_result_pickle",
    "load_model_result_pickles",
    "load_netcdf_stations",
    "pair_availability",
    "series_availability",
    "station_availability",
]
