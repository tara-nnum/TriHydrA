"""Canonical input boundary for TriHydrA."""

from trihydra.io.api import load_stations
from trihydra.io.models import StationData, SourceProvenance
from trihydra.io.availability import (
    pair_availability,
    series_availability,
    station_availability,
)

__all__ = [
    "StationData",
    "SourceProvenance",
    "load_stations",
    "pair_availability",
    "series_availability",
    "station_availability",
]
