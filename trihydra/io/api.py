"""Public, format-neutral loading for Python and Jupyter users."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import time
from typing import Callable, Mapping

from trihydra.io.models import StationData
from trihydra.io.selection import select_station_ids
from trihydra.io.series_sources import open_series_source, station_from_records
from trihydra.settings.loader import load_toml_config, resolve_station_selection


LoadProgress = Callable[[str, Mapping[str, object]], None]


def load_stations(
    config: str | Path = "trihydra.toml",
    *,
    continue_on_station_error: bool = False,
    progress: LoadProgress | None = None,
) -> list[StationData]:
    """Load configured stations without running or writing TriHydrA results.

    The TOML file is validated exactly as it is for the CLI. CSV, NetCDF and
    trusted AIFL-pickle inputs therefore use the same adapters, station
    selection, timespans, names, roles, units and provenance in both
    interfaces. A station missing from optional ``series2`` remains usable as
    a series1-only station.
    """
    public = load_toml_config(config)
    selection = resolve_station_selection(public)
    loaded: list[StationData] = []
    with ExitStack() as sources:
        series1_ids, load_series1 = sources.enter_context(
            open_series_source(public.series1)
        )
        series2_access = (
            sources.enter_context(open_series_source(public.series2))
            if public.series2.enabled
            else None
        )
        series2_ids, load_series2 = series2_access or ((), None)
        station_ids = select_station_ids(selection, series1_ids)
        for station_id in station_ids:
            started = time.perf_counter()
            try:
                primary = load_series1(station_id)
                secondary = None
                if load_series2 is not None and station_id in series2_ids:
                    secondary = load_series2(station_id)
                loaded.append(station_from_records(primary, secondary, public))
            except Exception as error:
                row = {
                    "station_id": str(station_id),
                    "status": "failed",
                    "series2_status": "not_attempted",
                    "review_required": False,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "layer3_status": "not_assessed",
                }
                if progress is not None:
                    progress("failed", row)
                if not continue_on_station_error:
                    raise
    return loaded


__all__ = ["load_stations"]
