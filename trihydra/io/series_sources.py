"""One functional adapter path for configured CSV, NetCDF, and pickle input."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trihydra.io.models import SourceProvenance, StationData
from trihydra.io.netcdf import (
    detect_netcdf_schema,
    netcdf_station_ids,
    open_netcdf,
    read_netcdf_series,
)
from trihydra.io.pickle_results import load_model_result_pickle
from trihydra.settings.models import Series1Config, TimespanConfig, TriHydrAConfig


@dataclass(frozen=True)
class SeriesRecord:
    """One station's untouched selected values and source description."""

    station_id: str
    values: pd.Series
    provenance: SourceProvenance
    metadata: dict


SeriesLoader = Callable[[str], SeriesRecord]


def _apply_timespan(values: pd.Series, timespan: TimespanConfig) -> pd.Series:
    if timespan.mode == "full":
        return values
    start = pd.Timestamp(timespan.start_date)
    end = pd.Timestamp(timespan.end_date) + pd.Timedelta(days=1)
    return values.loc[(values.index >= start) & (values.index < end)]


def _date_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _timespan_metadata(
    source: pd.Series, selected: pd.Series, timespan: TimespanConfig
) -> dict[str, object]:
    source_dates = pd.DatetimeIndex(source.index)
    selected_dates = pd.DatetimeIndex(selected.index)
    valid = selected.dropna()
    return {
        "requested_mode": timespan.mode,
        "requested_start": _date_text(timespan.start_date),
        "requested_end": _date_text(timespan.end_date),
        "source_calendar_start": _date_text(source_dates.min()) if len(source_dates) else None,
        "source_calendar_end": _date_text(source_dates.max()) if len(source_dates) else None,
        "selected_calendar_start": _date_text(selected_dates.min()) if len(selected_dates) else None,
        "selected_calendar_end": _date_text(selected_dates.max()) if len(selected_dates) else None,
        "first_valid_date": _date_text(valid.index.min()) if not valid.empty else None,
        "last_valid_date": _date_text(valid.index.max()) if not valid.empty else None,
        "selected_row_count": int(len(selected)),
        "selected_valid_count": int(valid.size),
    }


def _record(
    station_id: str,
    values: pd.Series,
    provenance: SourceProvenance,
    metadata: dict,
    config: Series1Config,
) -> SeriesRecord:
    selected = _apply_timespan(values, config.timespan).copy(deep=True)
    selected.name = config.name
    result_metadata = dict(metadata)
    result_metadata["timespan"] = _timespan_metadata(
        values, selected, config.timespan
    )
    return SeriesRecord(station_id, selected, provenance, result_metadata)


@contextmanager
def open_series_source(
    config: Series1Config,
) -> Iterator[tuple[tuple[str, ...], SeriesLoader]]:
    """Yield available station IDs and one loader for the configured source."""
    if config.format == "netcdf":
        dataset = open_netcdf(config.path, config.engine)
        try:
            variable, time_name, station_name = detect_netcdf_schema(
                dataset,
                variable=config.variable,
                time_coordinate=config.time_coordinate,
                station_coordinate=config.station_coordinate,
            )
            identifiers = netcdf_station_ids(dataset, station_name)

            def load(station_id: str) -> SeriesRecord:
                values = read_netcdf_series(
                    dataset,
                    station_id,
                    variable=variable,
                    time_coordinate=time_name,
                    station_coordinate=station_name,
                )
                provenance = SourceProvenance(
                    path=config.path,
                    format="NetCDF",
                    variable=variable,
                    station_coordinate=station_name,
                    time_coordinate=time_name,
                    unit=config.units,
                    transformations=("xarray missing-value decoding",),
                )
                return _record(station_id, values, provenance, {}, config)

            yield identifiers, load
        finally:
            dataset.close()
        return

    if config.format == "csv":
        frame = pd.read_csv(config.path)
        if config.date_column not in frame.columns:
            raise ValueError(f"CSV date column not found: {config.date_column!r}.")
        dates = pd.to_datetime(frame.pop(config.date_column), errors="raise")
        if not len(frame.columns):
            raise ValueError("Wide CSV requires at least one station column.")
        frame = frame.apply(pd.to_numeric, errors="raise")
        frame.index = pd.DatetimeIndex(dates, name=config.date_column)
        identifiers = tuple(map(str, frame.columns))

        def load(station_id: str) -> SeriesRecord:
            matches = [name for name in frame.columns if str(name) == str(station_id)]
            if len(matches) != 1:
                raise KeyError(
                    f"Expected one CSV station {station_id}; found {len(matches)}."
                )
            values = frame[matches[0]].copy(deep=True)
            values.name = config.name
            provenance = SourceProvenance(
                path=config.path,
                format="wide CSV",
                variable=str(matches[0]),
                station_coordinate="column name",
                time_coordinate=config.date_column,
                unit=config.units,
                transformations=("parsed configured date column",),
            )
            return _record(str(station_id), values, provenance, {}, config)

        yield identifiers, load
        return

    if config.format == "aifl_pickle":
        path = Path(config.path)
        candidates = [path] if path.is_file() else sorted(path.glob("*_results.p"))
        paths = {
            candidate.name.removesuffix("_results.p"): candidate
            for candidate in candidates
        }
        if not paths:
            raise FileNotFoundError(f"No *_results.p files found under: {path}")

        def load(station_id: str) -> SeriesRecord:
            if station_id not in paths:
                raise KeyError(f"AIFL result is unavailable for station {station_id!r}.")
            result = load_model_result_pickle(
                paths[station_id],
                station_id=station_id,
                unit=config.units,
                time_step=config.time_step,
                observation_variable=config.observation_variable,
                simulation_variable=config.simulation_variable,
            )
            use_observation = config.role in {"observation", "historical_observation"}
            values = result.obs if use_observation else result.ml
            provenance = result.obs_provenance if use_observation else result.ml_provenance
            return _record(station_id, values, provenance, result.metadata, config)

        yield tuple(paths), load
        return

    raise ValueError(f"Unsupported input format: {config.format}")


def station_from_records(
    primary: SeriesRecord,
    secondary: SeriesRecord | None,
    config: TriHydrAConfig,
) -> StationData:
    """Combine source records using TriHydrA's neutral two-series contract."""
    metadata = {
        "series1_metadata": primary.metadata,
        "series2_metadata": {} if secondary is None else secondary.metadata,
        "series2_status": "disabled" if not config.series2.enabled else (
            "not_available" if secondary is None else "available"
        ),
    }
    return StationData.from_series(
        station_id=primary.station_id,
        series1=primary.values,
        unit=config.series1.units,
        series1_provenance=primary.provenance,
        series2=None if secondary is None else secondary.values,
        series2_provenance=None if secondary is None else secondary.provenance,
        metadata=metadata,
        series1_name=config.series1.name,
        series1_role=config.series1.role,
        series2_name=config.series2.name,
        series2_role=config.series2.role,
    )


__all__ = ["SeriesLoader", "SeriesRecord", "open_series_source", "station_from_records"]
