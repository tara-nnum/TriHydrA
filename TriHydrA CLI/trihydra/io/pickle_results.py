"""Reader for trusted model-result pickles containing xarray datasets."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from trihydra.io.availability import pair_availability
from trihydra.io.models import SourceProvenance, StationData

ModelResultSelection = str | Sequence[str] | Literal["all"] | None


def _station_key(payload: object, path: Path, requested: str | None) -> str:
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError(f"Result pickle must contain exactly one station: {path}")
    discovered = str(next(iter(payload)))
    if requested is not None and str(requested) != discovered:
        raise ValueError(
            f"Requested station {requested!r} does not match pickle key "
            f"{discovered!r}: {path}"
        )
    filename_station = path.name.removesuffix("_results.p")
    if filename_station != discovered:
        raise ValueError(
            f"Pickle filename identifies {filename_station!r}, but its internal "
            f"station key is {discovered!r}: {path}"
        )
    return discovered


def _extract_contemporaneous(
    dataset: xr.Dataset,
    variable: str,
    time_step: int,
    name: str,
) -> pd.Series:
    if variable not in dataset:
        raise ValueError(f"Result dataset lacks required variable {variable!r}.")
    if "date" not in dataset.coords or "time_step" not in dataset.coords:
        raise ValueError("Result dataset requires date and time_step coordinates.")
    available_steps = np.asarray(dataset["time_step"].values)
    if time_step not in available_steps:
        raise ValueError(
            f"Requested time_step={time_step} is unavailable; "
            f"found {available_steps.tolist()}."
        )
    selected = dataset[variable].sel(time_step=time_step)
    if selected.dims != ("date",):
        raise ValueError(
            f"{variable!r} must reduce to one date dimension; found {selected.dims}."
        )
    dates = pd.DatetimeIndex(dataset["date"].values.copy(), name="date")
    if dates.has_duplicates:
        raise ValueError("Result date coordinate contains duplicate timestamps.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Result date coordinate must be sorted.")
    return pd.Series(selected.values.copy(), index=dates, name=name).astype(float)


def load_model_result_pickle(
    path: str | Path,
    station_id: str | None = None,
    *,
    unit: str = "mm/day",
    time_step: int = 0,
    observation_variable: str = "streamflow_obs",
    simulation_variable: str = "streamflow_sim",
) -> StationData:
    """Load one contemporaneous observation/simulation result pair.

    Pickle loading can execute code; only trusted result files are accepted.
    The supported result schema stores input-history positions at
    negative ``time_step`` values. Only ``time_step=0`` is contemporaneous
    with the labelled date and is therefore suitable for direct comparison.
    No missing or trailing values are filled, clipped, or otherwise changed.
    """
    resolved = Path(path).resolve()
    if time_step != 0:
        raise ValueError(
            "Direct result ingestion requires contemporaneous time_step=0. "
            "Negative steps are input-history context and cannot be treated "
            "as independent dated simulations."
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Result pickle does not exist: {resolved}")
    with resolved.open("rb") as handle:
        payload = pickle.load(handle)
    discovered = _station_key(payload, resolved, station_id)
    try:
        daily = payload[discovered]["1D"]
        dataset = daily["xr"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unexpected result-pickle hierarchy: {resolved}") from error
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Result 'xr' payload is not an xarray Dataset.")

    obs = _extract_contemporaneous(
        dataset, observation_variable, time_step, "obs"
    )
    simulation = _extract_contemporaneous(
        dataset, simulation_variable, time_step, "simulation"
    )
    steps = tuple(int(value) for value in dataset["time_step"].values)
    common = dict(
        path=resolved,
        format="trusted pickle/xarray result",
        station_coordinate="single pickle station key",
        time_coordinate="date",
        unit=unit,
        transformations=(f"selected contemporaneous time_step={time_step}",),
        details={
            "available_time_steps": steps,
            "negative_time_steps_role": "input-history context; not output dates",
            "unit_source": "reader argument/result-schema contract; absent from pickle attrs",
        },
    )
    result = StationData(
        station_id=discovered,
        obs=obs,
        ml=simulation,
        unit=unit,
        obs_provenance=SourceProvenance(variable=observation_variable, **common),
        ml_provenance=SourceProvenance(variable=simulation_variable, **common),
        metadata={
            "source_family": "model result pickle",
            "series_roles": {
                observation_variable: "observation",
                simulation_variable: "simulation",
            },
            "stored_performance_metrics": {
                key: value for key, value in daily.items() if key != "xr"
            },
            "comparison_availability": pair_availability(
                obs, simulation, "obs", "simulation"
            ),
            "negative_observation_count": int((obs < 0).sum()),
            "negative_simulation_count": int((simulation < 0).sum()),
        },
    )
    result.validate_raw_preservation()
    return result


def _normalise_selection(
    requested: ModelResultSelection, available: tuple[str, ...]
) -> tuple[str, ...]:
    if requested is None or (
        isinstance(requested, str) and requested.casefold() == "all"
    ):
        return available
    if isinstance(requested, str):
        selected = (requested,)
    elif isinstance(requested, Sequence):
        selected = tuple(dict.fromkeys(str(item) for item in requested))
    else:
        raise TypeError("stations must be one ID, a sequence of IDs, or 'all'.")
    missing = sorted(set(selected).difference(available))
    if missing:
        raise KeyError(f"Result station(s) not found: {missing}")
    return selected


def iter_model_result_pickles(
    directory: str | Path,
    stations: ModelResultSelection = "all",
    *,
    unit: str = "mm/day",
    time_step: int = 0,
    pattern: str = "*_results.p",
) -> Iterator[StationData]:
    """Yield one, selected, or all model-result pickles deterministically."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Result directory does not exist: {root}")
    paths = sorted(root.glob(pattern), key=lambda item: item.name.casefold())
    mapping = {path.name.removesuffix("_results.p"): path for path in paths}
    if len(mapping) != len(paths):
        raise ValueError("Result directory contains duplicate station filenames.")
    selected = _normalise_selection(stations, tuple(mapping))
    for station_id in selected:
        yield load_model_result_pickle(
            mapping[station_id], station_id, unit=unit, time_step=time_step
        )


def load_model_result_pickles(
    directory: str | Path,
    stations: ModelResultSelection = "all",
    **kwargs,
) -> list[StationData]:
    """Materialise selected results; iteration is preferred for batches."""
    return list(iter_model_result_pickles(directory, stations, **kwargs))


__all__ = [
    "ModelResultSelection",
    "iter_model_result_pickles",
    "load_model_result_pickle",
    "load_model_result_pickles",
]
