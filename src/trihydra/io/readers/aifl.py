"""Reader for trusted local AIFL result pickle files."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import xarray as xr

from src.trihydra.io.models import SourceProvenance, StationData


def load_aifl_result(path: str | Path, station_id: str) -> StationData:
    """Load contemporaneous OBS/ML arrays from a trusted AIFL pickle.

    Pickle loading can execute code. Only use this reader with trusted files.
    ``time_step=0`` is selected because negative steps are input-history
    context for the stated date, not separate output dates or units.
    """
    resolved = Path(path).resolve()
    with resolved.open("rb") as handle:
        payload = pickle.load(handle)
    try:
        daily = payload[station_id]["1D"]
        dataset = daily["xr"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unexpected AIFL pickle hierarchy: {resolved}") from error
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("AIFL 'xr' payload is not an xarray Dataset.")
    for variable in ("streamflow_obs", "streamflow_sim"):
        if variable not in dataset:
            raise ValueError(f"AIFL dataset lacks {variable}.")
    dates = pd.DatetimeIndex(dataset["date"].values.copy(), name="date")
    obs = pd.Series(
        dataset["streamflow_obs"].sel(time_step=0).values.copy(),
        index=dates, name="obs",
    ).astype(float)
    ml = pd.Series(
        dataset["streamflow_sim"].sel(time_step=0).values.copy(),
        index=dates, name="ml",
    ).astype(float)
    common = dict(
        path=resolved,
        format="pickle/xarray",
        station_coordinate="pickle station key",
        time_coordinate="date",
        unit="mm/day",
        transformations=("selected contemporaneous time_step=0",),
    )
    result = StationData(
        station_id=station_id,
        obs=obs,
        ml=ml,
        unit="mm/day",
        obs_provenance=SourceProvenance(
            variable="streamflow_obs", **common
        ),
        ml_provenance=SourceProvenance(
            variable="streamflow_sim", **common
        ),
        metadata={
            "stored_performance_metrics": {
                key: value for key, value in daily.items() if key != "xr"
            }
        },
    )
    result.validate_raw_preservation()
    return result


__all__ = ["load_aifl_result"]
