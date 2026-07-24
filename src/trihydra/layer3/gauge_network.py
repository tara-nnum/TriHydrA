"""
gauge_network.py

Loads the gauge-network metadata (catchment, river, coordinates, area
for every gauge in the network) and finds "context candidates" for a
target station: nearby gauges on the same catchment/river, ranked and
classified into a confidence tier.

Pure computation -- no discharge data is touched here at all, which is
exactly why this module works today even though only 3 stations have
full OBS+AIFL time series: candidate SELECTION only needs the network
metadata (coordinates, catchment, river names), never the discharge
itself. Getting a candidate's actual discharge is nc_loader.py's job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

# Fallback priority for catchment area -- not every gauge has a clean
# value in every source column, so try each in turn and keep the
# first positive number found.
AREA_COLUMNS = [
    "static_area_km2",
    "drainage_area_provided",
    "DrainingArea.km2.Provider",
    "area_MERIT_1min",
    "DrainingArea.km2.LDD",
]


def _first_positive_value(row: pd.Series, columns: list[str]) -> float:
    for column in columns:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value) and value > 0:
            return float(value)
    return np.nan


def load_gauge_network(
    outlets_path: str | Path,
    static_path: str | Path,
) -> pd.DataFrame:
    """
    Load and merge the network's gauge metadata.

    `outlets_path` supplies catchment, river, station name, and outlet
    coordinates. `static_path` supplies the preferred catchment area
    where available (falling back through AREA_COLUMNS otherwise).
    """
    outlets = pd.read_csv(outlets_path).copy()
    outlets["outlet_row_index"] = np.arange(len(outlets))

    static = (
        pd.read_csv(static_path, usecols=["basin", "area"])
        .rename(columns={"basin": "gauge_id", "area": "static_area_km2"})
    )

    meta = outlets.merge(static, on="gauge_id", how="left")

    meta["area_km2"] = meta.apply(
        lambda row: _first_positive_value(row, AREA_COLUMNS),
        axis=1,
    )

    meta["StationLat"] = pd.to_numeric(meta["StationLat"], errors="coerce")
    meta["StationLon"] = pd.to_numeric(meta["StationLon"], errors="coerce")

    meta = meta[
        meta["StationLat"].between(-90, 90)
        & meta["StationLon"].between(-180, 180)
    ].copy()

    return meta


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. No local projected CRS is needed
    for a global gauge network -- WGS84 lat/lon in, km out."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _normalised_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold()


def _adaptive_radius_km(area_km2: float) -> float:
    """Search radius scales with catchment size (2 x sqrt(area)),
    clamped to a sensible 50-500 km range -- deliberately a simple
    rule for this stage, not a hydraulically-derived one."""
    if pd.isna(area_km2) or area_km2 <= 0:
        return 200.0
    return float(np.clip(2.0 * np.sqrt(area_km2), 50.0, 500.0))


def find_context_candidates(
    meta: pd.DataFrame,
    target_id: str,
    maximum_candidates: int = 10,
) -> dict:
    """
    Find and rank context candidates for one target gauge.

    Candidate priority:
      1. Same named catchment AND same named river.
      2. Same named catchment but a different river/tributary.
      3. Geographic proximity ranks candidates within either group; it
         does not by itself prove hydrological connectivity.

    Context tiers (interpretation text lives in diagnostics.py, not
    here -- this function only returns the raw tier label):
      strong    -- >= 2 same-river candidates
      moderate  -- 1 same-river candidate plus >= 2 total
      weak      -- >= 2 same-catchment candidates, none same-river
      limited   -- exactly 1 candidate, no majority agreement possible
      unavailable -- no suitable candidate at all (a valid outcome,
                     not a data-quality problem)
    """
    target_rows = meta.loc[meta["gauge_id"].eq(target_id)]
    if target_rows.empty:
        raise KeyError(f"{target_id} was not found in the gauge network metadata.")

    target = target_rows.iloc[0]
    candidates = meta.loc[~meta["gauge_id"].eq(target_id)].copy()

    candidates["distance_km"] = haversine_km(
        target["StationLat"], target["StationLon"],
        candidates["StationLat"].to_numpy(), candidates["StationLon"].to_numpy(),
    )

    target_catchment = str(target.get("Catchment", "")).strip().casefold()
    target_river = str(target.get("River", "")).strip().casefold()

    candidates["same_catchment"] = (
        _normalised_text(candidates["Catchment"]).eq(target_catchment)
        & bool(target_catchment)
    )
    candidates["same_river"] = (
        candidates["same_catchment"]
        & _normalised_text(candidates["River"]).eq(target_river)
        & bool(target_river)
    )

    target_area = target["area_km2"]
    candidates["area_ratio_to_target"] = candidates["area_km2"] / target_area
    candidates["area_similarity"] = np.minimum(
        candidates["area_ratio_to_target"],
        1.0 / candidates["area_ratio_to_target"],
    )

    # Likely a duplicate record of the target itself: almost the same
    # point, almost the same area.
    candidates["likely_duplicate"] = (
        candidates["distance_km"].lt(1.0)
        & candidates["area_similarity"].ge(0.95)
    )

    radius = _adaptive_radius_km(target_area)

    eligible = candidates[
        candidates["same_catchment"]
        & candidates["distance_km"].le(radius)
        & ~candidates["likely_duplicate"]
    ].copy()

    eligible["priority"] = np.where(eligible["same_river"], 1, 2)
    eligible = eligible.sort_values(
        ["priority", "distance_km", "area_similarity"],
        ascending=[True, True, False],
    )

    same_river_count = int(eligible["same_river"].sum())
    total_count = len(eligible)

    if same_river_count >= 2:
        status = "strong"
    elif same_river_count == 1 and total_count >= 2:
        status = "moderate"
    elif total_count >= 2:
        status = "weak"
    elif total_count == 1:
        status = "limited"
    else:
        status = "unavailable"

    result_columns = [
        "gauge_id", "StationName", "Catchment", "River",
        "StationLat", "StationLon", "area_km2", "distance_km",
        "area_ratio_to_target", "area_similarity", "same_river",
        "outlet_row_index",
    ]

    return {
        "target_id": target_id,
        "target": target,
        "radius_km": radius,
        "status": status,
        "candidates": eligible[result_columns].head(maximum_candidates),
    }
