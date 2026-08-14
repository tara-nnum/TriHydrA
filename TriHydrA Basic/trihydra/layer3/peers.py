"""Select local and hydrological-analogue peers for Layer 3."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Any, Mapping

import numpy as np
import pandas as pd


PEER_COLUMNS = [
    "station_id",
    "river_name",
    "catchment_name",
    "catchment_area_km2",
    "climate_code",
    "climate_description",
    "distance_km",
    "catchment_area_ratio",
    "same_river",
    "same_catchment",
    "same_climate",
    "selection_reason",
]


@dataclass(frozen=True)
class PeerSelectionResult:
    """One peer group and an explicit readiness status."""

    target_station_id: str
    peer_type: str
    peers: pd.DataFrame
    status: str
    message: str

    @property
    def is_assessable(self) -> bool:
        return self.status == "ready"


@dataclass(frozen=True)
class ContextPeerGroups:
    """The two peer groups used for different Layer 3 questions."""

    target_station_id: str
    local: PeerSelectionResult
    analogues: PeerSelectionResult


def _haversine_km(latitude, longitude, other_latitudes, other_longitudes):
    """Calculate great-circle distance from one gauge to many gauges."""
    lat1, lon1 = radians(float(latitude)), radians(float(longitude))
    lat2 = np.radians(other_latitudes.astype(float).to_numpy())
    lon2 = np.radians(other_longitudes.astype(float).to_numpy())
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def _same_text(values: pd.Series, target: object) -> pd.Series:
    if pd.isna(target) or str(target).strip() == "":
        return pd.Series(False, index=values.index)
    return values.astype("string").str.strip().str.casefold() == str(target).strip().casefold()


def _candidate_table(
    stations: pd.DataFrame,
    target_station_id: str,
    series_type: str,
) -> tuple[pd.Series | None, pd.DataFrame, str | None]:
    target_rows = stations.loc[stations["station_id"] == str(target_station_id)]
    if target_rows.empty:
        return None, pd.DataFrame(), "Target station is absent from context metadata."
    if len(target_rows) > 1:
        return None, pd.DataFrame(), "Target station_id is not unique in context metadata."
    target = target_rows.iloc[0]
    candidates = stations.loc[
        (stations["station_id"] != target["station_id"])
        & (stations["series_type"] == series_type)
    ].copy()
    if candidates.empty:
        return target, candidates, "No other stations have the requested series type."

    candidates["distance_km"] = _haversine_km(
        target["latitude"], target["longitude"], candidates["latitude"], candidates["longitude"]
    )
    target_area = float(target["catchment_area_km2"])
    area = candidates["catchment_area_km2"].astype(float)
    candidates["catchment_area_ratio"] = np.maximum(area / target_area, target_area / area)
    candidates["same_catchment"] = _same_text(candidates["catchment_name"], target["catchment_name"])
    candidates["same_river"] = _same_text(candidates["river_name"], target["river_name"])
    candidates["same_climate"] = (
        candidates["climate_code"].notna()
        & pd.notna(target.get("climate_code"))
        & (candidates["climate_code"] == target.get("climate_code"))
    )
    if "climate_description" not in candidates:
        candidates["climate_description"] = pd.NA
    return target, candidates, None


def _finish(
    target_station_id: str,
    peer_type: str,
    candidates: pd.DataFrame,
    config: Mapping[str, Any],
) -> PeerSelectionResult:
    maximum = int(config.get("maximum_peers", 5))
    minimum = int(config.get("minimum_peers", 2))
    selected = candidates.head(maximum).reset_index(drop=True)
    selected = selected.loc[:, PEER_COLUMNS] if not selected.empty else pd.DataFrame(columns=PEER_COLUMNS)
    if len(selected) < minimum:
        return PeerSelectionResult(
            str(target_station_id), peer_type, selected, "not_assessed",
            f"Found {len(selected)} {peer_type} peer(s); at least {minimum} are required.",
        )
    return PeerSelectionResult(
        str(target_station_id), peer_type, selected, "ready",
        f"Selected {len(selected)} {peer_type} peers.",
    )


def select_local_peers(
    stations: pd.DataFrame,
    target_station_id: str,
    config: Mapping[str, Any],
    series_type: str = "observation",
) -> PeerSelectionResult:
    """Select nearby gauges for calendar-date comparisons."""
    _, candidates, error = _candidate_table(stations, target_station_id, series_type)
    if error:
        return PeerSelectionResult(str(target_station_id), "local", pd.DataFrame(columns=PEER_COLUMNS), "not_assessed", error)
    radius = float(config.get("maximum_search_radius_km", 50.0))
    candidates = candidates.loc[candidates["distance_km"] <= radius].copy()
    # River/catchment relationships rank local gauges; they do not bypass radius.
    candidates["relationship_rank"] = np.select(
        [candidates["same_river"] & candidates["same_catchment"], candidates["same_catchment"], candidates["same_river"]],
        [0, 1, 2], default=3,
    )
    candidates["selection_reason"] = candidates["relationship_rank"].map(
        {0: "nearby; same river and catchment", 1: "nearby; same catchment", 2: "nearby; same river", 3: "nearby station"}
    )
    candidates = candidates.sort_values(["relationship_rank", "distance_km", "catchment_area_ratio", "station_id"], kind="stable")
    return _finish(target_station_id, "local", candidates, config)


def select_analogue_peers(
    stations: pd.DataFrame,
    target_station_id: str,
    config: Mapping[str, Any],
    series_type: str = "observation",
    excluded_station_ids: set[str] | None = None,
) -> PeerSelectionResult:
    """Select climate- and scale-compatible gauges for behaviour comparisons."""
    _, candidates, error = _candidate_table(stations, target_station_id, series_type)
    if error:
        return PeerSelectionResult(str(target_station_id), "analogue", pd.DataFrame(columns=PEER_COLUMNS), "not_assessed", error)
    if excluded_station_ids:
        candidates = candidates.loc[
            ~candidates["station_id"].astype(str).isin(excluded_station_ids)
        ].copy()
    radius = float(config.get("maximum_search_radius_km", 1000.0))
    area_ratio = float(config.get("maximum_catchment_area_ratio", 2.0))
    mask = (candidates["distance_km"] <= radius) & (candidates["catchment_area_ratio"] <= area_ratio)
    if config.get("require_same_climate", True):
        mask &= candidates["same_climate"]
    candidates = candidates.loc[mask].copy()
    candidates["relationship_rank"] = np.select(
        [candidates["same_river"] & candidates["same_catchment"], candidates["same_catchment"], candidates["same_river"]],
        [0, 1, 2], default=3,
    )
    candidates["selection_reason"] = "similar area and climate"
    candidates.loc[candidates["relationship_rank"] == 0, "selection_reason"] += "; same river and catchment"
    candidates.loc[candidates["relationship_rank"] == 1, "selection_reason"] += "; same catchment"
    candidates.loc[candidates["relationship_rank"] == 2, "selection_reason"] += "; same river"
    # Area similarity is the primary analogue criterion; distance breaks ties.
    candidates = candidates.sort_values(["catchment_area_ratio", "relationship_rank", "distance_km", "station_id"], kind="stable")
    return _finish(target_station_id, "analogue", candidates, config)


def select_peer_groups(
    stations: pd.DataFrame,
    target_station_id: str,
    local_config: Mapping[str, Any],
    analogue_config: Mapping[str, Any],
    series_type: str = "observation",
) -> ContextPeerGroups:
    """Return local and analogue peers without mixing their purposes."""
    local = select_local_peers(
        stations, target_station_id, local_config, series_type
    )
    local_ids = set(local.peers.station_id.astype(str))
    comparable = select_analogue_peers(
        stations,
        target_station_id,
        analogue_config,
        series_type,
        excluded_station_ids=local_ids,
    )
    return ContextPeerGroups(str(target_station_id), local, comparable)


__all__ = [
    "ContextPeerGroups",
    "PeerSelectionResult",
    "select_analogue_peers",
    "select_local_peers",
    "select_peer_groups",
]
