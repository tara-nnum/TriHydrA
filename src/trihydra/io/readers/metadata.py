"""Readers for Caravan/GloFAS station and catchment metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AREA_COLUMNS = (
    "static_area_km2",
    "drainage_area_provided",
    "DrainingArea.km2.Provider",
    "area_MERIT_1min",
    "DrainingArea.km2.LDD",
)


def _first_positive(row: pd.Series) -> float:
    for column in AREA_COLUMNS:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value) and value > 0:
            return float(value)
    return np.nan


def load_gauge_metadata(
    outlets_path: str | Path, static_attributes_path: str | Path
) -> pd.DataFrame:
    """Merge station/network metadata without performing candidate selection."""
    outlets = pd.read_csv(outlets_path).copy()
    static = pd.read_csv(
        static_attributes_path, usecols=["basin", "area"]
    ).rename(columns={"basin": "gauge_id", "area": "static_area_km2"})
    metadata = outlets.merge(static, on="gauge_id", how="left")
    metadata["area_km2"] = metadata.apply(_first_positive, axis=1)
    metadata["StationLat"] = pd.to_numeric(metadata["StationLat"], errors="coerce")
    metadata["StationLon"] = pd.to_numeric(metadata["StationLon"], errors="coerce")
    metadata = metadata.drop_duplicates("gauge_id", keep="first")
    return metadata.reset_index(drop=True)


__all__ = ["load_gauge_metadata", "AREA_COLUMNS"]
