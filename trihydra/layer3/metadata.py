"""Read and validate station context metadata for Layer 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from trihydra.layer3.climate import ClimateLookup


REQUIRED_CONTEXT_COLUMNS = (
    "station_id",
    "longitude",
    "latitude",
    "river_name",
    "catchment_name",
    "catchment_area_km2",
    "series_type",
)

SERIES_TYPE_ALIASES = {
    "obs": "observation",
    "observed": "observation",
    "observation": "observation",
    "sim": "simulation",
    "simulated": "simulation",
    "simulation": "simulation",
    "ml": "simulation",
    "model": "simulation",
}


@dataclass(frozen=True)
class ContextValidationResult:
    """Usable metadata plus plain-language dataset and row-level issues."""

    stations: pd.DataFrame
    rejected_rows: pd.DataFrame
    messages: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        return not self.stations.empty


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def read_context_metadata(
    path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> ContextValidationResult:
    """Read the context CSV and retain valid rows without stopping other layers."""
    metadata_config = dict(config or {})
    required = tuple(metadata_config.get("required_columns", REQUIRED_CONTEXT_COLUMNS))
    data = pd.read_csv(path)
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        return ContextValidationResult(
            stations=pd.DataFrame(columns=required),
            rejected_rows=data,
            messages=("Layer 3 not assessed: missing columns: " + ", ".join(missing_columns),),
        )

    data = data.loc[:, list(required)].copy()
    data["source_row"] = data.index + 2  # Header is line 1 in the CSV.
    for column in ("station_id", "river_name", "catchment_name", "series_type"):
        data[column] = _clean_text(data[column])
    data["series_type"] = data["series_type"].str.lower().map(SERIES_TYPE_ALIASES)
    for column in ("longitude", "latitude", "catchment_area_km2"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    reasons = pd.Series("", index=data.index, dtype="string")

    def reject(mask: pd.Series, message: str) -> None:
        existing = reasons.loc[mask]
        reasons.loc[mask] = existing.where(existing == "", existing + "; ") + message

    reject(data["station_id"].isna() | (data["station_id"] == ""), "missing station_id")
    reject(data["station_id"].duplicated(keep=False), "duplicate station_id")
    reject(~data["latitude"].between(-90, 90), "invalid latitude")
    reject(~data["longitude"].between(-180, 180), "invalid longitude")
    reject(data["catchment_area_km2"].isna() | (data["catchment_area_km2"] <= 0), "invalid catchment area")
    reject(data["series_type"].isna(), "unknown series_type")

    rejected = data.loc[reasons != ""].copy()
    rejected["rejection_reason"] = reasons.loc[reasons != ""]
    stations = data.loc[reasons == ""].copy().reset_index(drop=True)

    messages = [f"Accepted {len(stations)} of {len(data)} metadata rows."]
    if not rejected.empty:
        messages.append(f"Rejected {len(rejected)} row(s); Layer 1 and Layer 2 may still run.")
    return ContextValidationResult(stations, rejected.reset_index(drop=True), tuple(messages))


def attach_climate_context(
    stations: pd.DataFrame,
    raster_path: str | Path,
    legend_path: str | Path,
) -> pd.DataFrame:
    """Add the climate class at each valid gauge coordinate."""
    enriched = stations.copy()
    with ClimateLookup(raster_path, legend_path) as lookup:
        climates = [
            lookup.lookup(row.latitude, row.longitude)
            for row in enriched.itertuples(index=False)
        ]
    enriched["climate_id"] = [item.climate_id for item in climates]
    enriched["climate_code"] = [item.climate_code for item in climates]
    enriched["climate_description"] = [item.climate_description for item in climates]
    enriched["climate_lookup_status"] = [item.lookup_status for item in climates]
    return enriched
