"""Canonical data containers shared by TriHydrA ingestion adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class SourceProvenance:
    """Describe exactly how one external series was decoded."""

    path: Path
    format: str
    variable: str
    station_coordinate: str
    time_coordinate: str
    unit: str
    transformations: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class StationData:
    """Raw station series plus optional model, metadata, and provenance."""

    station_id: str
    obs: pd.Series
    unit: str
    obs_provenance: SourceProvenance
    ml: Optional[pd.Series] = None
    ml_provenance: Optional[SourceProvenance] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_raw_preservation(self) -> None:
        """Validate the canonical boundary without cleaning the series."""
        if not isinstance(self.obs, pd.Series):
            raise TypeError("obs must be a pandas Series.")
        if self.obs.index.has_duplicates:
            raise ValueError("Raw OBS contains duplicate timestamps.")
        if self.ml is not None and not isinstance(self.ml, pd.Series):
            raise TypeError("ml must be a pandas Series when supplied.")


__all__ = ["SourceProvenance", "StationData"]
