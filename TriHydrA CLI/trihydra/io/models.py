"""Canonical data containers shared by TriHydrA ingestion adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class SourceProvenance:
    """Describe where a series came from, including in-memory inputs."""

    path: Path | None = None
    format: str = "memory"
    variable: str | None = None
    station_coordinate: str | None = None
    time_coordinate: str | None = None
    unit: str = "source units"
    transformations: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def in_memory(cls, *, unit: str, label: str = "pandas.Series") -> "SourceProvenance":
        """Create provenance for a series supplied directly by Python code."""
        return cls(format="memory", unit=unit, details={"source": label})


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
        if not isinstance(self.obs.index, pd.DatetimeIndex):
            raise TypeError("obs must use a pandas DatetimeIndex.")
        # Duplicate and irregular timestamp order are Layer 1 evidence. Keep
        # them unchanged here so the corresponding diagnostics can report them.
        if self.ml is not None and not isinstance(self.ml, pd.Series):
            raise TypeError("ml must be a pandas Series when supplied.")
        if self.ml is not None:
            if not isinstance(self.ml.index, pd.DatetimeIndex):
                raise TypeError("ml must use a pandas DatetimeIndex.")
            # Candidate timestamps are preserved for the same reason as OBS.
            if self.ml_provenance is None:
                raise ValueError("ml_provenance is required when ml is supplied.")


__all__ = ["SourceProvenance", "StationData"]
