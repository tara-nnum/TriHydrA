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
    """One required primary series and one optional comparison series.

    Use the neutral ``series1`` and ``series2`` properties when reading the
    inputs. Their configured names and roles describe whether either series
    represents observations, simulations, or another supported source.
    """

    station_id: str
    obs: pd.Series
    unit: str
    obs_provenance: SourceProvenance
    ml: Optional[pd.Series] = None
    ml_provenance: Optional[SourceProvenance] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    series1_name: str = "observation"
    series1_role: str = "observation"
    series2_name: str = "model"
    series2_role: str = "simulation"

    @classmethod
    def from_series(
        cls,
        *,
        station_id: str,
        series1: pd.Series,
        unit: str,
        series1_provenance: SourceProvenance,
        series2: Optional[pd.Series] = None,
        series2_provenance: Optional[SourceProvenance] = None,
        metadata: Optional[dict[str, Any]] = None,
        series1_name: str = "series1",
        series1_role: str = "observation",
        series2_name: str = "series2",
        series2_role: str = "simulation",
    ) -> "StationData":
        """Construct station data using the neutral public terminology."""
        return cls(
            station_id=station_id,
            obs=series1,
            unit=unit,
            obs_provenance=series1_provenance,
            ml=series2,
            ml_provenance=series2_provenance,
            metadata={} if metadata is None else metadata,
            series1_name=series1_name,
            series1_role=series1_role,
            series2_name=series2_name,
            series2_role=series2_role,
        )

    @property
    def series1(self) -> pd.Series:
        """Return the mandatory primary/reference time series."""
        return self.obs

    @property
    def series2(self) -> Optional[pd.Series]:
        """Return the optional comparison time series."""
        return self.ml

    @property
    def series1_provenance(self) -> SourceProvenance:
        """Return provenance for the primary/reference series."""
        return self.obs_provenance

    @property
    def series2_provenance(self) -> Optional[SourceProvenance]:
        """Return provenance for the optional comparison series."""
        return self.ml_provenance

    def validate_raw_preservation(self) -> None:
        """Validate the canonical boundary without cleaning the series."""
        if not isinstance(self.series1, pd.Series):
            raise TypeError("series1 must be a pandas Series.")
        if not isinstance(self.series1.index, pd.DatetimeIndex):
            raise TypeError("series1 must use a pandas DatetimeIndex.")
        # Duplicate and irregular timestamp order are Layer 1 evidence. Keep
        # them unchanged here so the corresponding diagnostics can report them.
        if self.series2 is not None and not isinstance(self.series2, pd.Series):
            raise TypeError("series2 must be a pandas Series when supplied.")
        if self.series2 is not None:
            if not isinstance(self.series2.index, pd.DatetimeIndex):
                raise TypeError("series2 must use a pandas DatetimeIndex.")
            # Secondary timestamps are preserved for the same reason as the
            # primary series.
            if self.series2_provenance is None:
                raise ValueError(
                    "series2_provenance is required when series2 is supplied."
                )


__all__ = ["SourceProvenance", "StationData"]
