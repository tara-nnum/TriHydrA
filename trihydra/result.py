"""In-memory result returned by the format-independent TriHydrA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from trihydra.io.models import StationData


@dataclass
class TriHydrAResult:
    """All completed assessments for one station, without writing files."""

    station: StationData
    layer1: dict[str, Any] | None = None
    layer1_composite: dict[str, Any] | None = None
    layer2: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    layer3: dict[str, Any] | None = None
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    configuration_used: dict[str, Any] = field(default_factory=dict)

    @property
    def station_id(self) -> str:
        return self.station.station_id

    @property
    def series(self) -> pd.Series:
        """Return the untouched primary series supplied to the pipeline."""
        return self.station.obs


@dataclass
class TriHydrANetworkResult:
    """Independent results for a station network and its Layer 3 context."""

    station_results: dict[str, TriHydrAResult]
    layer3_run: Any
    summary: pd.DataFrame
    series_by_station: dict[str, pd.Series]
    configuration_used: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriHydrABatchResult:
    """Explorable result from file-based or already-loaded batch workflows."""

    manifest: pd.DataFrame
    network: TriHydrANetworkResult | None
    output_directory: Path | None = None

    @property
    def station_results(self) -> dict[str, TriHydrAResult]:
        """Return completed station results, indexed by station ID."""
        return {} if self.network is None else self.network.station_results

    @property
    def summary(self) -> pd.DataFrame:
        """Return the combined scientific summary for completed stations."""
        return pd.DataFrame() if self.network is None else self.network.summary


__all__ = ["TriHydrABatchResult", "TriHydrANetworkResult", "TriHydrAResult"]
