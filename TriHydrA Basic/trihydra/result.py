"""In-memory result returned by the format-independent TriHydrA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
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
        """Return the untouched observation series supplied to the pipeline."""
        return self.station.obs


@dataclass
class TriHydrANetworkResult:
    """Independent results for a station network and its Layer 3 context."""

    station_results: dict[str, TriHydrAResult]
    layer3_run: Any
    summary: pd.DataFrame
    series_by_station: dict[str, pd.Series]
    configuration_used: dict[str, Any] = field(default_factory=dict)


__all__ = ["TriHydrANetworkResult", "TriHydrAResult"]
