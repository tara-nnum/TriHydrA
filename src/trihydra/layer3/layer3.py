"""Calculation-oriented Layer 3 orchestration.

Layer 3 assesses whether a target observed hydrograph is corroborated by
independent Caravan observations from hydrologically relevant context gauges.
All source decoding belongs to :mod:`trihydra.io`; this module accepts only
already-loaded pandas series and metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.trihydra.layer3.gauge_network import find_context_candidates
from src.trihydra.layer3.discharge_comparison import (
    compare_target_with_candidates,
)
from src.trihydra.plotting.diagnostics import run_layer3_diagnostics
from src.trihydra.plotting.visualisation import generate_layer3_visuals
from src.trihydra.plotting.mapviz import generate_layer3_maps


def run_layer3(
    obs_series: pd.Series,
    station_id: str,
    network_metadata: pd.DataFrame,
    candidate_observations: dict[str, pd.Series],
    show: bool = True,
    output_root: Optional[Path] = None,
    maximum_candidates: int = 10,
    comparison_period: Optional[tuple] = None,
    include_world_map: bool = False,
) -> dict:
    """Calculate and present nearby-gauge plausibility evidence.

    Parameters are source-agnostic. ``network_metadata`` must already contain
    station IDs, catchment/river labels, coordinates and ``area_km2``.
    ``candidate_observations`` maps station IDs to raw observed series.
    Missing candidate mappings are skipped rather than loaded here.
    """
    candidate_result = find_context_candidates(
        network_metadata,
        station_id,
        maximum_candidates=maximum_candidates,
    )
    comparison_table = compare_target_with_candidates(
        obs_series,
        candidate_result["candidates"],
        candidate_observations,
    )
    diagnostics = run_layer3_diagnostics(
        candidate_result, comparison_table
    )
    visuals_path = generate_layer3_visuals(
        obs_series,
        candidate_observations,
        diagnostics,
        station_id=station_id,
        output_root=output_root,
        show=show,
        period=comparison_period,
    )
    maps_path = generate_layer3_maps(
        network_metadata,
        candidate_result,
        station_id=station_id,
        output_root=output_root,
        show=show,
        include_world_map=include_world_map,
    )
    diagnostics.update(
        {
            "candidate_result": candidate_result,
            "output_path": visuals_path,
            "map_paths": maps_path,
        }
    )
    return diagnostics


__all__ = ["run_layer3"]
