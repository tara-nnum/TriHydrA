"""
layer3.py

The single entry point for Layer 3: run_layer3(obs_series, station_id, ...).

Layer 3 is deliberately OBS-only. The gauge-network NetCDF used to
source context-candidate discharge only has observed data -- there is
no simulated/model discharge for any gauge except the handful with
their own full CSVs -- so there is no model comparison Layer 3 could
meaningfully make. Model evaluation is Layer 2's job; Layer 3 asks a
different question: does the target's own observed record get
corroborated by nearby, independently-observed gauges.

Orchestrates:
  1. layer3.gauge_network       (candidate selection, no discharge needed)
  2. layer3.nc_loader           (candidate discharge, the only current
                                  source for gauges beyond the target's own)
  3. layer3.discharge_comparison (target-vs-candidate metrics)
  4. trihydra.plotting.diagnostics.run_layer3_diagnostics (tables)
  5. trihydra.plotting.visualisation.generate_layer3_visuals (comparison plot)
  6. trihydra.plotting.mapviz.generate_layer3_maps (context maps)

This file deliberately contains no candidate-finding, discharge-
loading, comparison-metric, table, or plotting logic of its own -- it
only calls the modules that already own those responsibilities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ----------------------------------------------------------------
# Path setup. This file lives at:
#   <project_root>/src/trihydra/layer3/layer3.py
# -- the same depth below trihydra/ as trihydra/plotting/, so
# PROJECT_ROOT (the folder that CONTAINS "src") is 4 parents up.
# ----------------------------------------------------------------
TRIHYDRA_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trihydra.layer3.gauge_network import load_gauge_network, find_context_candidates
from src.trihydra.layer3.nc_loader import (
    open_discharge_dataset,
    detect_discharge_variable_and_dims,
    load_discharge_by_gauge_id,
)
from src.trihydra.layer3.discharge_comparison import compare_target_with_candidates
from src.trihydra.plotting.diagnostics import run_layer3_diagnostics
from src.trihydra.plotting.visualisation import generate_layer3_visuals
from src.trihydra.plotting.mapviz import generate_layer3_maps

# ----------------------------------------------------------------
# Network reference files: these describe the WHOLE gauge network
# (not one station), so they live under io/input/ by convention,
# alongside wherever the future input-loading script ends up.
# ----------------------------------------------------------------
IO_INPUT_ROOT = TRIHYDRA_DIR / "io" / "input"
DEFAULT_OUTLETS_PATH = IO_INPUT_ROOT / "outlets_all_systems.csv"
DEFAULT_STATIC_PATH = IO_INPUT_ROOT / "static_attributes_filtered.csv"
DEFAULT_NC_PATH = IO_INPUT_ROOT / "caravan_observations.nc"


def run_layer3(
    obs_series: pd.Series,
    station_id: str,
    show: bool = True,
    output_root: Optional[Path] = None,
    maximum_candidates: int = 10,
    comparison_period: Optional[tuple] = None,
    outlets_path: Optional[Path] = None,
    static_path: Optional[Path] = None,
    nc_path: Optional[Path] = None,
    include_world_map: bool = False,
) -> dict:
    """
    Find this station's nearby-gauge context, compare its observed
    record against each candidate's observed record, display the
    comparison plot and context map inline, and return the full
    diagnostics dict.

    `obs_series` should be the same trusted, already-cleaned observed
    series used in Layer 1/Layer 2 (not re-extracted from the NetCDF)
    -- the NetCDF is used here only as the data source for CANDIDATE
    gauges, which have no other source available yet.

    show=True (the default here) displays the comparison plot and map
    inline -- the right default for interactive notebook use. Pass
    show=False for a silent batch run across many stations.

    Saves to: io/output/<station_id>/layer3/

    Returns a dict with:
      context_summary, comparison_table, interpretation  (tables, see
        diagnostics.run_layer3_diagnostics)
      candidate_result  (the raw result from find_context_candidates,
        kept in case the caller wants the full candidate table)
      output_path
    """
    outlets_path = Path(outlets_path) if outlets_path is not None else DEFAULT_OUTLETS_PATH
    static_path = Path(static_path) if static_path is not None else DEFAULT_STATIC_PATH
    nc_path = Path(nc_path) if nc_path is not None else DEFAULT_NC_PATH

    meta = load_gauge_network(outlets_path, static_path)
    candidate_result = find_context_candidates(meta, station_id, maximum_candidates=maximum_candidates)

    candidates = candidate_result["candidates"]
    candidate_series: dict[str, pd.Series] = {}
    comparison_table = pd.DataFrame()

    if not candidates.empty:
        ds = open_discharge_dataset(nc_path)
        variable_name, time_dim, station_dim = detect_discharge_variable_and_dims(ds)

        def _candidate_loader(gauge_id: str) -> pd.Series:
            series = load_discharge_by_gauge_id(gauge_id, meta, ds, variable_name, time_dim, station_dim)
            candidate_series[gauge_id] = series
            return series

        comparison_table = compare_target_with_candidates(obs_series, candidates, _candidate_loader)

    diagnostics = run_layer3_diagnostics(candidate_result, comparison_table)

    visuals_path = generate_layer3_visuals(
        obs_series, candidate_series, diagnostics,
        station_id=station_id, output_root=output_root, show=show, period=comparison_period,
    )
    maps_path = generate_layer3_maps(
        meta, candidate_result,
        station_id=station_id, output_root=output_root, show=show, include_world_map=include_world_map,
    )

    diagnostics["candidate_result"] = candidate_result
    diagnostics["output_path"] = visuals_path
    diagnostics["map_paths"] = maps_path
    return diagnostics


if __name__ == "__main__":
    print(
        "This module is meant to be imported. Call "
        "run_layer3(obs_series, station_id=...)."
    )
