"""
layer1.py

The single entry point for Layer 1: run_layer1(obs_series, sim_series, ...).

Orchestrates:
  1. trihydra.plotting.diagnostics.run_layer1_diagnostics    (all 10 checks, tables)
  2. trihydra.plotting.visualisation.generate_layer1_visuals (every plot, shown + saved)

This file deliberately contains no check logic and no plotting logic of
its own -- it only calls the two files that already own those
responsibilities, so there is exactly one place each kind of work
happens. If either of those files' internals change, this file does
not need to.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ----------------------------------------------------------------
# Path setup. This file lives at:
#   <project_root>/src/trihydra/layer1/layer1.py
# -- the same depth below trihydra/ as trihydra/plotting/, so
# PROJECT_ROOT (the folder that CONTAINS "src", needed on sys.path for
# "from src.trihydra..." imports to resolve) is 4 parents up.
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trihydra.plotting.diagnostics import run_layer1_diagnostics
from src.trihydra.plotting.visualisation import generate_layer1_visuals


def run_layer1(
    obs_series: pd.Series,
    sim_series: Optional[pd.Series] = None,
    station_id: str = "station",
    model_name: str = "AIFL",
    show: bool = True,
    output_root: Optional[Path] = None,
) -> dict:
    """
    Run all 10 Layer 1 checks on `obs_series` (and, if given,
    `sim_series`, tagged `model_name` -- not just "sim", since future
    work may evaluate more than one model), display every plot inline
    and save it, and return the full diagnostics dict for further
    inspection in the same notebook cell.

    show=True (the default here) displays each plot as it's built --
    the right default for interactive notebook use. Pass show=False
    for a silent batch run across many stations.

    Saves to: io/output/<station_id>/layer1/

    Returns the same dict as diagnostics.run_layer1_diagnostics
    (eda_summary, summary_all, summary_flagged, raw_results), with one
    extra key, "output_path", pointing at the folder everything was
    saved into.

    Example
    -------
        l1 = run_layer1(obs, sim, station_id="GRDC_4123300", model_name="AIFL")
        l1["summary_flagged"]
    """
    diagnostics = run_layer1_diagnostics(obs_series, sim_series, model_name=model_name)

    output_path = generate_layer1_visuals(
        obs_series,
        sim_series,
        diagnostics,
        station_id=station_id,
        model_name=model_name,
        output_root=output_root,
        show=show,
    )

    diagnostics["output_path"] = output_path
    return diagnostics


if __name__ == "__main__":
    print(
        "This module is meant to be imported. Call "
        "run_layer1(obs_series, sim_series, station_id=..., model_name=...)."
    )
