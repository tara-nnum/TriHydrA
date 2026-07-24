"""
layer2.py

The single entry point for Layer 2: run_layer2(obs_series, ml_series, ...).

Orchestrates:
  1. trihydra.plotting.diagnostics.run_layer2_diagnostics    (all 15 signature
     checks for OBS and model, plus the compact/full/flagged/percentile tables)
  2. trihydra.plotting.visualisation.generate_layer2_visuals (every plot,
     shown + saved)

This file deliberately contains no signature-calculation logic and no
plotting logic of its own -- it only calls the two files that already
own those responsibilities, so there is exactly one place each kind of
work happens. If either of those files' internals change, this file
does not need to.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ----------------------------------------------------------------
# Path setup. This file lives at:
#   <project_root>/src/trihydra/layer2/layer2.py
# -- the same depth below trihydra/ as trihydra/plotting/, so
# PROJECT_ROOT (the folder that CONTAINS "src", needed on sys.path for
# "from src.trihydra..." imports to resolve) is 4 parents up.
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trihydra.plotting.diagnostics import run_layer2_diagnostics
from src.trihydra.plotting.visualisation import generate_layer2_visuals


def run_layer2(
    obs_series: pd.Series,
    ml_series: pd.Series,
    station_id: str = "station",
    model_name: str = "AIFL",
    show: bool = True,
    output_root: Optional[Path] = None,
    relative_tolerance_percent: float = 10.0,
) -> dict:
    """
    Run all 15 Layer 2 signature checks on `obs_series` vs. `ml_series`
    (labelled `model_name` -- not just "sim"), display every plot
    inline and save it, and return the full diagnostics dict for
    further inspection in the same notebook cell.

    show=True (the default here) displays each plot as it's built --
    the right default for interactive notebook use. Pass show=False
    for a silent batch run across many stations.

    Saves to: io/output/<station_id>/layer2/

    Returns the same dict as diagnostics.run_layer2_diagnostics
    (compact_comparison, full_comparison, full_comparison_flagged,
    percentile_diagnostics, obs_results, model_results), with one
    extra key, "output_path", pointing at the folder everything was
    saved into.

    Example
    -------
        l2 = run_layer2(obs, ml, station_id="GRDC_4123300", model_name="AIFL")
        l2["full_comparison_flagged"]
    """
    diagnostics = run_layer2_diagnostics(
        obs_series,
        ml_series,
        model_name=model_name,
        relative_tolerance_percent=relative_tolerance_percent,
    )

    output_path = generate_layer2_visuals(
        obs_series,
        ml_series,
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
        "run_layer2(obs_series, ml_series, station_id=..., model_name=...)."
    )
