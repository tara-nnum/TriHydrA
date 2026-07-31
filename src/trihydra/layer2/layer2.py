"""
layer2.py

The single entry point for Layer 2: run_layer2(obs_series, ml_series, ...).

Orchestrates:
  1. trihydra.plotting.diagnostics.run_layer2_diagnostics    (all 13 signature
     checks for OBS, plus model comparison only when a model is supplied)
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
from typing import Any, Optional

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
    ml_series: Optional[pd.Series] = None,
    station_id: str = "station",
    model_name: str = "AIFL",
    show: bool = True,
    output_root: Optional[Path] = None,
    fill_method: str = "seasonal_climatology",
    layer1_obs_profile: Optional[dict] = None,
    discharge_unit: str = "source units",
    fill_window_days: int = 15,
    fill_min_samples: int = 5,
    signature_kwargs: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Run all 13 Layer 2 signature checks. When `ml_series` is supplied,
    OBS and model are restricted to the same period before comparison.
    When it is omitted, Layer 2 runs OBS-only without a comparison.

    Missing values are filled only in temporary analysis copies. The
    returned diagnostics include the original-missing counts, fill method,
    and a timestamp-level imputation log; the caller's Series is unchanged.

    show=True (the default here) displays each plot as it's built --
    the right default for interactive notebook use. Pass show=False
    for a silent batch run across many stations.

    Saves to: io/output/<station_id>/layer2/

    Returns the same dict as diagnostics.run_layer2_diagnostics
    (13-row signature_comparison, optional detailed comparison,
    percentile diagnostics, imputation provenance, and raw results), with one
    extra key, "output_path", pointing at the folder everything was
    saved into.

    Example
    -------
        l2 = run_layer2(obs, ml, station_id="GRDC_4123300", model_name="AIFL")
        l2["signature_comparison"]
    """
    diagnostics = run_layer2_diagnostics(
        obs_series,
        ml_series,
        model_name=model_name,
        fill_method=fill_method,
        layer1_obs_profile=layer1_obs_profile,
        discharge_unit=discharge_unit,
        fill_window_days=fill_window_days,
        fill_min_samples=fill_min_samples,
        signature_kwargs=signature_kwargs,
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
