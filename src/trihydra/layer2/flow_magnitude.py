"""Flow magnitude signature calculation for TriHydrA Layer 2.

This module owns this complete single-series signature calculation. It performs
no OBS–ML comparison and writes no files. Temporary imputation and provenance
are handled before the calculator is called.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.trihydra.layer2.signature_result import SignatureResult
from src.trihydra.layer2.signature_utils import *

def calculate_flow_magnitude_signatures(
    series: pd.Series,
) -> SignatureResult:
    """
    Calculate basic discharge-magnitude signatures.

    Returns
    -------
    SignatureResult
        Metrics include:
        - mean_flow
        - median_flow
        - minimum_flow
        - maximum_flow
        - standard_deviation
        - coefficient_of_variation
        - q05
        - q10
        - q25
        - q75
        - q90
        - q95
    """
    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    if valid.empty:
        warnings.append("No valid discharge values are available.")

    metrics = {
        "mean_flow": (
            float(valid.mean()) if not valid.empty else np.nan
        ),
        "median_flow": (
            float(valid.median()) if not valid.empty else np.nan
        ),
        "minimum_flow": (
            float(valid.min()) if not valid.empty else np.nan
        ),
        "maximum_flow": (
            float(valid.max()) if not valid.empty else np.nan
        ),
        "standard_deviation": (
            float(valid.std(ddof=1))
            if len(valid) >= 2
            else np.nan
        ),
        "coefficient_of_variation": _calculate_cv(valid),
        "q05": (
            float(valid.quantile(0.05))
            if not valid.empty
            else np.nan
        ),
        "q10": (
            float(valid.quantile(0.10))
            if not valid.empty
            else np.nan
        ),
        "q25": (
            float(valid.quantile(0.25))
            if not valid.empty
            else np.nan
        ),
        "q75": (
            float(valid.quantile(0.75))
            if not valid.empty
            else np.nan
        ),
        "q90": (
            float(valid.quantile(0.90))
            if not valid.empty
            else np.nan
        ),
        "q95": (
            float(valid.quantile(0.95))
            if not valid.empty
            else np.nan
        ),
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=1,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={},
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
