"""Autocorrelation signature calculation for TriHydrA Layer 2.

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

def calculate_autocorrelation_signatures(
    series: pd.Series,
    lags: Iterable[int] = (1, 2, 3, 7, 14, 30),
    maximum_decay_lag: int = 90,
) -> SignatureResult:
    """
    Calculate discharge autocorrelation and persistence metrics.

    Parameters
    ----------
    series
        Discharge time series.

    lags
        Specific lags for which autocorrelation should be returned.

    maximum_decay_lag
        Maximum lag examined when estimating decorrelation time.

    Returns
    -------
    SignatureResult
        Metrics include:
        - autocorrelation at requested lags,
        - decorrelation lag,
        - first non-positive autocorrelation lag,
        - integral correlation time,
        - persistence ratio AC7 / AC1.

    Notes
    -----
    The decorrelation lag is the first lag where autocorrelation is
    less than or equal to 1/e.

    Integral correlation time is calculated using the initial positive
    sequence:

        1 + 2 * sum(ACF[k])

    until the first non-positive autocorrelation.
    """
    requested_lags = sorted(
        {
            int(lag)
            for lag in lags
            if int(lag) > 0
        }
    )

    if maximum_decay_lag < 1:
        raise ValueError(
            "maximum_decay_lag must be at least 1."
        )

    discharge = _prepare_discharge_series(series)
    valid = discharge.dropna()

    warnings: list[str] = []

    if len(valid) < 3:
        warnings.append(
            "At least three valid values are required for "
            "autocorrelation calculation."
        )

    all_lags = sorted(
        set(
            requested_lags
            + list(range(1, maximum_decay_lag + 1))
        )
    )

    acf_rows: list[dict[str, Any]] = []

    for lag in all_lags:
        if len(discharge.dropna()) <= lag:
            acf = np.nan
            valid_pair_count = 0
        else:
            current = discharge
            lagged = discharge.shift(lag)

            paired = pd.concat(
                [
                    current.rename("current"),
                    lagged.rename("lagged"),
                ],
                axis=1,
            ).dropna()

            valid_pair_count = int(len(paired))

            if valid_pair_count < 3:
                acf = np.nan
            elif (
                paired["current"].std(ddof=1) == 0
                or paired["lagged"].std(ddof=1) == 0
            ):
                acf = np.nan
            else:
                acf = float(
                    paired["current"].corr(
                        paired["lagged"]
                    )
                )

        acf_rows.append(
            {
                "lag": lag,
                "autocorrelation": acf,
                "valid_pair_count": valid_pair_count,
            }
        )

    acf_table = pd.DataFrame(acf_rows)

    requested_acf = {
        f"autocorrelation_lag_{lag}": (
            float(
                acf_table.loc[
                    acf_table["lag"] == lag,
                    "autocorrelation",
                ].iloc[0]
            )
            if (
                not acf_table.loc[
                    acf_table["lag"] == lag,
                    "autocorrelation",
                ].empty
                and pd.notna(
                    acf_table.loc[
                        acf_table["lag"] == lag,
                        "autocorrelation",
                    ].iloc[0]
                )
            )
            else np.nan
        )
        for lag in requested_lags
    }

    decay_table = acf_table.loc[
        acf_table["lag"] <= maximum_decay_lag
    ].copy()

    one_over_e = 1 / np.e

    decorrelation_candidates = decay_table.loc[
        decay_table["autocorrelation"] <= one_over_e,
        "lag",
    ]

    if decorrelation_candidates.empty:
        decorrelation_lag = np.nan
    else:
        decorrelation_lag = int(
            decorrelation_candidates.iloc[0]
        )

    non_positive_candidates = decay_table.loc[
        decay_table["autocorrelation"] <= 0,
        "lag",
    ]

    if non_positive_candidates.empty:
        first_non_positive_lag = np.nan
    else:
        first_non_positive_lag = int(
            non_positive_candidates.iloc[0]
        )

    positive_acf_values: list[float] = []

    for _, row in decay_table.sort_values("lag").iterrows():
        acf_value = row["autocorrelation"]

        if pd.isna(acf_value):
            break

        if acf_value <= 0:
            break

        positive_acf_values.append(float(acf_value))

    integral_correlation_time = (
        float(1 + 2 * np.sum(positive_acf_values))
        if positive_acf_values
        else np.nan
    )

    ac1 = requested_acf.get(
        "autocorrelation_lag_1",
        np.nan,
    )
    ac7 = requested_acf.get(
        "autocorrelation_lag_7",
        np.nan,
    )

    metrics = {
        **requested_acf,
        "decorrelation_threshold": float(one_over_e),
        "decorrelation_lag": decorrelation_lag,
        "first_non_positive_autocorrelation_lag": (
            first_non_positive_lag
        ),
        "integral_correlation_time": (
            integral_correlation_time
        ),
        "acf_7_to_acf_1_ratio": _safe_divide(
            ac7,
            ac1,
        ),
    }

    status = _result_status(
        valid_count=len(valid),
        minimum_required=3,
        warnings=warnings,
    )

    return SignatureResult(
        status=status,
        metrics=metrics,
        tables={
            "autocorrelation_function": acf_table,
        },
        metadata=_basic_metadata(discharge),
        warnings=warnings,
    )
