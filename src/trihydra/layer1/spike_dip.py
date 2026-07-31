"""Isolated spike/dip candidate detection for TriHydrA Layer 1.

The check is intentionally limited to one-observation impulses. A legitimate
multi-day flood peak is a turning point too, so turning geometry alone is not
enough. Candidates must show strong immediate recovery, an unusually large
absolute change for that calendar month, and an unusually high robust score.
Results remain *candidates*, not declarations that observations are erroneous.
"""

import numpy as np
import pandas as pd

from src.trihydra.layer1.check_result import make_result
from src.trihydra.layer1.timeseries_validity import get_valid_record, timestamps_to_strings


def _score_scale(values: pd.Series) -> pd.DataFrame:
    """Return raw turning-point evidence on one numeric scale."""
    previous = values.shift(1)
    following = values.shift(-1)
    into = values - previous
    out = following - values
    turning = (
        (into * out < 0)
        & previous.notna()
        & values.notna()
        & following.notna()
    )
    jump = pd.Series(
        np.minimum(into.abs(), out.abs()),
        index=values.index,
    )
    denominator = into.abs() + out.abs()
    recovery = 1 - (following - previous).abs() / denominator.replace(0, np.nan)
    absolute_change = values.diff().abs()
    normal_change = absolute_change.groupby(values.index.month).transform("median")
    high_change = absolute_change.groupby(values.index.month).transform(
        lambda x: x.quantile(0.99)
    )
    positive = absolute_change[absolute_change > 0]
    resolution = float(positive.min()) if not positive.empty else np.nan
    normal_change = normal_change.where(normal_change > 0, resolution)
    high_change = high_change.where(high_change > 0, resolution)
    score = (jump / normal_change).replace([np.inf, -np.inf], np.nan)

    candidates = score[turning].dropna()
    if candidates.empty:
        robust_cutoff = np.nan
    else:
        median = float(candidates.median())
        mad = float((candidates - median).abs().median())
        robust_cutoff = max(
            8.0,
            float(candidates.quantile(0.995)),
            median + 6 * 1.4826 * mad,
        )

    flagged = (
        turning
        & recovery.ge(0.80)
        & jump.ge(high_change)
        & score.ge(robust_cutoff)
    )
    return pd.DataFrame({
        "score": score,
        "jump": jump,
        "recovery": recovery,
        "monthly_high_change": high_change,
        "flagged": flagged.fillna(False),
        "robust_cutoff": robust_cutoff,
    })


def check_spike_dip(series: pd.Series, series_type: str = "unknown") -> dict:
    """Detect highly isolated one-point impulses on raw and log scales.

    Both raw and log evidence is retained, but a tiny low-flow movement cannot
    be selected merely because its relative change is large: selection also
    requires the raw-scale jump to exceed that month’s historical 99th
    percentile of absolute daily change.
    """
    record = get_valid_record(series)
    if record.empty:
        return make_result(
            check="spike_dip",
            flag=False,
            value=None,
            series_type=series_type,
            status="skipped",
            reason_skipped="No valid data found.",
            message="Spike/dip check not calculated: no valid data found.",
            candidate_details=[],
        )

    flow = record.astype(float)
    raw = _score_scale(flow)
    # log1p is undefined below -1. Negative discharge has its own basic check;
    # invalid log positions remain unavailable rather than being coerced.
    log_values = pd.Series(
        np.where(flow >= 0, np.log1p(flow), np.nan),
        index=flow.index,
    )
    log = _score_scale(log_values)
    # Raw absolute plausibility is mandatory; log evidence can strengthen but
    # cannot independently promote a negligible low-flow fluctuation.
    flagged_mask = raw["flagged"] & (
        log["flagged"] | raw["score"].ge(raw["robust_cutoff"])
    )
    timestamps = flow.index[flagged_mask]
    previous = flow.shift(1)
    following = flow.shift(-1)

    details = []
    for timestamp in timestamps:
        into = float(flow.loc[timestamp] - previous.loc[timestamp])
        out = float(following.loc[timestamp] - flow.loc[timestamp])
        details.append({
            "timestamp": str(timestamp),
            "type": "spike" if into > 0 else "dip",
            "previous_value": float(previous.loc[timestamp]),
            "candidate_value": float(flow.loc[timestamp]),
            "next_value": float(following.loc[timestamp]),
            "change_into": into,
            "change_out": out,
            "absolute_two_sided_jump": float(raw.loc[timestamp, "jump"]),
            "monthly_q99_absolute_change": float(
                raw.loc[timestamp, "monthly_high_change"]
            ),
            "recovery_score": float(raw.loc[timestamp, "recovery"]),
            "raw_score": float(raw.loc[timestamp, "score"]),
            "log_score": (
                float(log.loc[timestamp, "score"])
                if pd.notna(log.loc[timestamp, "score"]) else None
            ),
            "interpretation": "isolated one-observation impulse candidate",
        })

    return make_result(
        check="spike_dip",
        flag=bool(len(timestamps)),
        value=int(len(timestamps)),
        flagged_timestamps=timestamps_to_strings(timestamps),
        series_type=series_type,
        finding_status="candidate_detected" if len(timestamps) else "passed",
        message=(
            f"Isolated spike/dip candidates = {len(timestamps)}. Selection "
            "requires immediate recovery >= 0.80, raw jump above the month-specific "
            "99th percentile of absolute change, and a robust score cutoff of at "
            "least 8. Coherent multi-day peaks are not automatically flagged."
        ),
        candidate_details=details,
        selection_rules={
            "minimum_recovery": 0.80,
            "minimum_score": 8.0,
            "absolute_change_reference_quantile": 0.99,
            "score_reference_quantile": 0.995,
        },
    )
