"""Isolated spike/dip candidate detection for TriHydrA Layer 1.

The check is intentionally limited to one-observation impulses. A legitimate
multi-day flood peak is a turning point too, so turning geometry alone is not
enough. Candidates must show strong immediate recovery, an unusually large
absolute change for that calendar month, and an unusually high robust score.
Results remain *candidates*, not declarations that observations are erroneous.
"""

import numpy as np
import pandas as pd

from trihydra.layer1.check_result import make_result
from trihydra.layer1.timeseries_validity import get_valid_record, timestamps_to_strings


def _score_scale(
    values: pd.Series,
    minimum_recovery: float,
    minimum_score: float,
    absolute_change_reference_quantile: float,
    score_reference_quantile: float,
    robust_mad_multiplier: float,
) -> pd.DataFrame:
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
    # Reference distributions represent genuine one-day changes only. A jump
    # across an absent date or duplicate timestamp is not a daily change.
    daily_pair = values.index.to_series().diff().eq(pd.Timedelta(days=1))
    absolute_change = values.diff().abs().where(daily_pair)
    normal_change = absolute_change.groupby(values.index.month).transform("median")
    high_change = absolute_change.groupby(values.index.month).transform(
        lambda x: x.quantile(absolute_change_reference_quantile)
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
            minimum_score,
            float(candidates.quantile(score_reference_quantile)),
            median + robust_mad_multiplier * 1.4826 * mad,
        )

    flagged = (
        turning
        & recovery.ge(minimum_recovery)
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


def check_spike_dip(
    series: pd.Series,
    series_type: str = "unknown",
    minimum_recovery: float = 0.80,
    minimum_score: float = 8.0,
    absolute_change_reference_quantile: float = 0.99,
    score_reference_quantile: float = 0.995,
    robust_mad_multiplier: float = 6.0,
    minimum_outer_change_multiplier: float = 1.0,
) -> dict:
    """Detect isolated impulses using raw-scale magnitude and recovery.

    Selection requires both a station-relative score and a month-specific
    absolute-change threshold, so tiny low-flow movements are not promoted.
    """
    if not 0 <= minimum_recovery <= 1:
        raise ValueError("minimum_recovery must be between 0 and 1")
    if minimum_score < 0:
        raise ValueError("minimum_score must be non-negative")
    for name, value in (
        ("absolute_change_reference_quantile", absolute_change_reference_quantile),
        ("score_reference_quantile", score_reference_quantile),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if robust_mad_multiplier < 0 or minimum_outer_change_multiplier < 0:
        raise ValueError("MAD and outer-change multipliers must be non-negative")
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

    score_kwargs = {
        "minimum_recovery": minimum_recovery,
        "minimum_score": minimum_score,
        "absolute_change_reference_quantile": absolute_change_reference_quantile,
        "score_reference_quantile": score_reference_quantile,
        "robust_mad_multiplier": robust_mad_multiplier,
    }
    flow = record.astype(float)
    raw = _score_scale(flow, **score_kwargs)
    initial_mask = raw["flagged"].copy()

    minus_two = flow.shift(2)
    minus_one = flow.shift(1)
    plus_one = flow.shift(-1)
    plus_two = flow.shift(-2)
    index_series = flow.index.to_series()
    five_daily = (
        index_series.sub(index_series.shift(1)).eq(pd.Timedelta(days=1))
        & index_series.shift(-1).sub(index_series).eq(pd.Timedelta(days=1))
        & index_series.shift(-2).sub(index_series.shift(-1)).eq(pd.Timedelta(days=1))
        & index_series.shift(1).sub(index_series.shift(2)).eq(pd.Timedelta(days=1))
    )
    five_valid = pd.concat(
        [minus_two, minus_one, flow, plus_one, plus_two], axis=1
    ).notna().all(axis=1) & five_daily
    daily_pair = flow.index.to_series().diff().eq(pd.Timedelta(days=1))
    absolute_change = flow.diff().abs().where(daily_pair)
    outer_reference = absolute_change.groupby(flow.index.month).transform("median")
    positive_changes = absolute_change[absolute_change > 0]
    resolution = float(positive_changes.min()) if not positive_changes.empty else np.nan
    outer_reference = outer_reference.where(outer_reference > 0, resolution)
    outer_large_enough = (
        (minus_one - minus_two).abs().ge(
            outer_reference * minimum_outer_change_multiplier
        )
        & (plus_two - plus_one).abs().ge(
            outer_reference * minimum_outer_change_multiplier
        )
    )
    coherent_peak = (
        minus_two.lt(minus_one) & minus_one.lt(flow)
        & flow.gt(plus_one) & plus_one.gt(plus_two)
    )
    coherent_trough = (
        minus_two.gt(minus_one) & minus_one.gt(flow)
        & flow.lt(plus_one) & plus_one.lt(plus_two)
    )
    coherent_veto = five_valid & outer_large_enough & (coherent_peak | coherent_trough)
    context_unavailable = initial_mask & ~five_valid
    rejected_coherent = initial_mask & coherent_veto
    flagged_mask = initial_mask & five_valid & ~coherent_veto
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
            "raw_robust_cutoff": float(raw.loc[timestamp, "robust_cutoff"]),
            "interpretation": "isolated one-observation impulse candidate",
        })

    rejected_details = []
    for timestamp in flow.index[rejected_coherent]:
        rejected_details.append({
            "timestamp": str(timestamp),
            "type": "spike" if flow.loc[timestamp] > minus_one.loc[timestamp] else "dip",
            "reason": "coherent_multiday_turning_pattern",
            "t_minus_2": float(minus_two.loc[timestamp]),
            "t_minus_1": float(minus_one.loc[timestamp]),
            "candidate_value": float(flow.loc[timestamp]),
            "t_plus_1": float(plus_one.loc[timestamp]),
            "t_plus_2": float(plus_two.loc[timestamp]),
        })

    return make_result(
        check="spike_dip",
        flag=bool(len(timestamps)),
        value=int(len(timestamps)),
        flagged_timestamps=timestamps_to_strings(timestamps),
        series_type=series_type,
        finding_status="candidate_detected" if len(timestamps) else "passed",
        message=(
            f"Isolated spike/dip candidates = {len(timestamps)}; coherent five-day "
            f"turning patterns rejected = {int(rejected_coherent.sum())}; candidates "
            f"without complete five-day context rejected = {int(context_unavailable.sum())}. "
            f"Selection requires immediate recovery >= {minimum_recovery:.2f}, raw "
            f"jump above the month-specific {absolute_change_reference_quantile:.3f} "
            f"quantile of absolute change, and robust score >= {minimum_score:g}."
        ),
        candidate_details=details,
        rejected_coherent_patterns=rejected_details,
        rejected_context_unavailable=int(context_unavailable.sum()),
        raw_robust_cutoff=(
            float(raw["robust_cutoff"].iloc[0])
            if len(raw) and pd.notna(raw["robust_cutoff"].iloc[0]) else None
        ),
        selection_rules={
            "minimum_recovery": float(minimum_recovery),
            "minimum_score": float(minimum_score),
            "absolute_change_reference_quantile": float(absolute_change_reference_quantile),
            "score_reference_quantile": float(score_reference_quantile),
            "robust_mad_multiplier": float(robust_mad_multiplier),
            "context_window_days": 5,
            "minimum_outer_change_multiplier": float(minimum_outer_change_multiplier),
            "daily_change_reference": "timestamp pairs exactly one day apart",
        },
    )
