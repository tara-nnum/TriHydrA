"""Compare hydrological behaviour with comparable-catchment peers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from trihydra.layer3.evidence import StationContextEvidence
from trihydra.layer3.local_comparison import ContextCheckResult


@dataclass(frozen=True)
class AnalogueContextResult:
    """Independently inspectable comparable-catchment checks."""

    target_station_id: str
    flashiness: ContextCheckResult
    baseflow: ContextCheckResult
    zero_flow: ContextCheckResult
    seasonality_shape: ContextCheckResult
    event_shape: ContextCheckResult
    event_time_to_peak: ContextCheckResult
    event_duration: ContextCheckResult


def _finite(value) -> bool:
    return value is not None and np.isfinite(value)


def _median_signature(evidence: StationContextEvidence, column: str) -> float | None:
    if column not in evidence.annual_signatures:
        return None
    values = pd.to_numeric(evidence.annual_signatures[column], errors="coerce").dropna()
    return None if values.empty else float(values.median())


def _ratio_similarity(first: float, second: float) -> float:
    """Similarity for non-negative values where scale is not naturally bounded."""
    first, second = max(float(first), 0.0), max(float(second), 0.0)
    if first == 0 and second == 0:
        return 1.0
    if first == 0 or second == 0:
        return 0.0
    return min(first, second) / max(first, second)


def _bounded_similarity(first: float, second: float) -> float:
    """Similarity for values already expressed on the zero-to-one scale."""
    return float(np.clip(1.0 - abs(float(first) - float(second)), 0.0, 1.0))


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float | None:
    valid = np.isfinite(first) & np.isfinite(second)
    if not valid.any():
        return None
    first, second = first[valid], second[valid]
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _finish_similarity_check(
    check: str,
    rows: list[dict],
    minimum_similarity: float,
) -> ContextCheckResult:
    details = pd.DataFrame(rows)
    if details.empty or "status" not in details:
        return ContextCheckResult(check, "not_assessed", None, 0, 0, details, f"{check} not assessed: no comparable analogue evidence.")
    assessed = details.loc[details.status.isin(["supported", "not_supported"])]
    if assessed.empty:
        return ContextCheckResult(check, "not_assessed", None, 0, 0, details, f"{check} not assessed: no comparable analogue evidence.")
    supporting = int((assessed.status == "supported").sum())
    similarity = float(pd.to_numeric(assessed["similarity"]).median())
    status = "supported" if similarity >= minimum_similarity else "not_supported"
    return ContextCheckResult(
        check, status, similarity, len(assessed), supporting, details,
        f"Median pairwise {check.lower()} similarity across {len(assessed)} peer(s) is {similarity:.3f}.",
    )


def _scalar_similarity_check(
    check: str,
    target_value: float | None,
    peer_values: Mapping[str, float | None],
    similarity: Callable[[float, float], float],
    minimum_similarity: float,
) -> ContextCheckResult:
    if not _finite(target_value):
        return ContextCheckResult(check, "not_assessed", None, 0, 0, pd.DataFrame(), f"{check} not assessed: target evidence is unavailable.")
    rows = []
    for peer_id, peer_value in peer_values.items():
        if not _finite(peer_value):
            rows.append({"peer_station_id": peer_id, "status": "not_assessed", "reason": "Peer evidence is unavailable."})
            continue
        score = similarity(float(target_value), float(peer_value))
        rows.append({
            "peer_station_id": peer_id,
            "target_value": float(target_value),
            "peer_value": float(peer_value),
            "similarity": score,
            "minimum_similarity": minimum_similarity,
            "status": "supported" if score >= minimum_similarity else "not_supported",
            "reason": "",
        })
    return _finish_similarity_check(check, rows, minimum_similarity)


def _seasonality_check(
    target: StationContextEvidence,
    peers: Mapping[str, StationContextEvidence],
    minimum_points: int,
    minimum_similarity: float,
) -> ContextCheckResult:
    rows = []
    target_values = target.seasonality_profile.reindex(range(1, 13)).to_numpy(float)
    for peer_id, peer in peers.items():
        peer_values = peer.seasonality_profile.reindex(range(1, 13)).to_numpy(float)
        valid_count = int((np.isfinite(target_values) & np.isfinite(peer_values)).sum())
        score = _cosine_similarity(target_values, peer_values) if valid_count >= minimum_points else None
        if score is None:
            rows.append({"peer_station_id": peer_id, "status": "not_assessed", "comparable_months": valid_count, "reason": "Too few comparable months."})
        else:
            rows.append({
                "peer_station_id": peer_id, "comparable_months": valid_count,
                "similarity": score,
                "minimum_similarity": minimum_similarity,
                "status": "supported" if score >= minimum_similarity else "not_supported",
                "reason": "",
            })
    return _finish_similarity_check("Seasonality shape", rows, minimum_similarity)


def _resampled_event_curve(curve: pd.Series, points: int = 101) -> np.ndarray | None:
    values = pd.to_numeric(curve, errors="coerce").dropna().to_numpy(float)
    if len(values) < 2:
        return None
    old_axis = np.linspace(0.0, 1.0, len(values))
    return np.interp(np.linspace(0.0, 1.0, points), old_axis, values)


def _event_shape_check(target, peers, minimum_similarity):
    """Compare both the event pattern and its actual peak magnitude."""
    target_curve = _resampled_event_curve(target.representative_event_curve)
    if target_curve is None:
        return ContextCheckResult("Representative event shape", "not_assessed", None, 0, 0, pd.DataFrame(), "Representative event shape not assessed: target event is unavailable.")
    rows = []
    for peer_id, peer in peers.items():
        peer_curve = _resampled_event_curve(peer.representative_event_curve)
        shape_score = None if peer_curve is None else _cosine_similarity(target_curve, peer_curve)
        target_peak = target.representative_event_metrics.get("peak_flow")
        peer_peak = peer.representative_event_metrics.get("peak_flow")
        magnitude_score = (
            _ratio_similarity(target_peak, peer_peak)
            if _finite(target_peak) and _finite(peer_peak) else None
        )
        if shape_score is None or magnitude_score is None:
            rows.append({"peer_station_id": peer_id, "status": "not_assessed", "reason": "Peer event is unavailable."})
        else:
            # Both aspects matter: a matching outline cannot compensate for a
            # substantially different real-world discharge magnitude.
            score = float(shape_score * magnitude_score)
            rows.append({
                "peer_station_id": peer_id,
                "shape_similarity": shape_score,
                "peak_magnitude_similarity": magnitude_score,
                "similarity": score,
                "minimum_similarity": minimum_similarity,
                "status": "supported" if score >= minimum_similarity else "not_supported",
                "reason": "",
            })
    return _finish_similarity_check(
        "Representative event shape and magnitude", rows, minimum_similarity
    )


def _event_metric_check(target, peers, metric, check, tolerance):
    target_value = target.representative_event_metrics.get(metric)
    if not _finite(target_value):
        return ContextCheckResult(check, "not_assessed", None, 0, 0, pd.DataFrame(), f"{check} not assessed: target event metric is unavailable.")
    rows = []
    for peer_id, peer in peers.items():
        peer_value = peer.representative_event_metrics.get(metric)
        if not _finite(peer_value):
            rows.append({"peer_station_id": peer_id, "status": "not_assessed", "reason": "Peer event metric is unavailable."})
            continue
        difference = abs(float(peer_value) - float(target_value))
        # A zero-day tolerance requests exact agreement. Positive tolerances
        # retain the continuous display scale where the cutoff equals 0.50.
        similarity = (
            1.0 if difference == 0 else 0.0
        ) if tolerance == 0 else float(
            np.clip(1.0 - difference / (2.0 * tolerance), 0.0, 1.0)
        )
        rows.append({
            "peer_station_id": peer_id, "target_value": float(target_value),
            "peer_value": float(peer_value), "absolute_difference_days": difference,
            "tolerance_days": tolerance,
            "similarity": similarity,
            "minimum_similarity": 0.50,
            "status": "supported" if difference <= tolerance else "not_supported",
            "reason": "",
        })
    return _finish_similarity_check(check, rows, 0.50)


def compare_analogue_context(
    target: StationContextEvidence,
    peers: Mapping[str, StationContextEvidence],
    config: Mapping[str, float],
) -> AnalogueContextResult:
    """Compare the target with climate- and scale-compatible catchments."""
    minimum = float(config.get("analogue_similarity_minimum", 0.80))
    flashiness = _scalar_similarity_check(
        "Median annual flashiness", _median_signature(target, "flashiness_index"),
        {key: _median_signature(value, "flashiness_index") for key, value in peers.items()},
        _ratio_similarity, minimum,
    )
    baseflow = _scalar_similarity_check(
        "Median annual baseflow index", _median_signature(target, "baseflow_index"),
        {key: _median_signature(value, "baseflow_index") for key, value in peers.items()},
        _bounded_similarity, minimum,
    )
    zero_flow = _scalar_similarity_check(
        "Zero-flow behaviour", target.zero_flow_ratio,
        {key: value.zero_flow_ratio for key, value in peers.items()},
        _bounded_similarity, minimum,
    )
    seasonality = _seasonality_check(
        target, peers, int(config.get("minimum_profile_points", 6)), minimum
    )
    event_shape = _event_shape_check(target, peers, minimum)
    time_to_peak = _event_metric_check(
        target, peers, "time_to_peak_days", "Representative event time to peak",
        float(config.get("event_time_to_peak_tolerance_days", 5.0)),
    )
    duration = _event_metric_check(
        target, peers, "event_duration_days", "Representative event duration",
        float(config.get("event_duration_tolerance_days", 7.0)),
    )
    return AnalogueContextResult(
        target.station_id, flashiness, baseflow, zero_flow, seasonality,
        event_shape, time_to_peak, duration,
    )


__all__ = ["AnalogueContextResult", "compare_analogue_context"]
