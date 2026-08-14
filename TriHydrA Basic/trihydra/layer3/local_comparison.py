"""Compare date-based evidence against genuinely local peer gauges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from trihydra.layer3.evidence import StationContextEvidence


@dataclass(frozen=True)
class ContextCheckResult:
    """One transparent Layer 3 check and its peer-level evidence."""

    check: str
    status: str
    similarity_fraction: float | None
    assessed_peer_count: int
    supporting_peer_count: int
    details: pd.DataFrame
    message: str


@dataclass(frozen=True)
class LocalContextResult:
    """Local timing checks kept separate from analogue comparisons."""

    target_station_id: str
    peak_timing: ContextCheckResult
    step_shift_timing: ContextCheckResult
    epoch_behaviour: ContextCheckResult


def _overlap(target: StationContextEvidence, peer: StationContextEvidence):
    starts = [value for value in (target.record_start, peer.record_start) if value is not None]
    ends = [value for value in (target.record_end, peer.record_end) if value is not None]
    if len(starts) < 2 or len(ends) < 2:
        return None, None
    start, end = max(starts), min(ends)
    return (start, end) if start <= end else (None, None)


def _within(dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    return dates[(dates >= start) & (dates <= end)]


def _matched_date_count(
    target_dates: pd.DatetimeIndex,
    peer_dates: pd.DatetimeIndex,
    tolerance_days: float,
) -> int:
    """Pair sorted dates one-to-one within tolerance and return the pair count."""
    if len(target_dates) == 0 or len(peer_dates) == 0:
        return 0
    target_days = np.sort(target_dates.to_numpy(dtype="datetime64[D]").astype("int64"))
    peer_days = np.sort(peer_dates.to_numpy(dtype="datetime64[D]").astype("int64"))
    target_index = peer_index = matched = 0
    while target_index < len(target_days) and peer_index < len(peer_days):
        difference = target_days[target_index] - peer_days[peer_index]
        if abs(difference) <= tolerance_days:
            matched += 1
            target_index += 1
            peer_index += 1
        elif difference < 0:
            target_index += 1
        else:
            peer_index += 1
    return matched


def _symmetric_date_similarity(matched: int, target_count: int, peer_count: int) -> float | None:
    """Return Dice agreement; swapping target and peer gives the same value."""
    total = target_count + peer_count
    return None if total == 0 else (2.0 * matched) / total


def _date_check(
    check: str,
    target: StationContextEvidence,
    peers: Mapping[str, StationContextEvidence],
    field: str,
    tolerance_days: float,
    minimum_match_fraction: float,
) -> ContextCheckResult:
    rows = []
    if not target.availability.get(field, False):
        return ContextCheckResult(check, "not_assessed", None, 0, 0, pd.DataFrame(), f"{check} not assessed: target evidence is unavailable.")
    target_all = getattr(target, field)
    for peer_id, peer in peers.items():
        start, end = _overlap(target, peer)
        if start is None or not peer.availability.get(field, False):
            rows.append({"peer_station_id": peer_id, "status": "not_assessed", "reason": "No common period or peer evidence."})
            continue
        target_dates = _within(target_all, start, end)
        peer_dates = _within(getattr(peer, field), start, end)
        if len(target_dates) == 0 and len(peer_dates) == 0:
            rows.append({"peer_station_id": peer_id, "status": "not_applicable", "reason": "Neither station has dates requiring comparison in the common period."})
            continue
        matched = _matched_date_count(target_dates, peer_dates, tolerance_days)
        similarity = _symmetric_date_similarity(matched, len(target_dates), len(peer_dates))
        supports = similarity >= minimum_match_fraction
        rows.append({
            "peer_station_id": peer_id,
            "status": "supported" if supports else "not_supported",
            "common_start": start,
            "common_end": end,
            "target_date_count": len(target_dates),
            "peer_date_count": len(peer_dates),
            "matched_date_pair_count": matched,
            "unmatched_target_date_count": len(target_dates) - matched,
            "unmatched_peer_date_count": len(peer_dates) - matched,
            "symmetric_similarity": similarity,
            "tolerance_days": tolerance_days,
            "minimum_match_fraction": minimum_match_fraction,
            "reason": "",
        })

    details = pd.DataFrame(rows)
    assessed = details.loc[details.get("status", pd.Series(dtype=str)).isin(["supported", "not_supported"])]
    if assessed.empty:
        return ContextCheckResult(check, "not_assessed", None, 0, 0, details, f"{check} not assessed: no peer has comparable evidence in a common period.")
    supporting = int((assessed["status"] == "supported").sum())
    similarity = float(assessed["symmetric_similarity"].median())
    status = "supported" if similarity >= minimum_match_fraction else "not_supported"
    return ContextCheckResult(
        check, status, similarity, len(assessed), supporting, details,
        f"Median symmetric {check.lower()} agreement across {len(assessed)} local peer(s) is {similarity:.3f}.",
    )


def compare_epoch_behaviour(
    target: StationContextEvidence,
    peers: Mapping[str, StationContextEvidence],
    minimum_overlap_years: float,
    peer_consensus_fraction: float,
) -> ContextCheckResult:
    """Compare the frozen dominant epoch diagnosis over sufficiently overlapping records."""
    check = "Epoch behaviour"
    if not target.availability.get("epoch_behaviour", False):
        return ContextCheckResult(check, "not_assessed", None, 0, 0, pd.DataFrame(), "Epoch behaviour not assessed: target evidence is unavailable.")
    rows = []
    minimum_days = minimum_overlap_years * 365.25
    for peer_id, peer in peers.items():
        start, end = _overlap(target, peer)
        if start is None or not peer.availability.get("epoch_behaviour", False):
            rows.append({"peer_station_id": peer_id, "status": "not_assessed", "reason": "No common period or peer epoch evidence."})
            continue
        overlap_days = float((end - start) / pd.Timedelta(days=1) + 1)
        if overlap_days < minimum_days:
            rows.append({"peer_station_id": peer_id, "status": "not_assessed", "overlap_years": overlap_days / 365.25, "reason": "Common period is too short."})
            continue
        supports = peer.epoch_behaviour == target.epoch_behaviour
        rows.append({
            "peer_station_id": peer_id,
            "status": "supported" if supports else "not_supported",
            "common_start": start,
            "common_end": end,
            "overlap_years": overlap_days / 365.25,
            "minimum_overlap_years": minimum_overlap_years,
            "target_behaviour": target.epoch_behaviour,
            "peer_behaviour": peer.epoch_behaviour,
            "similarity": 1.0 if supports else 0.0,
            "reason": "",
        })
    details = pd.DataFrame(rows)
    assessed = details.loc[details.get("status", pd.Series(dtype=str)).isin(["supported", "not_supported"])]
    if assessed.empty:
        return ContextCheckResult(check, "not_assessed", None, 0, 0, details, "Epoch behaviour not assessed: no peer has sufficient overlapping evidence.")
    supporting = int((assessed.status == "supported").sum())
    similarity = float(assessed["similarity"].median())
    status = "supported" if similarity >= peer_consensus_fraction else "not_supported"
    return ContextCheckResult(
        check, status, similarity, len(assessed), supporting, details,
        f"Median pairwise epoch-behaviour agreement across {len(assessed)} local peer(s) is {similarity:.3f}.",
    )


def compare_local_context(
    target: StationContextEvidence,
    peers: Mapping[str, StationContextEvidence],
    config: Mapping[str, float],
) -> LocalContextResult:
    """Run the local checks while keeping each check independently inspectable."""
    consensus = float(config.get("peer_consensus_fraction", 0.50))
    peaks = _date_check(
        "High-flow peak timing", target, peers, "peak_dates",
        float(config.get("peak_tolerance_days", 5)),
        float(config.get(
            "minimum_peak_timing_similarity",
            config.get("minimum_peak_match_fraction", 0.50),
        )),
    )
    shifts = _date_check(
        "Step-shift timing", target, peers, "step_shift_dates",
        float(config.get("step_shift_tolerance_days", 50)),
        float(config.get(
            "minimum_step_shift_timing_similarity",
            config.get("minimum_step_shift_match_fraction", 0.50),
        )),
    )
    epochs = compare_epoch_behaviour(
        target, peers, float(config.get("minimum_epoch_overlap_years", 5.0)), consensus
    )
    return LocalContextResult(target.station_id, peaks, shifts, epochs)


__all__ = ["ContextCheckResult", "LocalContextResult", "compare_local_context"]
