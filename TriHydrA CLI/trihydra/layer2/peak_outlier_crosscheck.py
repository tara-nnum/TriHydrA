"""Cross-check Layer 1 spike/dip candidates with Layer 2 high-flow events."""

from __future__ import annotations

import pandas as pd


COLUMNS = [
    "timestamp", "type", "candidate_value", "raw_score", "matched_event_id",
    "matched_event_duration_days", "coincides_with_event_peak",
    "crosscheck_status", "assessment",
]


def crosscheck_peak_outliers(
    candidate_details: list[dict],
    events: pd.DataFrame,
    *,
    minimum_event_duration_days: float = 3.0,
) -> pd.DataFrame:
    """Classify Layer 1 candidates using detected high-flow event context."""
    if minimum_event_duration_days < 1:
        raise ValueError("minimum_event_duration_days must be at least 1")
    rows = []
    for candidate in candidate_details or []:
        timestamp = pd.Timestamp(candidate["timestamp"])
        matching = events[
            (events["event_start"] <= timestamp) & (events["event_end"] >= timestamp)
        ] if not events.empty else pd.DataFrame()
        kind = candidate.get("type", "candidate")
        event = None if matching.empty else matching.iloc[0]
        event_id = None if event is None else int(event["event_id"])
        event_duration = (
            None if event is None else float(event["event_duration_days"])
        )
        is_event_peak = bool(
            kind == "spike" and event is not None
            and timestamp == pd.Timestamp(event["peak_date"])
        )
        if kind == "dip":
            status = "retained_for_review"
            assessment = "Dip candidate; a high-flow peak cannot directly explain this observation."
        elif matching.empty:
            status = "retained_for_review"
            assessment = "Spike candidate does not coincide with a detected coherent high-flow event."
        else:
            if is_event_peak:
                status = "spike_peak_overlap_review"
                assessment = (
                    "Layer 1 spike candidate and Layer 2 high-flow peak overlap. "
                    "The event remains in the raw catalogue but cannot be selected "
                    "as the representative event; check the observation."
                )
            elif event_duration < minimum_event_duration_days:
                status = "retained_for_review"
                assessment = (
                    f"Matched high-flow event lasts only {event_duration:g} day(s); "
                    "the candidate may have created the event and remains unresolved."
                )
            else:
                status = "plausible_event_context"
                assessment = (
                    "Spike candidate lies inside a coherent high-flow event but is "
                    "not its peak; event context reduces concern without proving validity."
                )
        rows.append({
            "timestamp": timestamp, "type": kind,
            "candidate_value": candidate.get("candidate_value"),
            "raw_score": candidate.get("raw_score"),
            "matched_event_id": event_id,
            "matched_event_duration_days": event_duration,
            "coincides_with_event_peak": is_event_peak,
            "crosscheck_status": status,
            "assessment": assessment,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def mark_representative_eligibility(
    events: pd.DataFrame,
    crosscheck: pd.DataFrame,
) -> pd.DataFrame:
    """Exclude event peaks that overlap flagged Layer 1 spike candidates."""
    marked = events.copy()
    marked["layer1_spike_peak_overlap"] = False
    marked["representative_eligible"] = True
    marked["representative_exclusion_reason"] = None
    if marked.empty or crosscheck.empty:
        return marked

    overlap_ids = set(
        pd.to_numeric(
            crosscheck.loc[
                crosscheck["coincides_with_event_peak"].fillna(False),
                "matched_event_id",
            ],
            errors="coerce",
        ).dropna().astype(int)
    )
    overlap = marked["event_id"].isin(overlap_ids)
    marked.loc[overlap, "layer1_spike_peak_overlap"] = True
    marked.loc[overlap, "representative_eligible"] = False
    marked.loc[overlap, "representative_exclusion_reason"] = (
        "Layer 1 spike candidate overlaps the event peak; manual check required."
    )
    return marked
