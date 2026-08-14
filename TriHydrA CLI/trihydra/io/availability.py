"""Coverage summaries that never alter or impute source series."""

from __future__ import annotations

import pandas as pd


def series_availability(series: pd.Series, prefix: str) -> dict:
    """Describe one complete source series, including its missing tail."""
    valid = pd.to_numeric(series, errors="coerce").notna()
    first = series.index[valid][0] if valid.any() else pd.NaT
    last = series.index[valid][-1] if valid.any() else pd.NaT
    return {
        f"{prefix}_calendar_start": series.index.min() if len(series) else pd.NaT,
        f"{prefix}_calendar_end": series.index.max() if len(series) else pd.NaT,
        f"{prefix}_calendar_count": int(len(series)),
        f"{prefix}_valid_count": int(valid.sum()),
        f"{prefix}_missing_count": int((~valid).sum()),
        f"{prefix}_first_valid": first,
        f"{prefix}_last_valid": last,
    }


def pair_availability(
    reference: pd.Series,
    candidate: pd.Series,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> dict:
    """Describe calendars and pairwise-valid overlap without filling either input."""
    frame = pd.concat(
        [reference.rename(reference_name), candidate.rename(candidate_name)],
        axis=1,
        join="outer",
    )
    pairwise = frame[reference_name].notna() & frame[candidate_name].notna()
    common_calendar = reference.index.intersection(candidate.index)
    candidate_only_valid = frame[candidate_name].notna() & frame[reference_name].isna()
    reference_only_valid = frame[reference_name].notna() & frame[candidate_name].isna()
    reference_valid = frame[reference_name].notna()
    candidate_valid = frame[candidate_name].notna()
    reference_first = frame.index[reference_valid][0] if reference_valid.any() else pd.NaT
    reference_last = frame.index[reference_valid][-1] if reference_valid.any() else pd.NaT
    candidate_first = frame.index[candidate_valid][0] if candidate_valid.any() else pd.NaT
    candidate_last = frame.index[candidate_valid][-1] if candidate_valid.any() else pd.NaT

    def _position_counts(mask: pd.Series, first, last, prefix: str) -> dict:
        if pd.isna(first) or pd.isna(last):
            return {
                f"{prefix}_before_other_valid_record": 0,
                f"{prefix}_within_other_valid_record": int(mask.sum()),
                f"{prefix}_after_other_valid_record": 0,
            }
        before = mask & (frame.index < first)
        after = mask & (frame.index > last)
        within = mask & (frame.index >= first) & (frame.index <= last)
        return {
            f"{prefix}_before_other_valid_record": int(before.sum()),
            f"{prefix}_within_other_valid_record": int(within.sum()),
            f"{prefix}_after_other_valid_record": int(after.sum()),
        }
    result = {
        **series_availability(reference, reference_name),
        **series_availability(candidate, candidate_name),
        "common_calendar_start": common_calendar.min() if len(common_calendar) else pd.NaT,
        "common_calendar_end": common_calendar.max() if len(common_calendar) else pd.NaT,
        "common_calendar_count": int(len(common_calendar)),
        "pairwise_valid_count": int(pairwise.sum()),
        "pairwise_valid_start": frame.index[pairwise][0] if pairwise.any() else pd.NaT,
        "pairwise_valid_end": frame.index[pairwise][-1] if pairwise.any() else pd.NaT,
        f"{candidate_name}_valid_without_{reference_name}": int(candidate_only_valid.sum()),
        f"{reference_name}_valid_without_{candidate_name}": int(reference_only_valid.sum()),
        **_position_counts(
            candidate_only_valid,
            reference_first,
            reference_last,
            f"{candidate_name}_valid_without_{reference_name}",
        ),
        **_position_counts(
            reference_only_valid,
            candidate_first,
            candidate_last,
            f"{reference_name}_valid_without_{candidate_name}",
        ),
        "values_imputed": False,
    }
    return result


__all__ = ["pair_availability", "series_availability"]
