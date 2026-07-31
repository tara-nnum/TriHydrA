"""
gauge_network.py

Selects hydrologically relevant context candidates from already-loaded
station metadata. File loading belongs to ``trihydra.io.readers``.

Pure computation -- no discharge data or source file is touched here.
Candidate selection needs already-loaded network metadata only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalised_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold()


def find_context_candidates(
    meta: pd.DataFrame,
    target_id: str,
    maximum_candidates: int = 10,
) -> dict:
    """
    Find and rank context candidates for one target gauge.

    Candidate priority:
      1. Same named catchment AND same named river.
      2. Same named catchment but a different river/tributary.
      3. Symmetric catchment-area similarity ranks candidates within
         either group. Geographic distance is not used for eligibility
         or ranking.

    Context tiers (interpretation text lives in diagnostics.py, not
    here -- this function only returns the raw tier label):
      strong    -- >= 2 same-river candidates
      moderate  -- 1 same-river candidate plus >= 2 total
      weak      -- >= 2 same-catchment candidates, none same-river
      limited   -- exactly 1 candidate, no majority agreement possible
      unavailable -- no suitable candidate at all (a valid outcome,
                     not a data-quality problem)
    """
    target_rows = meta.loc[meta["gauge_id"].eq(target_id)]
    if target_rows.empty:
        raise KeyError(f"{target_id} was not found in the gauge network metadata.")

    target = target_rows.iloc[0]
    candidates = meta.loc[~meta["gauge_id"].eq(target_id)].copy()

    target_catchment = str(target.get("Catchment", "")).strip().casefold()
    target_river = str(target.get("River", "")).strip().casefold()

    candidates["same_catchment"] = (
        _normalised_text(candidates["Catchment"]).eq(target_catchment)
        & bool(target_catchment)
    )
    candidates["same_river"] = (
        candidates["same_catchment"]
        & _normalised_text(candidates["River"]).eq(target_river)
        & bool(target_river)
    )

    target_area = target["area_km2"]
    candidates["area_ratio_to_target"] = candidates["area_km2"] / target_area
    candidates["area_similarity"] = np.minimum(
        candidates["area_ratio_to_target"],
        1.0 / candidates["area_ratio_to_target"],
    )

    eligible = candidates[
        candidates["same_catchment"]
        & candidates["area_similarity"].notna()
    ].copy()

    eligible["priority"] = np.where(eligible["same_river"], 1, 2)
    eligible = eligible.sort_values(
        ["priority", "area_similarity"],
        ascending=[True, False],
    )

    same_river_count = int(eligible["same_river"].sum())
    total_count = len(eligible)

    if same_river_count >= 2:
        status = "strong"
    elif same_river_count == 1 and total_count >= 2:
        status = "moderate"
    elif total_count >= 2:
        status = "weak"
    elif total_count == 1:
        status = "limited"
    else:
        status = "unavailable"

    result_columns = [
        "gauge_id", "StationName", "Catchment", "River",
        "StationLat", "StationLon", "area_km2",
        "area_ratio_to_target", "area_similarity", "same_river",
    ]

    return {
        "target_id": target_id,
        "target": target,
        "selection_method": (
            "same catchment; same river first; then descending symmetric "
            "area similarity = min(candidate area, target area) / max(...)"
        ),
        "status": status,
        "candidates": eligible[result_columns].head(maximum_candidates),
    }
