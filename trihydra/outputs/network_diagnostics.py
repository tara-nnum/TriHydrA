"""Network-level counts derived from completed Layer 1 assessments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from trihydra.result import TriHydrAResult


def _classification(value: object) -> str:
    """Return one stable machine-readable Layer 1 classification."""
    label = str(value).strip().casefold()
    if label in {"needs review", "review"}:
        return "needs_review"
    if label == "minor concerns":
        return "minor_concerns"
    if label == "no review needed":
        return "no_review"
    return "not_assessed"


def network_assessment_counts(summary: pd.DataFrame) -> dict[str, int]:
    """Count physical stations and independently assessed station-series."""
    if summary is None or summary.empty:
        return {
            name: 0 for name in (
                "unique_station_count", "station_series_assessment_count",
                "needs_review_station_count", "minor_concerns_station_count",
                "no_concerns_station_count", "not_assessed_station_count",
                "needs_review_series_count", "minor_concerns_series_count",
                "no_review_series_count", "not_assessed_series_count",
            )
        }

    frame = summary.copy()
    frame["_classification"] = frame.get(
        "layer1_class", pd.Series(index=frame.index, dtype=object)
    ).map(_classification)
    series_counts = frame["_classification"].value_counts()

    station_classes: list[str] = []
    for _, rows in frame.groupby("station_id", dropna=False, sort=False):
        labels = set(rows["_classification"])
        if "needs_review" in labels:
            station_classes.append("needs_review")
        elif "minor_concerns" in labels:
            station_classes.append("minor_concerns")
        elif "no_review" in labels:
            station_classes.append("no_review")
        else:
            station_classes.append("not_assessed")
    station_counts = pd.Series(station_classes, dtype=str).value_counts()

    return {
        "unique_station_count": int(frame["station_id"].nunique(dropna=True)),
        "station_series_assessment_count": int(len(frame)),
        "needs_review_station_count": int(station_counts.get("needs_review", 0)),
        "minor_concerns_station_count": int(station_counts.get("minor_concerns", 0)),
        "no_concerns_station_count": int(station_counts.get("no_review", 0)),
        "not_assessed_station_count": int(station_counts.get("not_assessed", 0)),
        "needs_review_series_count": int(series_counts.get("needs_review", 0)),
        "minor_concerns_series_count": int(series_counts.get("minor_concerns", 0)),
        "no_review_series_count": int(series_counts.get("no_review", 0)),
        "not_assessed_series_count": int(series_counts.get("not_assessed", 0)),
    }


def _component_tables(result: TriHydrAResult) -> list[tuple[str, pd.DataFrame]]:
    """Return enabled Layer 1 component tables for each assessed series."""
    tables: list[tuple[str, pd.DataFrame]] = []
    if result.layer1_composite:
        table = result.layer1_composite.get("components")
        if isinstance(table, pd.DataFrame) and not table.empty:
            tables.append((result.station.series1_name, table))
    if result.comparison:
        composite = result.comparison.get("candidate_layer1_composite")
        table = None if not composite else composite.get("components")
        if isinstance(table, pd.DataFrame) and not table.empty:
            tables.append((str(result.comparison.get("candidate_name")), table))
    return tables


def diagnostic_trigger_summary(
    results: Iterable[TriHydrAResult],
) -> pd.DataFrame:
    """Aggregate Layer 1 concern frequency without recalculating diagnostics."""
    rows: list[dict[str, object]] = []
    for result in results:
        for series_name, components in _component_tables(result):
            for component in components.to_dict("records"):
                rows.append({
                    "station_id": result.station_id,
                    "series_name": series_name,
                    "diagnostic": str(component.get("check", "")),
                    "assessable": bool(component.get("assessable", False)),
                    "tier": str(component.get("tier", "")),
                    "contribution": pd.to_numeric(
                        component.get("contribution"), errors="coerce"
                    ),
                })
    columns = [
        "diagnostic", "enabled_series_count", "assessable_series_count",
        "unassessable_series_count", "tier1_count", "tier2_count",
        "concern_series_count", "trigger_rate_percent", "total_contribution",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    evidence = pd.DataFrame(rows)
    # One diagnostic contributes at most once per independently assessed series.
    evidence = evidence.drop_duplicates(
        ["station_id", "series_name", "diagnostic"], keep="last"
    )
    evidence["contribution"] = evidence["contribution"].fillna(0.0)
    summaries: list[dict[str, object]] = []
    for diagnostic, group in evidence.groupby("diagnostic", sort=False):
        assessable = group["assessable"]
        concerns = assessable & group["contribution"].gt(0)
        denominator = int(assessable.sum())
        concern_count = int(concerns.sum())
        summaries.append({
            "diagnostic": diagnostic,
            "enabled_series_count": int(len(group)),
            "assessable_series_count": denominator,
            "unassessable_series_count": int((~assessable).sum()),
            "tier1_count": int((concerns & group["tier"].eq("Tier 1")).sum()),
            "tier2_count": int((concerns & group["tier"].eq("Tier 2")).sum()),
            "concern_series_count": concern_count,
            "trigger_rate_percent": (
                100.0 * concern_count / denominator if denominator else np.nan
            ),
            "total_contribution": float(group.loc[concerns, "contribution"].sum()),
        })
    return (
        pd.DataFrame(summaries, columns=columns)
        .sort_values(
            ["concern_series_count", "trigger_rate_percent", "diagnostic"],
            ascending=[False, False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )


__all__ = ["diagnostic_trigger_summary", "network_assessment_counts"]
