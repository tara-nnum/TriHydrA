"""Adapt frozen Layer 1 and Layer 2 results into Layer 3 evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from trihydra.composite import score_layer1


ANNUAL_COLUMNS = [
    "year",
    "flashiness_index",
    "baseflow_index",
    "seasonality_index",
]


@dataclass(frozen=True)
class StationContextEvidence:
    """The existing station evidence needed by Layer 3 and nothing more."""

    station_id: str
    series_type: str
    record_start: pd.Timestamp | None
    record_end: pd.Timestamp | None
    peak_dates: pd.DatetimeIndex
    step_shift_dates: pd.DatetimeIndex
    epoch_behaviour: str | None
    epoch_overview: pd.DataFrame
    zero_flow_ratio: float | None
    annual_signatures: pd.DataFrame
    seasonality_profile: pd.Series
    representative_event_metrics: dict[str, Any]
    representative_event_curve: pd.Series
    layer1_review_class: str | None = None
    availability: dict[str, bool] = field(default_factory=dict)
    unavailable_reasons: dict[str, str] = field(default_factory=dict)


def _empty_datetime_index() -> pd.DatetimeIndex:
    return pd.DatetimeIndex([], dtype="datetime64[ns]")


def _check_result(layer1_result: dict | None, check: str, series_key: str) -> dict | None:
    if not layer1_result:
        return None
    raw = layer1_result.get("raw_results", {}).get(series_key, [])
    return next((item for item in raw if item.get("check") == check), None)


def _dates(values: list[Any]) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(pd.Series(values, dtype="object"), errors="coerce")
    return pd.DatetimeIndex(parsed.dropna().sort_values().unique())


def _representative_event(
    raw_series: pd.Series,
    layer2_result: dict | None,
) -> tuple[dict[str, Any], pd.Series]:
    """Extract the already-selected real event in its original discharge units."""
    if not layer2_result:
        return {}, pd.Series(dtype=float, name="discharge")
    table = layer2_result.get("representative_event")
    if not isinstance(table, pd.DataFrame) or table.empty:
        return {}, pd.Series(dtype=float, name="discharge")
    row = table.iloc[0]
    start = pd.to_datetime(row.get("event_start"), errors="coerce")
    end = pd.to_datetime(row.get("event_end"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return row.to_dict(), pd.Series(dtype=float, name="discharge")

    observed = pd.to_numeric(raw_series.loc[start:end], errors="coerce")
    if observed.empty or not observed.notna().any():
        curve = pd.Series(dtype=float, name="discharge")
    else:
        curve = observed.copy()
        curve.index = (curve.index - start) / pd.Timedelta(days=1)
        curve.index.name = "relative_event_day"
        curve.name = "discharge"
    return row.to_dict(), curve


def build_station_context_evidence(
    station_id: str,
    raw_series: pd.Series,
    layer1_result: dict | None,
    layer2_result: dict | None,
    *,
    layer1_series_key: str = "obs",
    series_type: str = "observation",
    layer1_config: Mapping[str, Any] | None = None,
) -> StationContextEvidence:
    """Build Layer 3 input solely from existing diagnostic outputs."""
    valid = pd.to_numeric(raw_series, errors="coerce").dropna()
    record_start = pd.Timestamp(valid.index.min()) if not valid.empty else None
    record_end = pd.Timestamp(valid.index.max()) if not valid.empty else None

    zero_result = _check_result(layer1_result, "zero_flow_regime", layer1_series_key)
    step_result = _check_result(layer1_result, "step_shift", layer1_series_key)
    epoch_result = _check_result(layer1_result, "epoch_drift", layer1_series_key)

    boundaries = [] if step_result is None else step_result.get("regime_boundaries", []) or []
    # Tier 3 boundaries were classified as negligible by Layer 1 and are not
    # presented as meaningful contextual change dates.
    meaningful_boundaries = [
        item.get("boundary_timestamp")
        for item in boundaries
        if item.get("diagnosis") in {"Tier 1", "Tier 2"}
    ]

    events = None if not layer2_result else layer2_result.get("hydrograph_events")
    peak_dates = (
        _empty_datetime_index()
        if not isinstance(events, pd.DataFrame) or "peak_date" not in events
        else _dates(events["peak_date"].tolist())
    )

    annual = None if not layer2_result else layer2_result.get("annual_signatures")
    if isinstance(annual, pd.DataFrame):
        annual_signatures = annual.reindex(columns=ANNUAL_COLUMNS).copy()
    else:
        annual_signatures = pd.DataFrame(columns=ANNUAL_COLUMNS)

    profile = None if not layer2_result else layer2_result.get("seasonality_profile")
    if isinstance(profile, pd.DataFrame) and {"month", "median"}.issubset(profile.columns):
        seasonality = profile.set_index("month")["median"].reindex(range(1, 13)).astype(float)
        seasonality.name = "monthly_median_discharge"
    else:
        seasonality = pd.Series(index=range(1, 13), dtype=float, name="monthly_median_discharge")

    event_metrics, event_curve = _representative_event(raw_series, layer2_result)
    review_class = None
    if layer1_result:
        assessment = score_layer1(
            raw_series, layer1_result, layer2_result, config=layer1_config
        )
        review_class = str(assessment["summary"].iloc[0]["layer1_class"])
    overview = pd.DataFrame([] if epoch_result is None else epoch_result.get("overview_diagnosis", []) or [])
    zero_ratio = None if zero_result is None else zero_result.get("zero_ratio")
    zero_ratio = float(zero_ratio) if zero_ratio is not None and np.isfinite(zero_ratio) else None

    availability = {
        # An executed check with no detected events/shifts is valid evidence,
        # not missing evidence.
        "peak_dates": isinstance(events, pd.DataFrame),
        "step_shift_dates": bool(
            step_result and step_result.get("execution_status") != "skipped"
        ),
        "epoch_behaviour": bool(epoch_result and epoch_result.get("dominant_behaviour")),
        "zero_flow_ratio": zero_ratio is not None,
        "annual_signatures": not annual_signatures.empty,
        "seasonality_profile": bool(seasonality.notna().any()),
        "representative_event": bool(event_metrics) and not event_curve.empty,
    }
    reasons = {
        key: "Required Layer 1/2 evidence was unavailable or not calculable."
        for key, available in availability.items()
        if not available
    }

    return StationContextEvidence(
        station_id=str(station_id),
        series_type=str(series_type),
        record_start=record_start,
        record_end=record_end,
        peak_dates=peak_dates,
        step_shift_dates=_dates(meaningful_boundaries),
        epoch_behaviour=None if epoch_result is None else epoch_result.get("dominant_behaviour"),
        epoch_overview=overview,
        zero_flow_ratio=zero_ratio,
        annual_signatures=annual_signatures,
        seasonality_profile=seasonality,
        representative_event_metrics=event_metrics,
        representative_event_curve=event_curve,
        layer1_review_class=review_class,
        availability=availability,
        unavailable_reasons=reasons,
    )


__all__ = ["StationContextEvidence", "build_station_context_evidence"]
