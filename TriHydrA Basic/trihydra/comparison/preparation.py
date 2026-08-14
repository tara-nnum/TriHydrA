"""Raw-preserving preparation for arbitrary same-station comparisons.

Two comparison modes are deliberately distinct:

``paired``
    Two sources describing the same dates (OBS/model, model/model, OBS/OBS).
    Daily metrics may use only dates where both values are present.

``historical_profile``
    A selected period is compared with a historical baseline from the same
    station. Dates do not correspond one-to-one, so only distributions,
    signatures and event properties may be compared.

Neither mode imputes, clips, aggregates, or silently converts values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from trihydra.io.availability import pair_availability, series_availability
from trihydra.io.models import SourceProvenance

ComparisonMode = Literal["paired", "historical_profile"]


@dataclass(frozen=True)
class ComparisonSeries:
    """One named input and the information needed to interpret it."""

    name: str
    values: pd.Series
    station_id: str
    unit: str
    role: str = "unknown"
    source_name: str | None = None
    provenance: SourceProvenance | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedComparison:
    """Validated comparison inputs with explicit coverage and semantics."""

    mode: ComparisonMode
    reference: ComparisonSeries
    candidate: ComparisonSeries
    reference_selected: pd.Series
    candidate_selected: pd.Series
    pairwise_values: pd.DataFrame | None
    coverage: dict[str, Any]
    comparison_rules: tuple[str, ...]


def _normalise_unit(unit: str) -> str:
    token = str(unit).strip().casefold().replace(" ", "")
    aliases = {
        "mm/day": "mm/day",
        "mmd-1": "mm/day",
        "mm/day-1": "mm/day",
        "mmperday": "mm/day",
        "m³/s": "m3/s",
        "m3/s": "m3/s",
        "m^3/s": "m3/s",
        "cms": "m3/s",
    }
    return aliases.get(token, token)


def _validate_descriptor(item: ComparisonSeries) -> None:
    if not item.name.strip():
        raise ValueError("Comparison series name cannot be empty.")
    if not item.station_id.strip():
        raise ValueError(f"{item.name}: station_id cannot be empty.")
    if not isinstance(item.values, pd.Series):
        raise TypeError(f"{item.name}: values must be a pandas Series.")
    if not isinstance(item.values.index, pd.DatetimeIndex):
        raise TypeError(f"{item.name}: values must use a DatetimeIndex.")
    if item.values.index.has_duplicates:
        raise ValueError(f"{item.name}: duplicate timestamps are not comparable.")
    if not item.values.index.is_monotonic_increasing:
        raise ValueError(f"{item.name}: timestamps must be sorted.")


def _validate_pair(reference: ComparisonSeries, candidate: ComparisonSeries) -> None:
    _validate_descriptor(reference)
    _validate_descriptor(candidate)
    if reference.station_id != candidate.station_id:
        raise ValueError(
            "This comparison engine requires the same station ID; found "
            f"{reference.station_id!r} and {candidate.station_id!r}."
        )
    reference_unit = _normalise_unit(reference.unit)
    candidate_unit = _normalise_unit(candidate.unit)
    if reference_unit != candidate_unit:
        raise ValueError(
            "Comparison units differ and no silent conversion is allowed: "
            f"{reference.unit!r} versus {candidate.unit!r}."
        )


def _period_bounds(
    period: tuple[str | pd.Timestamp | None, str | pd.Timestamp | None] | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if period is None:
        return None, None
    if len(period) != 2:
        raise ValueError("period must contain exactly (start, end).")
    start = None if period[0] is None else pd.Timestamp(period[0])
    end = None if period[1] is None else pd.Timestamp(period[1])
    if start is not None and end is not None and start > end:
        raise ValueError("period start must not be after period end.")
    return start, end


def _select_period(
    series: pd.Series,
    period: tuple[str | pd.Timestamp | None, str | pd.Timestamp | None] | None,
) -> pd.Series:
    start, end = _period_bounds(period)
    selected = series
    if start is not None:
        selected = selected.loc[selected.index >= start]
    if end is not None:
        selected = selected.loc[selected.index <= end]
    return selected.copy(deep=True)


def prepare_paired_comparison(
    reference: ComparisonSeries,
    candidate: ComparisonSeries,
    *,
    period: tuple[str | pd.Timestamp | None, str | pd.Timestamp | None] | None = None,
) -> PreparedComparison:
    """Prepare contemporaneous same-station inputs on pairwise-valid dates."""
    _validate_pair(reference, candidate)
    reference_selected = _select_period(reference.values, period)
    candidate_selected = _select_period(candidate.values, period)
    full_coverage = pair_availability(
        reference_selected, candidate_selected, "reference", "candidate"
    )
    reference_first = reference_selected.first_valid_index()
    reference_last = reference_selected.last_valid_index()
    candidate_first = candidate_selected.first_valid_index()
    candidate_last = candidate_selected.last_valid_index()
    if any(value is None for value in (
        reference_first, reference_last, candidate_first, candidate_last
    )):
        raise ValueError("Both comparison inputs require at least one valid value.")
    common_start = max(reference_first, candidate_first)
    common_end = min(reference_last, candidate_last)
    if common_start > common_end:
        raise ValueError("The selected inputs have no common valid-record window.")
    reference_selected = reference_selected.loc[common_start:common_end].copy(deep=True)
    candidate_selected = candidate_selected.loc[common_start:common_end].copy(deep=True)
    frame = pd.concat(
        [
            pd.to_numeric(reference_selected, errors="coerce").rename("reference"),
            pd.to_numeric(candidate_selected, errors="coerce").rename("candidate"),
        ],
        axis=1,
        join="outer",
    )
    pairwise = frame.dropna(how="any")
    if pairwise.empty:
        raise ValueError("The selected inputs have no pairwise-valid dates.")
    coverage = pair_availability(
        reference_selected, candidate_selected, "reference", "candidate"
    )
    coverage.update(
        {
            "mode": "paired",
            "common_window_start": common_start,
            "common_window_end": common_end,
            "selected_period_start": period[0] if period else None,
            "selected_period_end": period[1] if period else None,
            "pairwise_values_only_for_daily_metrics": True,
            **{f"full_{key}": value for key, value in full_coverage.items()},
        }
    )
    return PreparedComparison(
        mode="paired",
        reference=reference,
        candidate=candidate,
        reference_selected=reference_selected,
        candidate_selected=candidate_selected,
        pairwise_values=pairwise,
        coverage=coverage,
        comparison_rules=(
            "Both series retain their own values and gaps inside one common window.",
            "Only optional pointwise metrics use pairwise-valid dates.",
            "Values outside the common valid-record window are excluded from comparison.",
            "No values are imputed, clipped, aggregated or unit-converted.",
        ),
    )


def prepare_historical_comparison(
    source: ComparisonSeries,
    selected_period: tuple[str | pd.Timestamp, str | pd.Timestamp],
    *,
    historical_period: tuple[
        str | pd.Timestamp | None, str | pd.Timestamp | None
    ] | None = None,
    exclude_selected_period: bool = True,
    selected_name: str = "selected_period",
    historical_name: str = "historical_baseline",
) -> PreparedComparison:
    """Prepare one selected period against an unpaired historical baseline."""
    _validate_descriptor(source)
    selected = _select_period(source.values, selected_period)
    historical = _select_period(source.values, historical_period)
    start, end = _period_bounds(selected_period)
    if start is None or end is None:
        raise ValueError("selected_period requires both a start and an end.")
    if exclude_selected_period:
        historical = historical.loc[
            (historical.index < start) | (historical.index > end)
        ].copy(deep=True)
    if not selected.notna().any():
        raise ValueError("The selected period contains no valid values.")
    if not historical.notna().any():
        raise ValueError("The historical baseline contains no valid values.")

    historical_descriptor = ComparisonSeries(
        name=historical_name,
        values=historical,
        station_id=source.station_id,
        unit=source.unit,
        role="historical_baseline",
        source_name=source.source_name,
        provenance=source.provenance,
        metadata=dict(source.metadata),
    )
    selected_descriptor = ComparisonSeries(
        name=selected_name,
        values=selected,
        station_id=source.station_id,
        unit=source.unit,
        role="selected_period",
        source_name=source.source_name,
        provenance=source.provenance,
        metadata=dict(source.metadata),
    )
    coverage = {
        **series_availability(historical, "reference"),
        **series_availability(selected, "candidate"),
        "mode": "historical_profile",
        "selected_period_start": start,
        "selected_period_end": end,
        "historical_period_start": historical_period[0] if historical_period else None,
        "historical_period_end": historical_period[1] if historical_period else None,
        "selected_period_excluded_from_baseline": exclude_selected_period,
        "pairwise_valid_count": None,
        "values_imputed": False,
    }
    return PreparedComparison(
        mode="historical_profile",
        reference=historical_descriptor,
        candidate=selected_descriptor,
        reference_selected=historical,
        candidate_selected=selected,
        pairwise_values=None,
        coverage=coverage,
        comparison_rules=(
            "Historical and selected dates are not paired.",
            "Compare distributions, signatures and event properties only.",
            "The selected period is excluded from the baseline when requested.",
            "No values are imputed, clipped, aggregated or unit-converted.",
        ),
    )


__all__ = [
    "ComparisonMode",
    "ComparisonSeries",
    "PreparedComparison",
    "prepare_historical_comparison",
    "prepare_paired_comparison",
]
