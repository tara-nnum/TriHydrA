"""Adaptive block screening for persistent level and regime shifts.

The check summarises the supplied series in adequately observed multi-year
blocks, learns each station's ordinary adjacent-block variation, and merges
neighbouring blocks until only unusually large structural differences remain.

Each retained boundary is tiered by its absolute median change against
station-specific low-flow references. Boundary tiers are then averaged into
one series-level step-shift diagnosis.

The detector uses no imputation, station metadata, station IDs, or known dates.
It accepts observations and model series through the same public API.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trihydra.layer1.check_result import make_result
from trihydra.layer1.timeseries_validity import get_valid_record


FEATURES = (
    "level_log",
    "high_log",
    "seasonal_amplitude_log",
    "variability_log",
)

# Conservative numerical floors prevent nearly constant records from turning
# negligible differences into enormous station-standardised scores.
FEATURE_SCALE_FLOORS = {
    "level_log": 0.10,
    "high_log": 0.15,
    "seasonal_amplitude_log": 0.15,
    "variability_log": 0.12,
}


def _monthly_observed_statistics(
    series: pd.Series,
    minimum_valid_days_per_month: int,
) -> pd.DataFrame:
    """Return observed monthly quantiles without filling missing values."""
    daily = series.astype(float).groupby(series.index).median().sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar).clip(lower=0)
    grouped = daily.resample("MS")
    monthly = pd.DataFrame({
        "valid_days": grouped.count(),
        "median": grouped.median(),
        "q25": grouped.quantile(0.25),
        "q75": grouped.quantile(0.75),
        "q90": grouped.quantile(0.90),
    })
    insufficient = monthly["valid_days"] < minimum_valid_days_per_month
    monthly.loc[insufficient, ["median", "q25", "q75", "q90"]] = np.nan
    return monthly


def _station_log_offset(series: pd.Series) -> float:
    """Return a small station-relative offset for logarithmic features."""
    positive = series[(series > 0) & series.notna()]
    if positive.empty:
        return 1e-6
    return max(float(positive.quantile(0.10)) * 0.10, 1e-6)


def _adaptive_block_rule(
    series: pd.Series,
    long_record_min_years: float,
    long_record_block_years: int,
    short_record_divisor: float,
    minimum_block_years: int,
) -> dict[str, float | int]:
    """Choose a block length that leaves short records roughly three blocks."""
    first = series.first_valid_index()
    last = series.last_valid_index()
    if first is None or last is None:
        raise ValueError("Series contains no valid values.")
    record_years = ((last - first).days + 1) / 365.2425
    if record_years >= long_record_min_years:
        block_years = int(long_record_block_years)
    else:
        block_years = max(
            int(minimum_block_years),
            math.floor(record_years / short_record_divisor),
        )
    return {
        "record_years": float(record_years),
        "block_years": int(block_years),
    }


def _block_edges(
    series: pd.Series,
    block_years: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Construct calendar blocks across the observed record."""
    first_start = pd.Timestamp(series.index.min().year, 1, 1)

    edges = []
    start = first_start
    while start <= series.index.max():
        end = start + pd.DateOffset(years=block_years) - pd.Timedelta(days=1)
        clipped_start = max(start, series.index.min())
        clipped_end = min(end, series.index.max())
        if clipped_start <= clipped_end:
            edges.append((clipped_start, clipped_end))
        start += pd.DateOffset(years=block_years)
    return edges


def _summarise_block(
    series: pd.Series,
    monthly: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    log_offset: float,
    minimum_block_coverage: float,
    minimum_calendar_months: int,
) -> dict | None:
    """Return a seasonally balanced block summary or reject poor coverage."""
    block_monthly = monthly.loc[start:end].dropna(subset=["median"])
    expected_months = len(pd.date_range(
        start.to_period("M").start_time,
        end.to_period("M").start_time,
        freq="MS",
    ))
    valid_months = len(block_monthly)
    coverage = valid_months / expected_months if expected_months else 0.0
    calendar_month_count = int(block_monthly.index.month.nunique())
    if (
        coverage < minimum_block_coverage
        or calendar_month_count < minimum_calendar_months
    ):
        return None

    log_median = np.log(block_monthly["median"] + log_offset)
    log_q25 = np.log(block_monthly["q25"] + log_offset)
    log_q75 = np.log(block_monthly["q75"] + log_offset)
    log_q90 = np.log(block_monthly["q90"] + log_offset)
    month_climatology = log_median.groupby(block_monthly.index.month).median()
    return {
        "start": start,
        "end": end,
        "coverage": float(coverage),
        "valid_months": int(valid_months),
        "level_log": float(log_median.median()),
        "high_log": float(log_q90.median()),
        "seasonal_amplitude_log": float(
            month_climatology.quantile(0.90)
            - month_climatology.quantile(0.10)
        ),
        "variability_log": float(np.nanmedian(log_q75 - log_q25)),
    }


def _build_blocks(
    series: pd.Series,
    monthly: pd.DataFrame,
    block_years: int,
    log_offset: float,
    minimum_block_coverage: float,
    minimum_calendar_months: int,
) -> pd.DataFrame:
    rows = []
    for start, end in _block_edges(series, block_years):
        summary = _summarise_block(
            series,
            monthly,
            start,
            end,
            log_offset,
            minimum_block_coverage,
            minimum_calendar_months,
        )
        if summary is not None:
            rows.append(summary)
    return pd.DataFrame(rows)


def _robust_feature_scales(blocks: pd.DataFrame) -> dict[str, float]:
    """Learn ordinary adjacent-block differences, with conservative floors."""
    scales = {}
    for feature in FEATURES:
        if feature not in blocks:
            scales[feature] = FEATURE_SCALE_FLOORS[feature]
            continue
        differences = blocks[feature].diff().abs().dropna().to_numpy()
        if len(differences):
            cutoff = np.quantile(differences, 0.50)
            ordinary = differences[differences <= cutoff]
            centre = float(np.median(ordinary)) if len(ordinary) else 0.0
            mad = (
                float(np.median(np.abs(ordinary - centre)))
                if len(ordinary)
                else 0.0
            )
            learned = centre + 1.4826 * mad
        else:
            learned = 0.0
        scales[feature] = max(learned, FEATURE_SCALE_FLOORS[feature])
    return scales


def _component_scores(
    left: dict | pd.Series,
    right: dict | pd.Series,
    scales: dict[str, float],
) -> dict[str, float]:
    return {
        feature: abs(float(right[feature]) - float(left[feature]))
        / scales[feature]
        for feature in FEATURES
    }


def _segment_from_block(row: pd.Series) -> dict:
    return {
        "start": pd.Timestamp(row["start"]),
        "end": pd.Timestamp(row["end"]),
        "weight": float(row["valid_months"]),
        **{feature: float(row[feature]) for feature in FEATURES},
    }


def _merge_segments(left: dict, right: dict) -> dict:
    total_weight = left["weight"] + right["weight"]
    merged = {
        "start": left["start"],
        "end": right["end"],
        "weight": total_weight,
    }
    for feature in FEATURES:
        merged[feature] = (
            left[feature] * left["weight"]
            + right[feature] * right["weight"]
        ) / total_weight
    return merged


def _bottom_up_merge(
    blocks: pd.DataFrame,
    scales: dict[str, float],
    structural_threshold: float,
) -> list[dict]:
    """Merge the most similar neighbours until all remaining jumps are large."""
    segments = [_segment_from_block(row) for _, row in blocks.iterrows()]
    while len(segments) > 1:
        distances = []
        for index in range(len(segments) - 1):
            components = _component_scores(
                segments[index], segments[index + 1], scales
            )
            distances.append((max(components.values()), index))
        smallest_distance, merge_index = min(distances)
        if smallest_distance >= structural_threshold:
            break
        segments[merge_index:merge_index + 2] = [
            _merge_segments(segments[merge_index], segments[merge_index + 1])
        ]
    return segments


def _refine_level_boundary(
    monthly: pd.DataFrame,
    approximate: pd.Timestamp,
    log_offset: float,
    search_months: int,
    window_months: int,
) -> pd.Timestamp:
    """Refine a level-related boundary using adjusted monthly medians."""
    values = np.log(monthly["median"] + log_offset)
    climatology = values.groupby(values.index.month).median()
    anomalies = values - values.index.month.map(climatology).to_numpy()
    candidates = pd.date_range(
        approximate - pd.DateOffset(months=search_months),
        approximate + pd.DateOffset(months=search_months),
        freq="MS",
    )
    best_date = approximate
    best_score = -np.inf
    for candidate in candidates:
        before = anomalies.loc[
            candidate - pd.DateOffset(months=window_months):
            candidate - pd.offsets.MonthBegin(1)
        ].dropna()
        after = anomalies.loc[
            candidate:candidate + pd.DateOffset(months=window_months - 1)
        ].dropna()
        minimum_evidence = max(3, math.ceil(window_months / 2))
        if len(before) < minimum_evidence or len(after) < minimum_evidence:
            continue
        score = abs(float(after.median()) - float(before.median()))
        if score > best_score:
            best_score = score
            best_date = candidate
    return pd.Timestamp(best_date)


def _consolidate_same_direction(
    candidates: list[dict],
    block_years: int,
    maximum_block_widths: float,
) -> list[dict]:
    """Collapse nearby same-direction edges while keeping reversals separate."""
    if not candidates:
        return []
    maximum_days = 366 * block_years * maximum_block_widths
    groups: list[list[dict]] = [[candidates[0]]]
    for candidate in candidates[1:]:
        previous = groups[-1][-1]
        separation = (
            candidate["approximate_block_boundary"]
            - previous["approximate_block_boundary"]
        ).days
        if (
            candidate["direction"] == previous["direction"]
            and separation <= maximum_days
        ):
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    consolidated = []
    for group in groups:
        combined = dict(group[0])
        combined["score"] = max(item["score"] for item in group)
        for feature in FEATURES:
            key = f"score_{feature}"
            combined[key] = max(item[key] for item in group)
        combined["dominant_feature"] = max(
            FEATURES,
            key=lambda feature: combined[f"score_{feature}"],
        )
        combined["collapsed_edge_count"] = len(group)
        consolidated.append(combined)
    return consolidated


def _summarise_regime(
    series: pd.Series,
    monthly: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    regime_number: int,
) -> dict:
    monthly_values = monthly.loc[start:end, "median"].dropna()
    observed = series.loc[start:end].dropna()
    if monthly_values.empty:
        median = math.nan
    else:
        median = float(monthly_values.median())
    return {
        "regime": regime_number,
        "start": str(start),
        "end": str(end),
        "valid_day_count": int(len(observed)),
        "median": median,
    }


def _detect(
    series: pd.Series,
    *,
    long_record_min_years: float,
    long_record_block_years: int,
    short_record_divisor: float,
    minimum_block_years: int,
    minimum_valid_days_per_month: int,
    minimum_block_coverage: float,
    minimum_calendar_months: int,
    structural_threshold: float,
    tier_3_maximum_quantile: float,
    tier_1_minimum_quantile: float,
    refinement_block_fraction: float,
    consolidation_max_block_widths: float,
) -> dict:
    """Detect and tier retained candidates in the valid record."""
    rule = _adaptive_block_rule(
        series,
        long_record_min_years,
        long_record_block_years,
        short_record_divisor,
        minimum_block_years,
    )
    block_years = int(rule["block_years"])
    monthly = _monthly_observed_statistics(
        series, minimum_valid_days_per_month
    )
    valid_values = series.dropna().clip(lower=0)
    fdc_q95_low_flow = float(valid_values.quantile(tier_3_maximum_quantile))
    fdc_q75_low_flow = float(valid_values.quantile(tier_1_minimum_quantile))
    log_offset = _station_log_offset(series)
    primary_blocks = _build_blocks(
        series,
        monthly,
        block_years,
        log_offset,
        minimum_block_coverage,
        minimum_calendar_months,
    )
    metadata = {
        "start": str(series.index.min()),
        "end": str(series.index.max()),
        "record_years": rule["record_years"],
        "block_years": block_years,
        "primary_block_count": int(len(primary_blocks)),
    }
    if len(primary_blocks) < 3:
        return {
            "boundaries": [],
            "regimes": [],
            "metadata": metadata,
            "feature_scales": {},
        }

    primary_scales = _robust_feature_scales(primary_blocks)
    segments = _bottom_up_merge(
        primary_blocks, primary_scales, structural_threshold
    )

    candidates = []
    for index in range(len(segments) - 1):
        left, right = segments[index], segments[index + 1]
        approximate = pd.Timestamp(right["start"])
        components = _component_scores(left, right, primary_scales)
        score = max(components.values())
        dominant = max(components, key=components.get)
        candidates.append({
            "approximate_block_boundary": approximate,
            "boundary_timestamp": approximate,
            "score": float(score),
            "dominant_feature": dominant,
            "direction": int(np.sign(
                float(right[dominant]) - float(left[dominant])
            )),
            **{
                f"score_{feature}": float(value)
                for feature, value in components.items()
            },
        })

    candidates = _consolidate_same_direction(
        candidates,
        block_years,
        consolidation_max_block_widths,
    )
    refinement_months = max(
        1, int(round(block_years * 12 * refinement_block_fraction))
    )
    for candidate in candidates:
        if candidate["score_level_log"] >= structural_threshold:
            candidate["boundary_timestamp"] = _refine_level_boundary(
                monthly,
                candidate["approximate_block_boundary"],
                log_offset,
                refinement_months,
                refinement_months,
            )

    candidates.sort(key=lambda item: item["boundary_timestamp"])
    unique_candidates = []
    for candidate in candidates:
        if (
            unique_candidates
            and candidate["boundary_timestamp"]
            <= unique_candidates[-1]["boundary_timestamp"]
        ):
            candidate["boundary_timestamp"] = candidate[
                "approximate_block_boundary"
            ]
        unique_candidates.append(candidate)
    candidates = unique_candidates

    first = series.first_valid_index()
    last = series.last_valid_index()
    all_dates = [item["boundary_timestamp"] for item in candidates]
    all_starts = [first, *all_dates]
    all_ends = [
        *(date - pd.Timedelta(days=1) for date in all_dates),
        last,
    ]
    all_regimes = [
        _summarise_regime(
            series, monthly, start, end, number
        )
        for number, (start, end) in enumerate(
            zip(all_starts, all_ends), start=1
        )
        if start <= end
    ]

    boundaries = []
    for index, candidate in enumerate(candidates):
        before = float(all_regimes[index]["median"])
        after = float(all_regimes[index + 1]["median"])
        absolute_median_change = abs(after - before)
        if absolute_median_change <= fdc_q95_low_flow:
            diagnosis = "Tier 3"
        elif absolute_median_change >= fdc_q75_low_flow:
            diagnosis = "Tier 1"
        else:
            diagnosis = "Tier 2"
        boundaries.append({
            "boundary_timestamp": str(candidate["boundary_timestamp"]),
            "before_median": before,
            "after_median": after,
            "absolute_median_change": float(absolute_median_change),
            "fdc_q95_percentile_q05_threshold": fdc_q95_low_flow,
            "fdc_q75_percentile_q25_threshold": fdc_q75_low_flow,
            "diagnosis": diagnosis,
            "structural_score": float(candidate["score"]),
            "approximate_block_boundary": str(
                candidate["approximate_block_boundary"]
            ),
            "collapsed_edge_count": int(
                candidate.get("collapsed_edge_count", 1)
            ),
            "score_level_log": float(candidate["score_level_log"]),
            "score_high_log": float(candidate["score_high_log"]),
            "score_seasonal_amplitude_log": float(
                candidate["score_seasonal_amplitude_log"]
            ),
            "score_variability_log": float(
                candidate["score_variability_log"]
            ),
        })

    public_dates = [
        pd.Timestamp(item["boundary_timestamp"])
        for item in boundaries
        if item["diagnosis"] in {"Tier 1", "Tier 2"}
    ]
    public_starts = [first, *public_dates]
    public_ends = [
        *(date - pd.Timedelta(days=1) for date in public_dates),
        last,
    ]
    public_regimes = [
        _summarise_regime(
            series, monthly, start, end, number
        )
        for number, (start, end) in enumerate(
            zip(public_starts, public_ends), start=1
        )
        if start <= end
    ]
    return {
        "boundaries": boundaries,
        "regimes": public_regimes,
        "metadata": metadata,
        "feature_scales": primary_scales,
    }


def check_step_shift(
    series: pd.Series,
    series_type: str = "unknown",
    long_record_min_years: float = 12.0,
    long_record_block_years: int = 4,
    short_record_divisor: float = 3.0,
    minimum_block_years: int = 1,
    minimum_valid_days_per_month: int = 10,
    minimum_block_coverage: float = 0.55,
    minimum_calendar_months: int = 8,
    structural_threshold: float = 3.0,
    tier_3_maximum_quantile: float = 0.05,
    tier_1_minimum_quantile: float = 0.25,
    tier_3_points: float = 0.0,
    tier_2_points: float = 1.0,
    tier_1_points: float = 2.0,
    composite_tier_2_minimum_score: float = 1.0,
    composite_tier_1_above_score: float = 1.5,
    refinement_block_fraction: float = 0.5,
    consolidation_max_block_widths: float = 2.0,
) -> dict:
    """Detect persistent adaptive-block shifts and assign Tier 1/2/3."""
    if not 0 <= tier_3_maximum_quantile < tier_1_minimum_quantile <= 1:
        raise ValueError(
            "Step-shift quantiles must satisfy 0 <= Tier 3 cutoff "
            "< Tier 1 cutoff <= 1."
        )
    if not tier_3_points <= tier_2_points <= tier_1_points:
        raise ValueError("Step-shift tier points must increase from Tier 3 to Tier 1.")
    if not (
        tier_3_points <= composite_tier_2_minimum_score
        < composite_tier_1_above_score <= tier_1_points
    ):
        raise ValueError(
            "Step-shift composite cutoffs must lie in increasing order within "
            "the configured boundary-tier point range."
        )
    if not 0 < refinement_block_fraction <= 1:
        raise ValueError("refinement_block_fraction must be greater than 0 and at most 1.")
    record = get_valid_record(series)
    if record.empty:
        return make_result(
            check="step_shift",
            flag=False,
            value=None,
            series_type=series_type,
            status="skipped",
            reason_skipped="No valid data found.",
            message="Level/regime shift not calculated: no valid data found.",
            regime_boundaries=[],
            regime_summary=[],
            adaptive_block_summary={},
            tier_1_count=0,
            tier_2_count=0,
            tier_3_count=0,
        )

    detected = _detect(
        record,
        long_record_min_years=long_record_min_years,
        long_record_block_years=long_record_block_years,
        short_record_divisor=short_record_divisor,
        minimum_block_years=minimum_block_years,
        minimum_valid_days_per_month=minimum_valid_days_per_month,
        minimum_block_coverage=minimum_block_coverage,
        minimum_calendar_months=minimum_calendar_months,
        structural_threshold=structural_threshold,
        tier_3_maximum_quantile=tier_3_maximum_quantile,
        tier_1_minimum_quantile=tier_1_minimum_quantile,
        refinement_block_fraction=refinement_block_fraction,
        consolidation_max_block_widths=consolidation_max_block_widths,
    )
    boundaries = detected["boundaries"]
    regimes = detected["regimes"]

    tier_1 = [item for item in boundaries if item["diagnosis"] == "Tier 1"]
    tier_2 = [item for item in boundaries if item["diagnosis"] == "Tier 2"]
    tier_3 = [item for item in boundaries if item["diagnosis"] == "Tier 3"]
    # A series-level diagnosis must reflect all retained structural boundaries,
    # rather than allowing one Tier 1 boundary to dominate the whole record.
    boundary_points = {
        "Tier 3": float(tier_3_points),
        "Tier 2": float(tier_2_points),
        "Tier 1": float(tier_1_points),
    }
    step_shift_score = (
        sum(boundary_points[item["diagnosis"]] for item in boundaries)
        / len(boundaries)
        if boundaries else 0.0
    )
    composite_tier = (
        "Tier 1" if step_shift_score > composite_tier_1_above_score
        else "Tier 2" if step_shift_score >= composite_tier_2_minimum_score
        else "Tier 3"
    )
    public = [*tier_1, *tier_2]
    if detected["metadata"]["primary_block_count"] < 3:
        return make_result(
            check="step_shift",
            flag=False,
            value=None,
            series_type=series_type,
            status="skipped",
            reason_skipped="Too few adequately observed adaptive blocks.",
            message=(
                "Level/regime shift not calculated: fewer than three adequately "
                "observed adaptive blocks were available."
            ),
            regime_boundaries=boundaries,
            regime_summary=regimes,
            adaptive_block_summary=detected["metadata"],
            tier_1_count=len(tier_1),
            tier_2_count=len(tier_2),
            tier_3_count=len(tier_3),
            retained_candidate_count=len(boundaries),
            step_shift_score=step_shift_score,
            composite_tier=composite_tier,
        )

    series_requires_attention = composite_tier in {"Tier 1", "Tier 2"}
    return make_result(
        check="step_shift",
        flag=series_requires_attention,
        value=len(public),
        flagged_timestamps=[item["boundary_timestamp"] for item in public],
        series_type=series_type,
        finding_status="candidate_detected" if series_requires_attention else "passed",
        message=(
            f"Level/regime shift screening retained {len(boundaries)} structural "
            f"candidate(s): {len(tier_1)} Tier 1, {len(tier_2)} Tier 2, "
            f"and {len(tier_3)} Tier 3. Composite score = "
            f"{step_shift_score:.3f} ({composite_tier})."
        ),
        tier_1_count=len(tier_1),
        tier_2_count=len(tier_2),
        tier_3_count=len(tier_3),
        step_shift_score=step_shift_score,
        composite_tier=composite_tier,
        retained_candidate_count=len(boundaries),
        regime_boundaries=boundaries,
        regime_summary=regimes,
        adaptive_block_summary=detected["metadata"],
        feature_scales=detected["feature_scales"],
        boundary_tier_points=boundary_points,
        resolved_settings={
            "long_record_min_years": long_record_min_years,
            "long_record_block_years": long_record_block_years,
            "short_record_divisor": short_record_divisor,
            "minimum_block_years": minimum_block_years,
            "minimum_valid_days_per_month": minimum_valid_days_per_month,
            "minimum_block_coverage": minimum_block_coverage,
            "minimum_calendar_months": minimum_calendar_months,
            "structural_threshold": structural_threshold,
            "tier_3_maximum_quantile": tier_3_maximum_quantile,
            "tier_1_minimum_quantile": tier_1_minimum_quantile,
            "composite_tier_2_minimum_score": composite_tier_2_minimum_score,
            "composite_tier_1_above_score": composite_tier_1_above_score,
            "refinement_block_fraction": refinement_block_fraction,
            "consolidation_max_block_widths": consolidation_max_block_widths,
        },
    )


__all__ = ["check_step_shift"]
