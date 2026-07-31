"""Monthly-baseline regime-shift screening for TriHydrA Layer 1.

The check deliberately avoids daily flood geometry. It reduces adequately
observed months to robust Q25/median/Q75 summaries, splits at unresolved
monthly gaps, and asks whether the *baseline and central distribution* change
for at least six calendar months on each side. Nearby proposals are
consolidated so one hydrological transition cannot create many shift lines.

The output describes persistent regime candidates, not proven data errors.
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import mannwhitneyu

from src.trihydra.layer1.check_result import make_result
from src.trihydra.layer1.timeseries_validity import get_valid_record


def _monthly_summary(
    series: pd.Series,
    minimum_valid_days_per_month: int,
) -> pd.DataFrame:
    """Return robust monthly quantiles only for adequately observed months."""
    daily = series.astype(float).groupby(series.index).median().sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar)
    monthly = pd.DataFrame({
        "q25": daily.resample("MS").quantile(0.25),
        "median": daily.resample("MS").median(),
        "q75": daily.resample("MS").quantile(0.75),
        "q90": daily.resample("MS").quantile(0.90),
        "valid_day_count": daily.resample("MS").count(),
    })
    adequate = monthly["valid_day_count"] >= minimum_valid_days_per_month
    monthly.loc[~adequate, ["q25", "median", "q75", "q90"]] = np.nan
    return monthly


def _continuous_monthly_periods(monthly: pd.DataFrame) -> list[pd.DataFrame]:
    """Split at every month whose median is unavailable."""
    valid = monthly["median"].notna()
    groups = (valid != valid.shift(fill_value=False)).cumsum()
    return [
        monthly.loc[mask.index]
        for _, mask in valid.groupby(groups)
        if bool(mask.iloc[0])
    ]


def _candidate_evidence(
    period: pd.DataFrame,
    position: int,
    window_months: int,
    minimum_standardised_effect: float,
) -> dict | None:
    """Evaluate baseline persistence and multi-quantile agreement."""
    before = period.iloc[position - window_months:position]
    after = period.iloc[position:position + window_months]
    if len(before) < window_months or len(after) < window_months:
        return None

    changes = {}
    for column in ["q25", "median", "q75", "q90"]:
        changes[column] = float(after[column].median() - before[column].median())
    median_values = pd.concat([before["median"], after["median"]])
    pooled_iqr = float(median_values.quantile(0.75) - median_values.quantile(0.25))
    median_effect = (
        abs(changes["median"]) / pooled_iqr
        if pooled_iqr > 0
        else np.inf if changes["median"] else 0.0
    )
    median_direction = np.sign(changes["median"])
    central_agreement = sum(
        np.sign(changes[column]) == median_direction and changes[column] != 0
        for column in ["q25", "median", "q75"]
    )
    # A high-flow-only difference has little movement in the median/baseline
    # but a pronounced Q90 change. Such a period is hydrological-event context,
    # not a whole-distribution step shift.
    flood_only = bool(
        median_effect < minimum_standardised_effect
        and abs(changes["q90"]) > abs(changes["median"])
    )
    p_value = float(
        mannwhitneyu(
            before["median"],
            after["median"],
            alternative="two-sided",
        ).pvalue
    )
    return {
        "boundary_timestamp": str(period.index[position]),
        "before_start": str(before.index[0]),
        "before_end": str(before.index[-1]),
        "after_start": str(after.index[0]),
        "after_end": str(after.index[-1]),
        "before_q25": float(before["q25"].median()),
        "after_q25": float(after["q25"].median()),
        "before_median": float(before["median"].median()),
        "after_median": float(after["median"].median()),
        "before_q75": float(before["q75"].median()),
        "after_q75": float(after["q75"].median()),
        "before_q90": float(before["q90"].median()),
        "after_q90": float(after["q90"].median()),
        "q25_change": changes["q25"],
        "median_change": changes["median"],
        "q75_change": changes["q75"],
        "q90_change": changes["q90"],
        "pooled_median_iqr": pooled_iqr,
        "standardised_median_effect": float(median_effect),
        "central_quantile_agreement": int(central_agreement),
        "flood_only": flood_only,
        "p_value": p_value,
    }


def _consolidate_candidates(
    candidates: list[dict],
    cooldown_months: int,
) -> tuple[list[dict], list[dict]]:
    """Keep the strongest representative inside each cooldown neighbourhood."""
    if not candidates:
        return [], []
    ordered = sorted(candidates, key=lambda x: pd.Timestamp(x["boundary_timestamp"]))
    groups: list[list[dict]] = [[ordered[0]]]
    for candidate in ordered[1:]:
        previous = pd.Timestamp(groups[-1][-1]["boundary_timestamp"])
        current = pd.Timestamp(candidate["boundary_timestamp"])
        if (current.year - previous.year) * 12 + current.month - previous.month <= cooldown_months:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    retained, suppressed = [], []
    for group_number, group in enumerate(groups, start=1):
        strongest = max(
            group,
            key=lambda x: (
                x["standardised_median_effect"],
                -x["p_value"],
            ),
        )
        strongest["candidate_group"] = group_number
        strongest["consolidated_proposal_count"] = len(group)
        retained.append(strongest)
        for candidate in group:
            if candidate is strongest:
                continue
            candidate["candidate_group"] = group_number
            candidate["suppression_reason"] = (
                f"Consolidated within {cooldown_months} months of stronger proposal."
            )
            suppressed.append(candidate)
    return retained, suppressed


def check_step_shift(
    series: pd.Series,
    series_type: str = "unknown",
    evidence_window_months: int = 6,
    cooldown_months: int = 6,
    significance_level: float = 0.05,
    minimum_standardised_effect: float = 0.75,
    minimum_valid_days_per_month: int = 20,
    include_flow_duration_curve: bool = False,
    min_regime_days: int | None = None,
    max_gap_days: int | None = None,
) -> dict:
    """Detect consolidated monthly baseline-regime candidates.

    ``min_regime_days`` and ``max_gap_days`` remain accepted for call
    compatibility but are no longer used: evidence and gaps are now explicitly
    calendar-month based.
    """
    record = get_valid_record(series)
    if record.empty:
        return make_result(
            check="step_shift",
            flag=False,
            value=None,
            series_type=series_type,
            status="skipped",
            reason_skipped="No valid data found.",
            message="Step shift not calculated: no valid data found.",
            regime_boundaries=[],
            regime_summary=[],
        )

    monthly = _monthly_summary(record, minimum_valid_days_per_month)
    periods = [
        period for period in _continuous_monthly_periods(monthly)
        if len(period) >= 2 * evidence_window_months
    ]
    raw_proposals = []
    period_metadata = []
    for period_number, period in enumerate(periods, start=1):
        anomaly = (
            period["median"]
            - period["median"].groupby(period.index.month).transform("median")
        )
        variance = float(np.var(anomaly))
        penalty = variance * np.log(len(anomaly)) if variance > 0 else 0.0
        breakpoints = (
            rpt.Pelt(
                model="l2",
                min_size=evidence_window_months,
            ).fit(anomaly.to_numpy()).predict(pen=penalty)
            if variance > 0 else [len(anomaly)]
        )
        period_metadata.append({
            "period": period_number,
            "start": str(period.index[0]),
            "end": str(period.index[-1]),
            "month_count": int(len(period)),
            "proposed_boundary_count": int(max(0, len(breakpoints) - 1)),
        })
        for position in breakpoints[:-1]:
            evidence = _candidate_evidence(
                period,
                position,
                evidence_window_months,
                minimum_standardised_effect,
            )
            if evidence:
                evidence["continuous_period"] = period_number
                raw_proposals.append(evidence)

    retained, suppressed = _consolidate_candidates(raw_proposals, cooldown_months)
    for candidate in retained:
        candidate["effect_threshold"] = minimum_standardised_effect
        candidate["significance_threshold"] = significance_level
        candidate["confirmed"] = bool(
            candidate["p_value"] < significance_level
            and candidate["standardised_median_effect"] >= minimum_standardised_effect
            and candidate["central_quantile_agreement"] >= 2
            and not candidate["flood_only"]
        )
        if candidate["flood_only"]:
            candidate["decision"] = "rejected_flood_only"
        elif candidate["central_quantile_agreement"] < 2:
            candidate["decision"] = "rejected_quantile_disagreement"
        elif candidate["standardised_median_effect"] < minimum_standardised_effect:
            candidate["decision"] = "rejected_small_effect"
        elif candidate["p_value"] >= significance_level:
            candidate["decision"] = "rejected_weak_evidence"
        else:
            candidate["decision"] = "confirmed"

    confirmed = [candidate for candidate in retained if candidate["confirmed"]]
    # Regimes are defined by confirmed monthly boundaries within each period.
    regime_summary = []
    for period_number, period in enumerate(periods, start=1):
        dates = sorted(
            pd.Timestamp(x["boundary_timestamp"])
            for x in confirmed
            if x["continuous_period"] == period_number
        )
        starts = [period.index[0]] + dates
        ends = [date - pd.offsets.MonthBegin(1) for date in dates] + [period.index[-1]]
        for regime_number, (start, end) in enumerate(zip(starts, ends), start=1):
            section = period.loc[start:end]
            regime_summary.append({
                "continuous_period": period_number,
                "regime": regime_number,
                "start": str(section.index[0]),
                "end": str(section.index[-1]),
                "calendar_month_count": int(len(section)),
                "valid_day_count": int(section["valid_day_count"].sum()),
                "q25": float(section["q25"].median()),
                "median": float(section["median"].median()),
                "q75": float(section["q75"].median()),
                "q90": float(section["q90"].median()),
            })

    return make_result(
        check="step_shift",
        flag=bool(confirmed),
        value=len(confirmed),
        flagged_timestamps=[x["boundary_timestamp"] for x in confirmed],
        series_type=series_type,
        finding_status="candidate_detected" if confirmed else "passed",
        message=(
            f"Monthly PELT proposed {len(raw_proposals)} boundary location(s); "
            f"{len(retained)} remained after {cooldown_months}-month consolidation, "
            f"and {len(confirmed)} met six-month baseline, effect, p-value, and "
            "multi-quantile requirements. Flood-only differences were rejected."
        ),
        raw_proposal_count=len(raw_proposals),
        consolidated_candidate_count=len(retained),
        confirmed_boundary_count=len(confirmed),
        rejected_boundary_count=len(retained) - len(confirmed),
        consolidated_away_count=len(suppressed),
        regime_boundaries=retained,
        suppressed_proposals=suppressed,
        regime_summary=regime_summary,
        continuous_periods=period_metadata,
        evidence_window_months=evidence_window_months,
        cooldown_months=cooldown_months,
        minimum_valid_days_per_month=minimum_valid_days_per_month,
    )
