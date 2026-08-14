"""Human-readable diagnostics for a two-series comparison."""

import pandas as pd

from trihydra.formatting import date, field, metric, section, value


def _name(component: object) -> str:
    return {
        "flow_behaviour": "Flow distribution",
        "annual_flashiness_shape": "Annual flashiness pattern",
        "annual_baseflow_shape": "Annual baseflow pattern",
        "seasonal_profile_shape": "Seasonal profile shape",
        "seasonal_timing": "Wettest/driest-month timing",
        "event_time_to_peak": "Event time to peak",
        "event_duration": "Event duration",
        "representative_event_shape": "Representative-event shape",
    }.get(str(component), str(component).replace("_", " ").capitalize())


def _measured(row: pd.Series) -> str:
    measured = row.get("value")
    if measured is None or pd.isna(measured):
        return "not assessable"
    method = str(row.get("metric", ""))
    if "difference (days)" in method:
        return f"{float(measured):.1f} days difference"
    if "month separation" in method:
        return f"{float(measured):.1f} month(s) difference"
    return f"{float(measured):.3f} {method.lower()}".strip()


def render_comparison_summary(result, rows: pd.DataFrame) -> list[str]:
    """Render the final score and every component that produced it."""
    lines = section("COMPARISON")
    candidate_rows = rows.loc[rows["series_role"] == "simulation"]
    if candidate_rows.empty or result.comparison is None:
        return lines + ["  Not performed - only one series was supplied."]
    candidate = candidate_rows.iloc[0]
    lines += [
        field("Candidate series", candidate.get("series_name"), 2),
        field("Common comparison period", f"{date(candidate.get('comparison_common_start_date'))} to {date(candidate.get('comparison_common_end_date'))}", 2),
        field("Pairwise-valid daily values", metric(candidate, "comparison_pairwise_valid_count"), 2),
        field("Layer 2 comparison score", metric(candidate, "comparison_layer2_comparison_score"), 2),
        field("Comparison classification", metric(candidate, "comparison_layer2_comparison_class"), 2),
        field("Assessable components", metric(candidate, "comparison_assessable_component_count"), 2),
    ]
    evidence = result.comparison.get("evidence", pd.DataFrame())
    components = evidence.loc[evidence.get("evidence_type", pd.Series(dtype=str)) == "component_score"] if isinstance(evidence, pd.DataFrame) else pd.DataFrame()
    if not components.empty:
        lines += ["", "  Scored comparison components"]
        for _, component in components.iterrows():
            lines += [
                field(_name(component.get("component")), _measured(component), 4),
                field("Classification / score contribution", f"{value(component.get('tier'))}; {value(component.get('contribution'), decimals=1)} point(s)", 6),
            ]
    return lines


__all__ = ["render_comparison_summary"]
