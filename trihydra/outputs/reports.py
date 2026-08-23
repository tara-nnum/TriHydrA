"""Assemble domain-owned text sections into readable result reports."""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from trihydra.outputs.network_diagnostics import network_assessment_counts

from trihydra.comparison.diagnostics import render_comparison_summary
from trihydra.layer1.diagnostics import render_layer1_summary
from trihydra.layer2.diagnostics import render_layer2_summary
from trihydra.layer3.diagnostics import render_layer3_summary, render_layer3_thresholds
from trihydra.result import TriHydrAResult
from trihydra.formatting import WIDTH, date, field, line, section, value


def _title(text: str) -> list[str]:
    return [line("="), text.center(WIDTH), line("=")]


def render_station_summary(result: TriHydrAResult) -> str:
    """Assemble one station report without interpreting layer internals here."""
    rows = result.summary
    primary_rows = rows.loc[rows["series_name"] == result.station.series1_name]
    primary = primary_rows.iloc[0] if not primary_rows.empty else rows.iloc[0]
    layer1_class = primary.get("layer1_class", "Not assessed")
    layer1_score = primary.get("layer1_score")
    layer1_percent = primary.get("layer1_score_percent")
    layer1_scope = primary.get("layer1_assessment_scope", "Full")
    scope_conclusion = primary.get("layer1_scope_conclusion")
    status = value(layer1_class)
    if str(layer1_scope).casefold() == "focused":
        status += " (selected checks only)"
    score_text = (
        "" if pd.isna(layer1_percent)
        else f"  ({float(layer1_percent):.1f}% of enabled assessable maximum)"
    )
    valid = result.station.obs.dropna()
    selected_start = primary.get("selected_calendar_start")
    selected_end = primary.get("selected_calendar_end")
    selected_period = (
        "not available" if pd.isna(selected_start) or pd.isna(selected_end)
        else f"{date(selected_start)} to {date(selected_end)}"
    )
    requested_mode = primary.get("requested_timespan_mode")
    if requested_mode == "range":
        requested_period = f"{date(primary.get('requested_start'))} to {date(primary.get('requested_end'))}"
    else:
        requested_period = "full available record"
    lines = _title("TRIHYDRA SCREENING SUMMARY")
    lines += ["", f"  Station: {result.station_id:<35} Status: {status}{score_text}"]
    lines += section("STATION") + [
        field("Station ID", result.station_id, 2),
        field("Series", f"{result.station.series1_name} (role: {result.station.series1_role})", 2),
        field("Units", result.station.unit, 2),
        field("Requested timespan", requested_period, 2),
        field("Selected calendar", selected_period, 2),
        field("First to last valid", "not available" if valid.empty else f"{date(valid.index.min())} to {date(valid.index.max())}", 2),
        field("Valid observations", int(valid.size), 2),
    ]
    for label, key in (
        ("River", "river_name"), ("Catchment", "catchment_name"),
        ("Latitude", "latitude"), ("Longitude", "longitude"),
        ("Catchment area", "catchment_area_km2"),
    ):
        if key in primary and pd.notna(primary.get(key)):
            suffix = " km2" if key == "catchment_area_km2" else ""
            lines.append(field(label, value(primary.get(key), suffix=suffix), 2))

    lines += section("FINAL ASSESSMENT") + [
        field("Assessment scope", value(layer1_scope), 2),
        field(
            "Enabled composite checks",
            f"{value(primary.get('layer1_enabled_check_count'))}/"
            f"{value(primary.get('layer1_total_composite_check_count'))}",
            2,
        ),
        field(
            "Assessable enabled checks",
            f"{value(primary.get('layer1_assessable_check_count'))}/"
            f"{value(primary.get('layer1_enabled_check_count'))}",
            2,
        ),
        field(
            "Evidence coverage",
            value(primary.get("layer1_evidence_coverage_percent"), decimals=1, suffix="%"),
            2,
        ),
        field("Review status", status, 2),
        field("Layer 1 score", value(layer1_score), 2),
        field(
            "Layer 1 normalized score",
            value(layer1_percent, decimals=1, suffix="%"),
            2,
        ),
        "",
        "  " + (
            value(scope_conclusion)
            if pd.notna(scope_conclusion)
            else "No material data-quality concerns were detected."
            if status == "No review needed"
            else "The screening result contains concerns requiring interpretation."
        ),
        "", line("="), "FINDINGS".center(WIDTH), line("="),
    ]
    lines += render_layer1_summary(primary, result.station.unit)
    lines += render_layer2_summary(primary, result.station.unit)
    lines += render_comparison_summary(result, rows)
    lines += render_layer3_summary(primary, result.layer3)

    lines += ["", line("="), "APPENDIX".center(WIDTH), line("=")]
    lines += section("IMPORTANT THRESHOLDS USED")
    threshold_rows = []
    for name, threshold in primary.items():
        if not str(name).startswith("threshold_") or pd.isna(threshold):
            continue
        label = str(name).removeprefix("threshold_").replace("_", " ").capitalize()
        threshold_rows.append(field(label, threshold, 2))
    lines += threshold_rows or ["  No configurable thresholds were recorded."]
    lines += render_layer3_thresholds(result.layer3)
    lines += section("FILES AVAILABLE") + [
        "  Interactive diagnostics", "    layer1.html", "    layer2.html",
        "", "  Detailed evidence", "    layer1_evidence.txt", "    layer2_evidence.txt",
    ]
    if result.comparison is not None:
        lines.append("    comparison_evidence.txt")
    if result.layer3 is not None:
        lines.append("    layer3_evidence.txt")
    lines += [
        "", line("="),
        "This report is a screening summary. A review classification indicates that",
        "inspection is recommended; it does not automatically mean that the input data",
        "are erroneous.", line("="), "",
    ]
    return "\n".join(lines)


def _display_name(value_: object) -> str:
    return str(value_).replace("_", " ").strip().upper()


def _evidence_value(value_: object, column: str = "") -> str:
    if value_ is None or (not isinstance(value_, str) and pd.isna(value_)):
        return "not provided"
    if column in {"start", "end", "timestamp", "peak_date", "event_start", "event_end"}:
        return date(value_)
    if isinstance(value_, (float, np.floating)) and float(value_).is_integer():
        return str(int(value_))
    if isinstance(value_, (float, np.floating)):
        return f"{float(value_):.3f}"
    if isinstance(value_, pd.Timestamp):
        return date(value_)
    return str(value_)


def _evidence_block(frame: pd.DataFrame) -> list[str]:
    """Render heterogeneous evidence without producing a sparse wide table."""
    frame = frame.dropna(axis=1, how="all").copy()
    hidden = {"station_id", "series_name", "series_role", "evidence_type"}
    columns = [column for column in frame.columns if column not in hidden]
    if not columns:
        return ["  No additional values were recorded."]
    lines: list[str] = []
    for number, (_, row) in enumerate(frame.iterrows(), start=1):
        if len(frame) > 1:
            lines.append(f"  Record {number}")
        for column in columns:
            value_ = row.get(column)
            if value_ is None or (not isinstance(value_, str) and pd.isna(value_)):
                continue
            rendered = _evidence_value(value_, str(column))
            label = str(column).replace("_", " ").capitalize()
            if len(rendered) <= 72:
                lines.append(field(label, rendered, 4))
            else:
                lines.append(" " * 4 + label + ":")
                lines.extend(" " * 6 + part for part in textwrap.wrap(rendered, 68))
        if len(frame) > 1 and number < len(frame):
            lines.append("")
    return lines


def render_evidence_report(station_id: str, layer_name: str, evidence: pd.DataFrame) -> str:
    """Render one domain's complete evidence as a grouped TXT report."""
    lines = _title(f"TRIHYDRA {layer_name.upper()} EVIDENCE") + ["", field("Station ID", station_id, 2)]
    if evidence is None or evidence.empty:
        return "\n".join(lines + ["", "  No detailed evidence was recorded.", "", line("=")]) + "\n"
    groups = evidence.groupby("evidence_type", sort=False, dropna=False) if "evidence_type" in evidence else [("evidence", evidence)]
    for name, frame in groups:
        lines += section(_display_name(name)) + _evidence_block(frame.reset_index(drop=True))
    return "\n".join(lines + ["", line("="), "End of evidence report.".center(WIDTH), line("="), ""])


def render_network_summary(
    summary: pd.DataFrame,
    diagnostic_summary: pd.DataFrame | None = None,
) -> str:
    """Render one compact index for a completed multi-station run."""
    counts = network_assessment_counts(summary)
    lines = _title("TRIHYDRA NETWORK SUMMARY") + [
        "",
        field("Unique stations", counts["unique_station_count"], 2),
        field("Station-series assessments", counts["station_series_assessment_count"], 2),
        "",
        "  Physical stations",
        field("Needs review (one or more series)", counts["needs_review_station_count"], 4),
        field("Minor concerns only", counts["minor_concerns_station_count"], 4),
        field("No Layer 1 concerns", counts["no_concerns_station_count"], 4),
        field("Not assessed", counts["not_assessed_station_count"], 4),
        "",
        "  Independently assessed series",
        field("Needs review", counts["needs_review_series_count"], 4),
        field("Minor concerns", counts["minor_concerns_series_count"], 4),
        field("No review needed", counts["no_review_series_count"], 4),
        field("Not assessed", counts["not_assessed_series_count"], 4),
    ]
    if diagnostic_summary is not None and not diagnostic_summary.empty:
        lines += section("Layer 1 checks causing concerns") + [
            "  Trigger rate uses enabled, assessable station-series only.",
            "",
            "  Check                    Assessable  Tier 1  Tier 2  Triggered    Rate",
            "  " + "-" * 76,
        ]
        for _, row in diagnostic_summary.iterrows():
            rate = value(row.get("trigger_rate_percent"), decimals=1, suffix="%")
            lines.append(
                f"  {_display_name(row['diagnostic']):<25}"
                f"{int(row['assessable_series_count']):>10}"
                f"{int(row['tier1_count']):>7}"
                f"{int(row['tier2_count']):>8}"
                f"{int(row['concern_series_count']):>11}"
                f"{rate:>10}"
            )
    for _, row in summary.iterrows():
        lines += section(f"{row.get('station_id')}  -  {row.get('series_name', 'observation')}") + [
            field("Series role", value(row.get("series_role")), 2),
            field(
                "Layer 1 status",
                value(row.get("layer1_class"))
                + (
                    " (selected checks only)"
                    if str(row.get("layer1_assessment_scope", "")).casefold()
                    == "focused"
                    else ""
                ),
                2,
            ),
            field("Assessment scope", value(row.get("layer1_assessment_scope")), 2),
            field(
                "Enabled/total checks",
                f"{value(row.get('layer1_enabled_check_count'))}/"
                f"{value(row.get('layer1_total_composite_check_count'))}",
                2,
            ),
            field("Layer 1 score", value(row.get("layer1_score")), 2),
            field(
                "Layer 1 normalized score",
                value(row.get("layer1_score_percent"), decimals=1, suffix="%"),
                2,
            ),
            field("Layer 3 status", value(row.get("layer3_status")), 2),
        ]
        if pd.notna(row.get("layer3_context_agreement_class")):
            lines.append(field("Contextual agreement", value(row.get("layer3_context_agreement_class")), 2))
    return "\n".join(lines + ["", line("="), ""])


__all__ = ["render_evidence_report", "render_network_summary", "render_station_summary"]
