"""Assemble domain-owned text sections into readable result reports."""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
import pandas as pd

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
    observation = rows.loc[rows["series_role"] == "observation"].iloc[0]
    layer1_class = observation.get("layer1_class", "Not assessed")
    layer1_score = observation.get("layer1_score")
    status = value(layer1_class)
    score_text = "" if pd.isna(layer1_score) else f"  (score {int(layer1_score)})"
    valid = result.station.obs.dropna()
    lines = _title("TRIHYDRA SCREENING SUMMARY")
    lines += ["", f"  Station: {result.station_id:<35} Status: {status}{score_text}"]
    lines += section("STATION") + [
        field("Station ID", result.station_id, 2),
        field("Series", "observation (role: Observation)", 2),
        field("Units", result.station.unit, 2),
        field("Assessment period", "not provided" if valid.empty else f"{date(valid.index.min())} to {date(valid.index.max())}", 2),
        field("Valid observations", int(valid.size), 2),
    ]
    for label, key in (
        ("River", "river_name"), ("Catchment", "catchment_name"),
        ("Latitude", "latitude"), ("Longitude", "longitude"),
        ("Catchment area", "catchment_area_km2"),
    ):
        if key in observation and pd.notna(observation.get(key)):
            suffix = " km2" if key == "catchment_area_km2" else ""
            lines.append(field(label, value(observation.get(key), suffix=suffix), 2))

    lines += section("FINAL ASSESSMENT") + [
        field("Review status", status, 2),
        field("Layer 1 score", value(layer1_score), 2),
        "",
        "  " + ("No material data-quality concerns were detected." if status == "No review needed" else "The screening result contains concerns requiring interpretation."),
        "", line("="), "FINDINGS".center(WIDTH), line("="),
    ]
    lines += render_layer1_summary(observation, result.station.unit)
    lines += render_layer2_summary(observation, result.station.unit)
    lines += render_comparison_summary(result, rows)
    lines += render_layer3_summary(observation, result.layer3)

    lines += ["", line("="), "APPENDIX".center(WIDTH), line("=")]
    lines += section("IMPORTANT THRESHOLDS USED")
    threshold_rows = []
    for name, threshold in observation.items():
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


def render_network_summary(summary: pd.DataFrame) -> str:
    """Render one compact index for a completed multi-station run."""
    lines = _title("TRIHYDRA NETWORK SUMMARY") + ["", field("Station-series assessments", len(summary), 2)]
    for _, row in summary.iterrows():
        lines += section(f"{row.get('station_id')}  -  {row.get('series_name', 'observation')}") + [
            field("Series role", value(row.get("series_role")), 2),
            field("Layer 1 status", value(row.get("layer1_class")), 2),
            field("Layer 1 score", value(row.get("layer1_score")), 2),
            field("Layer 3 status", value(row.get("layer3_status")), 2),
        ]
        if pd.notna(row.get("layer3_context_agreement_class")):
            lines.append(field("Contextual agreement", value(row.get("layer3_context_agreement_class")), 2))
    return "\n".join(lines + ["", line("="), ""])


__all__ = ["render_evidence_report", "render_network_summary", "render_station_summary"]
