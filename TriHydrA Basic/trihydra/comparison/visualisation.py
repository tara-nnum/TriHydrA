"""Interactive figures for source-neutral series comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from trihydra.plotting._shared import (
    CHECK_LABELS_L1,
    DEFAULT_OUTPUT_ROOT,
    FLAG_MARKER_OPACITY,
    FLAG_MARKER_SIZE,
    LAYER1_CHECK_COLORS,
    MODEL_COLOR,
    MODEL_SHADE,
    OBS_COLOR,
    OBS_SHADE,
    PLOT_TEMPLATE,
    _apply_layer2_number_format,
    _flow_duration_curve,
    _format_display_number,
    _group_into_spans,
    _save,
    _save_csv_resilient,
    _save_dashboard,
    _shapes_for_spans,
)
from trihydra.layer2.visualisation import layer2_typical_rows

# ==========================================================
# GENERIC SAME-STATION COMPARISON OVERVIEW
# ==========================================================

REFERENCE_COLOR = "#3E8E74"
CANDIDATE_COLOR = "#9A6F8F"


def _comparison_event_trace(
    fig: go.Figure, diagnostics: dict, series: pd.Series,
    name: str, color: str, dash: str
) -> None:
    event = diagnostics["representative_event"]
    events = diagnostics["hydrograph_events"]
    if event.empty or events.empty:
        return
    selected = event.iloc[0]
    event_id = selected["event_id"]
    values = series.loc[selected["event_start"]:selected["event_end"]].dropna()
    if values.empty:
        return
    x = (values.index - values.index[0]).days
    y = values.values
    dates = values.index.strftime("%Y-%m-%d")
    event_start = pd.Timestamp(selected["event_start"]).strftime("%Y-%m-%d")
    peak_date = pd.Timestamp(selected["peak_date"]).strftime("%Y-%m-%d")
    event_end = pd.Timestamp(selected["event_end"]).strftime("%Y-%m-%d")
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=f"{name} representative event",
        line=dict(color=color, width=2.5, dash=dash), opacity=0.85,
        legend="legend2", customdata=dates,
        hovertemplate=(
            f"{name}<br>Event {event_id}<br>Start: {event_start}"
            f"<br>Peak: {peak_date}<br>End: {event_end}"
            "<br>Current date: %{customdata}<br>Relative day=%{x:.0f}"
            "<br>Discharge=%{y:.3f}<extra></extra>"
        ),
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=[selected["time_to_peak_days"]], y=[selected["peak_flow"]],
        mode="markers", name=f"{name} event peak",
        marker=dict(color=color, size=9, symbol="diamond"),
        showlegend=True, legend="legend2",
        hovertemplate=(
            f"{name} event peak<br>Date: {peak_date}"
            "<br>Relative day=%{x:.0f}<br>Discharge=%{y:.3f}<extra></extra>"
        ),
    ), row=3, col=2)
    peak_day = float(selected["time_to_peak_days"])
    fig.add_vline(
        x=peak_day, line_color=color, line_dash=dash, line_width=1.5,
        row=3, col=2,
    )


def build_comparison_overview(result: dict) -> go.Figure:
    """Build a descriptive overlay without assigning concern thresholds."""
    reference_name = result["reference_name"]
    candidate_name = result["candidate_name"]
    overlay = result["overlay_data"]
    reference_l2 = result["reference_comparison_layer2"]
    candidate_l2 = result["candidate_comparison_layer2"]
    fig = make_subplots(
        rows=6, cols=2,
        specs=[
            [{"colspan": 2}, None], [{"colspan": 2}, None], [{}, {}],
            [{"colspan": 2}, None], [{"type": "table", "colspan": 2}, None],
            [{"type": "table", "colspan": 2}, None],
        ],
        row_heights=[0.19, 0.19, 0.16, 0.16, 0.15, 0.15],
        horizontal_spacing=0.16, vertical_spacing=0.09,
        subplot_titles=(
            "Pairwise-comparable hydrograph", "Annual flow statistics",
            "Annual catchment-response indices", "Representative high-flow events",
            "Historical seasonality", "Diagnostic comparison",
            "Layer 2 composite assessment",
        ),
    )
    for column, name, color, dash in [
        ("reference", reference_name, REFERENCE_COLOR, "solid"),
        ("candidate", candidate_name, CANDIDATE_COLOR, "dot"),
    ]:
        fig.add_trace(go.Scatter(
            x=overlay.index, y=overlay[column], mode="lines", name=name,
            line=dict(color=color, width=1.1, dash=dash), opacity=0.85,
            hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<br>Discharge=%{{y:.3f}}<extra></extra>",
        ), row=1, col=1)

    flow_metrics = [
        ("maximum_flow", "Maximum", 0.36, 0.50),
        ("mean_flow", "Mean", 0.29, 0.72),
        ("median_flow", "Median", 0.22, 0.82),
        ("minimum_flow", "Minimum", 0.14, 0.90),
    ]
    for diagnostics, source_name, color, dash in [
        (reference_l2, reference_name, REFERENCE_COLOR, "solid"),
        (candidate_l2, candidate_name, CANDIDATE_COLOR, "dot"),
    ]:
        annual = diagnostics["annual_signatures"]
        x_offset = -0.20 if dash == "solid" else 0.20
        for metric, label, width, opacity in flow_metrics:
            fig.add_trace(go.Bar(
                x=annual["year"] + x_offset, y=annual[metric], width=width,
                name=f"{source_name} {label}",
                marker=dict(color=color, line=dict(width=0)), opacity=opacity,
            ), row=2, col=1)
        years = annual["year"]
        if not annual.empty:
            line_x = [years.min(), years.max()]
            for key, label in [
                ("q05_percentile_low_flow_fdc_q95", "Q05"),
                ("q95_percentile_high_flow_fdc_q05", "Q95"),
            ]:
                value = diagnostics["references"][key]
                fig.add_trace(go.Scatter(
                    x=line_x, y=[value, value], mode="lines",
                    name=f"{source_name} {label}",
                    line=dict(color=color, width=1.5, dash=dash), opacity=0.85,
                ), row=2, col=1)

    for metric, label, color in [
        ("flashiness_index", "Flashiness", "#C08A3E"),
        ("baseflow_index", "Baseflow index", "#4F8A70"),
        ("seasonality_index", "Seasonality index", "#7A5C91"),
    ]:
        for diagnostics, source_name, dash in [
            (reference_l2, reference_name, "solid"),
            (candidate_l2, candidate_name, "dot"),
        ]:
            annual = diagnostics["annual_signatures"].dropna(subset=[metric])
            fig.add_trace(go.Scatter(
                x=annual["year"], y=annual[metric], mode="lines",
                name=f"{source_name} {label}",
                line=dict(color=color, width=2, dash=dash), opacity=0.85,
            ), row=3, col=1)

    _comparison_event_trace(
        fig, reference_l2, overlay["reference"],
        reference_name, REFERENCE_COLOR, "solid"
    )
    _comparison_event_trace(
        fig, candidate_l2, overlay["candidate"],
        candidate_name, CANDIDATE_COLOR, "dot"
    )

    for diagnostics, source_name, color, dash in [
        (reference_l2, reference_name, REFERENCE_COLOR, "solid"),
        (candidate_l2, candidate_name, CANDIDATE_COLOR, "dot"),
    ]:
        profile = diagnostics["seasonality_profile"]
        fig.add_trace(go.Scatter(
            x=profile["month"], y=profile["median"], mode="lines",
            name=f"{source_name} historical seasonality",
            line=dict(color=color, width=2.7, dash=dash), opacity=0.85,
        ), row=4, col=1)
        valid = profile.dropna(subset=["median"])
        if not valid.empty:
            wet = valid.loc[valid["median"].idxmax()]
            dry = valid.loc[valid["median"].idxmin()]
            wet_color = "#2D82B7" if dash == "solid" else "#2CB7C9"
            dry_color = "#C65B58" if dash == "solid" else "#D98770"
            for row, marker_name, marker_color in [
                (wet, "wettest", wet_color), (dry, "driest", dry_color),
            ]:
                fig.add_trace(go.Scatter(
                    x=[row["month"]], y=[row["median"]], mode="markers",
                    name=f"{source_name} {marker_name} month",
                    marker=dict(color=marker_color, size=10, symbol="circle"),
                ), row=4, col=1)
    reference_rows = dict(layer2_typical_rows(reference_l2))
    candidate_rows = dict(layer2_typical_rows(candidate_l2))
    metrics = list(reference_rows)
    reference_display, candidate_display, difference_display, comments = [], [], [], []
    for metric in metrics:
        reference_value = reference_rows.get(metric, np.nan)
        candidate_value = candidate_rows.get(metric, np.nan)
        is_month = "month" in metric.lower()
        if pd.isna(reference_value) or pd.isna(candidate_value):
            reference_display.append("Not available" if pd.isna(reference_value) else f"{float(reference_value):.3f}")
            candidate_display.append("Not available" if pd.isna(candidate_value) else f"{float(candidate_value):.3f}")
            difference_display.append("Not available")
            comments.append("Insufficient values for comparison.")
            continue
        if is_month:
            reference_month = int(reference_value)
            candidate_month = int(candidate_value)
            month_distance = min(abs(candidate_month - reference_month), 12 - abs(candidate_month - reference_month))
            reference_display.append(pd.Timestamp(2000, reference_month, 1).strftime("%b"))
            candidate_display.append(pd.Timestamp(2000, candidate_month, 1).strftime("%b"))
            difference_display.append(f"{month_distance} month(s)")
            comments.append("Same typical month." if month_distance == 0 else "Typical months differ.")
        else:
            difference = float(candidate_value) - float(reference_value)
            reference_display.append(f"{float(reference_value):.3f}")
            candidate_display.append(f"{float(candidate_value):.3f}")
            difference_display.append(f"{difference:.3f}")
            if metric == "Median event time to peak (days)":
                absolute_days = abs(difference)
                tier = (
                    "Tier 3" if absolute_days <= 3.0
                    else "Tier 2" if absolute_days < 5.0
                    else "Tier 1"
                )
                comments.append(
                    f"{tier}: time to peak differs by {absolute_days:.3f} days."
                )
            else:
                comments.append(
                    "No difference." if difference == 0
                    else "Candidate is higher." if difference > 0
                    else "Candidate is lower."
                )
    fig.add_trace(go.Table(
        columnwidth=[0.32, 0.15, 0.15, 0.15, 0.23],
        header=dict(
            values=["Diagnostic", reference_name, candidate_name, "Difference", "Comment"],
            fill_color="#E9ECEF", align="left",
        ),
        cells=dict(
            values=[metrics, reference_display, candidate_display, difference_display, comments],
            align="left",
        ),
    ), row=5, col=1)
    composite = result["layer2_composite"]["components"]
    composite_summary = result["layer2_composite"]["summary"].iloc[0]
    fig.add_trace(go.Table(
        columnwidth=[0.28, 0.25, 0.14, 0.12, 0.10, 0.11],
        header=dict(
            values=["Component", "Metric", "Value", "Tier", "Points", "Assessable"],
            fill_color="#E9ECEF", align="left",
        ),
        cells=dict(values=[
            composite["component"], composite["metric"],
            [f"{value:.3f}" if pd.notna(value) else "Not available" for value in composite["value"]],
            composite["tier"], composite["contribution"],
            composite["assessable"].map({True: "Yes", False: "No"}),
        ], align="left"),
    ), row=6, col=1)
    fig.update_layout(
        title=f"{result['station_id']} - comparison overview",
        template="plotly_white", height=2050, hovermode="closest",
        barmode="overlay",
        legend=dict(orientation="v", x=1.01, y=1.0),
        legend2=dict(
            title=dict(text="High-flow events"), orientation="v",
            x=1.01, y=0.48, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#D0D5DD",
            borderwidth=1,
        ),
        margin=dict(r=300),
    )
    fig.add_annotation(
        x=1.0, y=0.0, xref="paper", yref="paper", xanchor="right",
        yanchor="bottom", showarrow=False,
        text=(
            f"Layer 2 score: {int(composite_summary['layer2_score'])} - "
            f"{composite_summary['layer2_class']}"
        ),
        bgcolor="rgba(244,247,251,0.96)", bordercolor="#B7C4CF",
        borderwidth=1, borderpad=5,
    )
    fig.update_yaxes(title_text="Discharge", row=1, col=1)
    fig.update_yaxes(title_text="Discharge", row=2, col=1)
    fig.update_yaxes(title_text="Index", row=3, col=1)
    fig.update_yaxes(title_text="Discharge", row=3, col=2)
    fig.update_yaxes(title_text="Monthly median discharge", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Calendar year", row=2, col=1)
    fig.update_xaxes(title_text="Calendar year", row=3, col=1)
    fig.update_xaxes(title_text="Relative event day", row=3, col=2)
    fig.update_xaxes(
        title_text="Month", tickmode="array", tickvals=list(range(1, 13)),
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        row=4, col=1,
    )
    return fig


def generate_comparison_visuals(
    result: dict, output_root: str | Path, *, show: bool = False,
    flat_filename: str | None = None,
) -> Path:
    directory = (
        Path(output_root) if flat_filename
        else Path(output_root) / result["station_id"] / "comparison"
    )
    directory.mkdir(parents=True, exist_ok=True)
    figure = build_comparison_overview(result)
    if show:
        figure.show()
    path = directory / (flat_filename or "comparison_overview.html")
    figure.write_html(path, include_plotlyjs="cdn")
    return path
