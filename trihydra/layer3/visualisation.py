"""Compact interactive Layer 3 context dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from trihydra.layer3.evidence import StationContextEvidence
from trihydra.layer3.orchestrator import Layer3StationResult


GREEN = "#4F9B83"
MAUVE = "#9A7197"
CYAN = "#35A8BF"
ORANGE = "#C97855"
YELLOW = "#D7B642"
RED = "#C94F4F"
NEUTRAL = "#9AA5B1"

LAYER3_PLOT_MODES = {"recommended", "all", "none"}


def _plot_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in LAYER3_PLOT_MODES:
        allowed = ", ".join(sorted(LAYER3_PLOT_MODES))
        raise ValueError(f"Layer 3 plot mode must be one of: {allowed}.")
    return mode


def _overall_assessment(fig: go.Figure, result: Layer3StationResult) -> None:
    summary = result.summary
    checks = summary.combined_check_summary.copy()
    rows = [
        ("<b>Overall contextual agreement</b>", summary.combined_similarity_percent),
        ("Nearby-gauge hydrological agreement", summary.local.similarity_percent),
        ("Comparable-catchment hydrological agreement", summary.analogues.similarity_percent),
    ] + [
        (str(row.check).replace("Representative ", ""), row.combined_similarity_percent)
        for row in checks.itertuples()
    ]
    labels = [label for label, _ in rows]
    scores = [score for _, score in rows]
    partial = float(summary.partial_minimum_percent)
    strong = float(summary.similar_minimum_percent)

    def category(score):
        if score is None or pd.isna(score):
            return "Not assessed"
        if float(score) >= strong:
            return "Strong match"
        if float(score) >= partial:
            return "Moderate match"
        return "Poor match"

    # Centre the display at 50 without changing the underlying 0-100 scores.
    # Negative bars therefore show weaker agreement and positive bars stronger
    # agreement. Classification still uses the configured Layer 3 thresholds.
    display_scores = [
        None if score is None or pd.isna(score)
        else float(np.clip((float(score) - 50.0) * 2.0, -100.0, 100.0))
        for score in scores
    ]
    categories = [category(score) for score in scores]
    hover = [
        f"{label}<br>Agreement: {'Not assessed' if score is None or pd.isna(score) else f'{float(score):.1f}%'}"
        f"<br>Classification: {state}"
        f"<br>Moderate threshold: {partial:.1f}%"
        f"<br>Strong threshold: {strong:.1f}%"
        for label, score, state in zip(labels, scores, categories)
    ]

    partial_display = float(np.clip((partial - 50.0) * 2.0, -100.0, 100.0))
    strong_display = float(np.clip((strong - 50.0) * 2.0, -100.0, 100.0))
    fig.add_vrect(x0=-100, x1=partial_display, fillcolor=ORANGE, opacity=0.08,
                  line_width=0, row=1, col=1)
    fig.add_vrect(x0=partial_display, x1=strong_display, fillcolor=YELLOW, opacity=0.09,
                  line_width=0, row=1, col=1)
    fig.add_vrect(x0=strong_display, x1=100, fillcolor=GREEN, opacity=0.08,
                  line_width=0, row=1, col=1)

    positive_colors = sample_colorscale([[0.0, YELLOW], [1.0, GREEN]],
                                        [(index + 0.5) / 20 for index in range(20)])
    negative_colors = sample_colorscale([[0.0, ORANGE], [1.0, RED]],
                                        [(index + 0.5) / 20 for index in range(20)])
    for segment in range(20):
        start = segment * 5.0
        positive = [
            0.0 if value is None else float(np.clip(value - start, 0.0, 5.0))
            for value in display_scores
        ]
        fig.add_trace(
            go.Bar(
                x=positive, y=labels, orientation="h", width=0.42,
                marker={"color": positive_colors[segment], "line": {"width": 0}},
                customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            ), row=1, col=1,
        )
        negative = [
            0.0 if value is None else -float(np.clip((-value) - start, 0.0, 5.0))
            for value in display_scores
        ]
        fig.add_trace(
            go.Bar(
                x=negative, y=labels, orientation="h", width=0.42,
                marker={"color": negative_colors[segment], "line": {"width": 0}},
                customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            ), row=1, col=1,
        )

    result_text = [
        "Not assessed" if score is None or pd.isna(score)
        else f"{float(score):.1f}% · {state.replace(' match', '')}"
        for score, state in zip(scores, categories)
    ]
    fig.add_trace(
        go.Scatter(
            x=[104] * len(labels), y=labels, mode="text", text=result_text,
            textposition="middle right", textfont={"size": 10, "color": "#17365D"},
            customdata=hover, hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ), row=1, col=1,
    )
    fig.add_vline(x=0, line={"color": "#536170", "width": 1}, row=1, col=1)
    fig.add_annotation(text="← Poor match", x=-72, y=1.08, xref="x", yref="y domain", showarrow=False)
    fig.add_annotation(text="Strong match →", x=72, y=1.08, xref="x", yref="y domain", showarrow=False)
    fig.update_xaxes(
        range=[-100, 124], tickvals=[-100, -50, 0, 50, 100],
        ticktext=["", "", "0", "", ""], title_text="Contextual agreement",
        row=1, col=1,
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)


def _common_bounds(
    station_ids: list[str],
    series_by_station: Mapping[str, pd.Series],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    bounds = []
    for station_id in station_ids:
        series = series_by_station.get(station_id)
        if series is None:
            return None, None
        valid = pd.to_numeric(series, errors="coerce").dropna()
        if valid.empty:
            return None, None
        bounds.append((pd.Timestamp(valid.index.min()), pd.Timestamp(valid.index.max())))
    start = max(item[0] for item in bounds)
    end = min(item[1] for item in bounds)
    if start > end:
        return None, None
    return start, end


def _recent_hydrographs(
    fig: go.Figure,
    result: Layer3StationResult,
    series_by_station: Mapping[str, pd.Series],
    recent_years: int,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    local_ids = result.peer_groups.local.peers.station_id.astype(str).tolist()
    analogue_ids = result.peer_groups.analogues.peers.station_id.astype(str).tolist()
    all_ids = list(dict.fromkeys([result.station_id, *local_ids, *analogue_ids]))
    common_start, common_end = _common_bounds(all_ids, series_by_station)
    if common_start is None:
        fig.add_annotation(
            text="No common observed period is available for the target and all selected peers.",
            x=0.5, y=0.5, xref="x3 domain", yref="y3 domain", showarrow=False,
        )
        return None, None, None, None
    view_start = max(common_start, common_end - pd.DateOffset(years=recent_years))
    full_starts, full_ends = [], []

    groups = [
        ([result.station_id], "Target station", GREEN, 2.8, 0.95),
        (local_ids, "Local peers", CYAN, 1.2, 0.52),
        (analogue_ids, "Comparable catchments", MAUVE, 1.1, 0.40),
    ]
    for ids, legend_name, color, width, opacity in groups:
        for number, station_id in enumerate(ids):
            raw = pd.to_numeric(series_by_station[station_id], errors="coerce").sort_index()
            valid = raw.dropna()
            if valid.empty:
                continue
            full_starts.append(pd.Timestamp(valid.index.min()))
            full_ends.append(pd.Timestamp(valid.index.max()))
            daily = (
                len(raw.index) < 2
                or (raw.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
            )
            x_values = (
                {"x0": raw.index[0].strftime("%Y-%m-%d"), "dx": 86_400_000}
                if daily
                else {"x": raw.index.strftime("%Y-%m-%d")}
            )
            fig.add_trace(
                go.Scatter(
                    y=raw.round(4).values,
                    mode="lines",
                    name=station_id,
                    legendgroup=station_id,
                    showlegend=True,
                    line={"color": color, "width": width},
                    opacity=opacity,
                    connectgaps=False,
                    hovertemplate=(
                        f"<b>{station_id}</b><br>%{{x}}"
                        "<br>Discharge: %{y:.3f}<extra></extra>"
                    ),
                    **x_values,
                ),
                row=2, col=1,
            )
    return view_start, common_end, min(full_starts), max(full_ends)


def _event_shapes(
    fig: go.Figure,
    result: Layer3StationResult,
    evidence: Mapping[str, StationContextEvidence],
) -> None:
    analogue_ids = result.peer_groups.analogues.peers.station_id.astype(str).tolist()
    local_ids = result.peer_groups.local.peers.station_id.astype(str).tolist()
    groups = [
        ([result.station_id], GREEN, 2.8, 0.95),
        (local_ids, CYAN, 1.2, 0.52),
        (analogue_ids, MAUVE, 1.2, 0.42),
    ]
    for ids, color, width, opacity in groups:
        for station_id in ids:
            if station_id not in evidence:
                continue
            curve = evidence[station_id].representative_event_curve
            if curve.empty:
                continue
            curve_values = pd.to_numeric(curve, errors="coerce").to_numpy(float)
            finite = np.isfinite(curve_values)
            if not finite.any():
                continue
            event_days = pd.to_numeric(pd.Index(curve.index), errors="coerce")
            if np.asarray(pd.isna(event_days)).any():
                event_days = np.arange(len(curve_values), dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=event_days,
                    y=curve_values,
                    mode="lines",
                    name=station_id,
                    legendgroup=station_id,
                    showlegend=False,
                    line={"color": color, "width": width},
                    opacity=opacity,
                    hovertemplate=(
                        f"<b>{station_id}</b><br>Days since event start: %{{x:.1f}}"
                        "<br>Discharge: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=2, col=2,
            )


def _combined_peer_table(result: Layer3StationResult) -> go.Table:
    metadata = result.metadata or {}
    frames = [pd.DataFrame([{
        "station_id": result.station_id,
        "distance_km": 0.0,
        "catchment_area_ratio": 1.0,
        "climate_code": metadata.get("climate_code", "Not available"),
        "climate_description": metadata.get("climate_description"),
        "selection_reason": "Reference station",
        "peer_type": "Reference",
    }])]
    for label, peers in (
        ("Local", result.peer_groups.local.peers),
        ("Comparable", result.peer_groups.analogues.peers),
    ):
        if peers.empty:
            continue
        frame = peers.copy()
        frame["peer_type"] = label
        frames.append(frame)
    if not frames:
        return go.Table(
            header={"values": ["<b>Peer selection</b>"], "align": "left"},
            cells={"values": [["No eligible peers"]], "align": "left"},
        )
    peers = pd.concat(frames, ignore_index=True)
    return go.Table(
        columnwidth=[0.7, 1.2, 0.9, 0.8, 2.5],
        header={
            "values": [
                "<b>Type</b>", "<b>Station</b>", "<b>Distance (km)</b>",
                "<b>Area ratio</b>", "<b>Selection</b>",
            ],
            "fill_color": "#EBE2EA",
            "align": "left",
            "height": 27,
        },
        cells={
            "values": [
                peers["peer_type"],
                peers["station_id"],
                peers["distance_km"].map(lambda value: f"{value:.1f}"),
                peers["catchment_area_ratio"].map(lambda value: f"{value:.2f}"),
                peers["selection_reason"].str.capitalize(),
            ],
            "align": "left",
            "height": 25,
            "fill_color": "#F6F8FA",
        },
    )


def _not_assessed_dashboard(result: Layer3StationResult) -> go.Figure:
    """Return a compact status page when no contextual comparison is possible."""
    metadata = result.metadata or {}
    climate = metadata.get("climate_code", "Not available")
    description = metadata.get("climate_description")
    if description:
        climate = f"{climate} — {description}"
    area = metadata.get("catchment_area_km2")
    area_text = (
        "Not available" if area is None or pd.isna(area)
        else f"{float(area):,.1f} km²"
    )
    figure = go.Figure()
    figure.add_annotation(
        text=(
            f"<span style='font-size:24px;color:#2F6F78'><b>"
            f"{result.station_id} — Layer 3 context</b></span><br><br>"
            "<b>Context comparison not assessed</b><br>"
            "No eligible nearby gauges or comparable catchments were available."
        ),
        x=0.03, y=0.82, xref="paper", yref="paper",
        xanchor="left", yanchor="top", align="left", showarrow=False,
        font={"size": 14, "color": "#17365D"},
    )
    figure.add_annotation(
        text=(
            f"<b>River:</b> {metadata.get('river_name', 'Not available')}<br>"
            f"<b>Catchment:</b> {metadata.get('catchment_name', 'Not available')}<br>"
            f"<b>Area:</b> {area_text}<br><b>Climate:</b> {climate}"
        ),
        x=0.58, y=0.82, xref="paper", yref="paper",
        xanchor="left", yanchor="top", align="left", showarrow=False,
        font={"size": 12, "color": "#17365D"},
    )
    figure.update_layout(
        height=320,
        template="plotly_white",
        margin={"l": 35, "r": 35, "t": 30, "b": 30},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def _insufficient_comparability_dashboard(
    result: Layer3StationResult,
    _minimum_similarity: float,
) -> go.Figure:
    """Return evidence scores without misleading behaviour-comparison plots."""
    metadata = result.metadata or {}
    figure = go.Figure()
    figure.add_annotation(
        text=(
            f"<span style='font-size:24px;color:#2F6F78'><b>"
            f"{result.station_id} — Layer 3 context</b></span><br><br>"
            "<b>Context comparison not plotted</b><br>"
            f"Overall contextual agreement: {result.summary.combined_classification}<br>"
            f"Nearby-gauge agreement: {result.summary.local.classification}<br>"
            f"Comparable-catchment agreement: {result.summary.analogues.classification}"
        ),
        x=0.03, y=0.86, xref="paper", yref="paper",
        xanchor="left", yanchor="top", align="left", showarrow=False,
        font={"size": 14, "color": "#17365D"},
    )
    figure.add_annotation(
        text=(
            f"<b>River:</b> {metadata.get('river_name', 'Not available')}<br>"
            f"<b>Catchment:</b> {metadata.get('catchment_name', 'Not available')}<br>"
            "The selected gauges do not provide sufficient comparability for "
            "a detailed station-context plot."
        ),
        x=0.60, y=0.86, xref="paper", yref="paper",
        xanchor="left", yanchor="top", align="left", showarrow=False,
        font={"size": 12, "color": "#17365D"},
    )
    figure.update_layout(
        height=340, template="plotly_white",
        margin={"l": 35, "r": 35, "t": 30, "b": 30},
        xaxis={"visible": False}, yaxis={"visible": False},
    )
    return figure


def _no_context_report_needed_dashboard(result: Layer3StationResult) -> go.Figure:
    """Explain why adequate context did not require a detailed report."""
    figure = go.Figure()
    figure.add_annotation(
        text=(
            f"<span style='font-size:24px;color:#2F6F78'><b>"
            f"{result.station_id} — Layer 3 context</b></span><br><br>"
            "<b>No detailed context report required</b><br>"
            f"Contextual agreement: {result.summary.combined_classification}<br>"
            "Layer 1 did not classify this station as needing review."
        ),
        x=0.04, y=0.84, xref="paper", yref="paper",
        xanchor="left", yanchor="top", align="left", showarrow=False,
        font={"size": 14, "color": "#17365D"},
    )
    figure.update_layout(
        height=300, template="plotly_white",
        margin={"l": 35, "r": 35, "t": 30, "b": 30},
        xaxis={"visible": False}, yaxis={"visible": False},
    )
    return figure


def build_layer3_overview(
    result: Layer3StationResult,
    evidence: Mapping[str, StationContextEvidence],
    series_by_station: Mapping[str, pd.Series],
    *,
    recent_years: int = 3,
    plot_mode: str = "recommended",
) -> go.Figure:
    """Build a compact dashboard readable within a few seconds."""
    mode = _plot_mode(plot_mode)
    if mode == "none":
        raise ValueError("Plot mode 'none' does not build a Layer 3 figure.")
    report_minimum = result.summary.report_minimum_similarity_percent
    if result.summary.combined_similarity_percent is None:
        return _not_assessed_dashboard(result)
    if mode == "recommended" and result.summary.combined_similarity_percent < report_minimum:
        return _insufficient_comparability_dashboard(result, report_minimum)
    if mode == "recommended" and not result.upstream_review_triggered:
        return _no_context_report_needed_dashboard(result)
    figure = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "table", "colspan": 2}, None],
        ],
        column_widths=[0.74, 0.26],
        row_heights=[0.23, 0.47, 0.30],
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
        subplot_titles=[
            "",
            "Observed hydrograph comparison", "Typical high-flow event comparison",
            "Selected context stations",
        ],
    )
    _overall_assessment(figure, result)
    view_start, view_end, full_start, full_end = _recent_hydrographs(
        figure, result, series_by_station, recent_years
    )
    _event_shapes(figure, result, evidence)
    figure.add_trace(_combined_peer_table(result), row=3, col=1)

    metadata = result.metadata or {}
    climate = metadata.get("climate_code", "Not available")
    description = metadata.get("climate_description")
    if description:
        climate = f"{climate} — {description}"
    area = metadata.get("catchment_area_km2")
    area_text = "Not available" if area is None or pd.isna(area) else f"{float(area):,.1f} km²"
    metadata_summary = (
        f"<b>River:</b> {metadata.get('river_name', 'Not available')}<br>"
        f"<b>Catchment:</b> {metadata.get('catchment_name', 'Not available')}<br>"
        f"<b>Area:</b> {area_text}<br>"
        f"<b>Climate:</b> {climate}"
    )
    composite_text = result.summary.combined_classification
    local_text = result.summary.local.classification
    comparable_text = result.summary.analogues.classification
    context_summary = (
        f"<b>Overall contextual agreement: {composite_text}</b><br>"
        f"Nearby-gauge agreement: {local_text}<br>"
        f"Comparable-catchment agreement: {comparable_text}"
    )
    figure.update_layout(
        title=None,
        height=980,
        template="plotly_white",
        barmode="stack",
        margin={"l": 300, "r": 55, "t": 180, "b": 35},
        hoverlabel={"align": "left"},
        legend={
            "orientation": "v",
            "x": -0.075,
            "xanchor": "right",
            "y": 0.505,
            "yanchor": "top",
            "font": {"size": 9},
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "#D8DEE5",
            "borderwidth": 1,
            "groupclick": "togglegroup",
        },
    )
    figure.add_annotation(
        text=(
            f"<span style='font-size:24px;color:#2F6F78'>"
            f"<b>{result.station_id} — Layer 3 context</b></span><br>"
            f"{context_summary}"
        ),
        x=0.0,
        y=1.18,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        font={"size": 13, "color": "#17365D"},
    )
    figure.add_annotation(
        text=metadata_summary,
        x=0.55,
        y=1.18,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        font={"size": 12},
    )
    figure.update_xaxes(
        title_text="Date",
        range=None if view_start is None else [view_start, view_end],
        row=2, col=1,
    )
    figure.update_yaxes(title_text="Discharge", row=2, col=1)
    figure.update_xaxes(
        title_text="Days since event start",
        autorange=True,
        row=2, col=2,
    )
    figure.update_yaxes(
        title_text="Discharge", autorange=True, row=2, col=2
    )
    if view_start is not None:
        five_year_start = max(
            pd.Timestamp(full_start),
            pd.Timestamp(view_end) - pd.DateOffset(years=5),
        )
        figure.update_layout(
            updatemenus=[{
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 0.595,
                "xanchor": "left",
                "yanchor": "bottom",
                "showactive": True,
                "buttons": [
                    {"label": "3 years", "method": "relayout",
                     "args": [{"xaxis2.range": [view_start, view_end]}]},
                    {"label": "5 years", "method": "relayout",
                     "args": [{"xaxis2.range": [five_year_start, view_end]}]},
                    {"label": "All", "method": "relayout",
                     "args": [{"xaxis2.range": [full_start, full_end]}]},
                ],
            }]
        )
    return figure


def write_layer3_overview(
    result: Layer3StationResult,
    evidence: Mapping[str, StationContextEvidence],
    series_by_station: Mapping[str, pd.Series],
    output_path: str | Path,
    *,
    recent_years: int = 3,
    plot_mode: str = "recommended",
) -> Path | None:
    """Write a compact dashboard that reuses one local Plotly asset."""
    mode = _plot_mode(plot_mode)
    if mode == "none":
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assets = path.parent.parent / "_assets"
    assets.mkdir(parents=True, exist_ok=True)
    plotly_javascript = assets / "plotly.min.js"
    if not plotly_javascript.exists():
        plotly_javascript.write_text(get_plotlyjs(), encoding="utf-8")
    plotly_reference = Path(
        os.path.relpath(plotly_javascript, path.parent)
    ).as_posix()
    build_layer3_overview(
        result, evidence, series_by_station, recent_years=recent_years,
        plot_mode=mode,
    ).write_html(path, include_plotlyjs=plotly_reference, full_html=True)
    return path


__all__ = [
    "build_layer3_overview",
    "write_layer3_overview",
    "LAYER3_PLOT_MODES",
]
