"""Layer 2 hydrological-signature and event figures."""

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

# ==========================================================
# LAYER 2: OVERVIEW AND PEAK/OUTLIER CROSS-CHECK
# ==========================================================

COLORS = {
    "minimum": "#2CB7C9", "median": "#E6B655", "mean": "#E58B45",
    "maximum": "#A84A5C", "q95": "#C65353", "q05": "#27AFA6",
    "flashiness": "#C08A3E", "baseflow": "#4F8A70",
    "seasonality": "#7A5C91", "event": "#3976A8",
    "wettest": "#35AFC4", "driest": "#D47755",
    "spike_dip": "#E64980", "peak": "#4C8A78",
}


def _add_flow(fig: go.Figure, annual: pd.DataFrame, references: dict) -> None:
    years = annual["year"]
    for column, name, color, width, opacity in [
        ("maximum_flow", "Annual maximum", COLORS["maximum"], 0.82, 0.66),
        ("mean_flow", "Annual mean", COLORS["mean"], 0.62, 0.88),
        ("median_flow", "Annual median", COLORS["median"], 0.44, 0.88),
        ("minimum_flow", "Annual minimum", COLORS["minimum"], 0.82, 0.90),
    ]:
        screened_extreme = column in {"minimum_flow", "maximum_flow"}
        hover = f"Year=%{{x}}<br>{name}=%{{y:.3f}}"
        customdata = None
        if screened_extreme:
            customdata = annual["extrema_excluded_candidate_count"]
            hover += "<br>Excluded Layer 1 candidate(s)=%{customdata}"
        fig.add_trace(go.Bar(
            x=years, y=annual[column], width=width, name=name,
            marker=dict(color=color, line=dict(width=0)), opacity=opacity,
            customdata=customdata,
            hovertemplate=hover + "<extra></extra>",
        ), row=1, col=1)
    line_x = [float(years.min()) - 0.5, float(years.max()) + 0.5]
    for key, name, color in [
        ("q95_percentile_high_flow_fdc_q05", "High flow: percentile Q95 / FDC Q5", COLORS["q95"]),
        ("q05_percentile_low_flow_fdc_q95", "Low flow: percentile Q05 / FDC Q95", COLORS["q05"]),
    ]:
        fig.add_trace(go.Scatter(
            x=line_x, y=[references[key]] * 2, mode="lines", name=name,
            line=dict(color=color, width=2, dash="dot"),
            hovertemplate=f"{name}=%{{y:.3f}}<extra></extra>",
        ), row=1, col=1)


def _add_indices(fig: go.Figure, annual: pd.DataFrame) -> None:
    for column, name, color in [
        ("flashiness_index", "Annual flashiness", COLORS["flashiness"]),
        ("baseflow_index", "Annual baseflow index", COLORS["baseflow"]),
        ("seasonality_index", "Annual seasonality index", COLORS["seasonality"]),
    ]:
        available = annual.dropna(subset=[column])
        fig.add_trace(go.Scatter(
            x=available["year"], y=available[column], mode="lines+markers", name=name,
            line=dict(color=color, width=1.8), marker=dict(size=5),
            hovertemplate=f"Year=%{{x}}<br>Value=%{{y:.3f}}<extra>{name}</extra>",
        ), row=2, col=1)


def _add_event(
    fig: go.Figure,
    raw: pd.Series,
    representative: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    if representative.empty:
        return
    event = representative.iloc[0]
    values = raw.loc[event["event_start"]:event["event_end"]].dropna()
    days = (values.index - values.index[0]).days
    peak_day = int((event["peak_date"] - event["event_start"]).days)
    fig.add_trace(go.Scatter(
        x=days, y=values.values, mode="lines", name="Representative high-flow event",
        line=dict(color=COLORS["event"], width=2.6),
        customdata=values.index.strftime("%Y-%m-%d"),
        hovertemplate="Event day=%{x}<br>Date=%{customdata}<br>Discharge=%{y:.3f}<extra></extra>",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[peak_day], y=[event["peak_flow"]], mode="markers",
        name="Representative-event peak",
        marker=dict(color=COLORS["maximum"], size=10, symbol="diamond"),
        hovertemplate="Peak day=%{x}<br>Peak flow=%{y:.3f}<extra></extra>",
    ), row=2, col=2)
    fig.add_vline(x=peak_day, line_dash="dot", line_color=COLORS["maximum"], annotation_text="Event peak", row=2, col=2)
    event_summary = summary[summary["diagnostic"].str.startswith("Median event")]
    text = (
        f"Representative event: {event['event_start']:%Y-%m-%d} to {event['event_end']:%Y-%m-%d}"
        f"<br>Time to peak: {event['time_to_peak_days']:.3f} days"
        f"<br>Rising slope: {event['rising_slope']:.3f}"
        f"<br>Recession slope: {event['recession_slope']:.3f}"
        f"<br>Duration: {event['event_duration_days']:.3f} days"
        f"<br>Peak flow: {event['peak_flow']:.3f}"
        "<br><br><b>Station event medians</b><br>"
        + "<br>".join(
            f"{row.diagnostic.replace('Median event ', '').title()}: {row.median:.3f}"
            for row in event_summary.itertuples() if pd.notna(row.median)
        )
    )
    fig.add_annotation(
        x=1.02, y=0.54, xref="paper", yref="paper", text=text,
        showarrow=False, align="left", xanchor="left", yanchor="top",
        font=dict(size=10, color="#344054"),
        bgcolor="rgba(255,255,255,0.94)", bordercolor="#B7C4CF",
        borderwidth=1, borderpad=5,
    )


def _add_seasonality(
    fig: go.Figure,
    monthly: pd.DataFrame,
    profile: pd.DataFrame,
) -> None:
    first = True
    for _, values in monthly.groupby("year"):
        fig.add_trace(go.Scatter(
            x=values["month"], y=values["monthly_median"], mode="lines",
            line=dict(color="rgba(122,92,145,0.40)", width=1),
            name="Individual annual cycles", legendgroup="annual_cycles",
            showlegend=first, hoverinfo="skip",
        ), row=3, col=1)
        first = False
    fig.add_trace(go.Scatter(
        x=profile["month"], y=profile["median"], mode="lines+markers",
        name="Historical monthly median",
        line=dict(color=COLORS["seasonality"], width=2.7),
        marker=dict(size=5, color=COLORS["seasonality"]),
        customdata=profile["year_count"],
        hovertemplate="Month=%{x}<br>Median=%{y:.3f}<br>Years=%{customdata}<extra></extra>",
    ), row=3, col=1)
    valid = profile.dropna(subset=["median"])
    if valid.empty:
        return
    wet, dry = valid.loc[valid["median"].idxmax()], valid.loc[valid["median"].idxmin()]
    for row, name, color in [
        (wet, "Typical wettest month", COLORS["wettest"]),
        (dry, "Typical driest month", COLORS["driest"]),
    ]:
        fig.add_trace(go.Scatter(
            x=[row["month"]], y=[row["median"]], mode="markers", name=name,
            marker=dict(color=color, size=11, symbol="circle", line=dict(color="white", width=1)),
        ), row=3, col=1)


def build_layer2_overview(
    obs_series: pd.Series,
    diagnostics: dict,
    station_id: str,
) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}], [{"colspan": 2}, None]],
        column_widths=[0.5, 0.5], row_heights=[0.40, 0.27, 0.33],
        horizontal_spacing=0.20, vertical_spacing=0.10,
        subplot_titles=(
            "Annual flow behavior", "Annual catchment-response indices",
            "Representative observed high-flow event", "Historical seasonality",
        ),
    )
    _add_flow(fig, diagnostics["annual_signatures"], diagnostics["references"])
    _add_indices(fig, diagnostics["annual_signatures"])
    _add_event(fig, obs_series, diagnostics["representative_event"], diagnostics["diagnostic_summary"])
    _add_seasonality(fig, diagnostics["annual_monthly_profiles"], diagnostics["seasonality_profile"])
    fig.update_layout(
        title=f"{station_id} - Layer 2 overview", template="plotly_white",
        height=1180, barmode="overlay", hovermode="closest", margin=dict(r=300),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02, tracegroupgap=8),
    )
    fig.update_yaxes(title_text="Discharge", row=1, col=1)
    fig.update_yaxes(title_text="Index", row=2, col=1)
    fig.update_yaxes(title_text="Discharge", row=2, col=2)
    fig.update_yaxes(title_text="Monthly median discharge", row=3, col=1)
    fig.update_xaxes(title_text="Calendar year", row=2, col=1)
    fig.update_xaxes(title_text="Days since event start", row=2, col=2)
    fig.update_xaxes(
        title_text="Month", tickmode="array", tickvals=list(range(1, 13)),
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        row=3, col=1,
    )
    return fig


def build_peak_review(
    obs_series: pd.Series,
    events: pd.DataFrame,
    crosscheck: pd.DataFrame,
    station_id: str,
) -> go.Figure:
    valid = obs_series.dropna()
    fig = go.Figure(go.Scatter(
        x=valid.index, y=valid.values, mode="lines", name="Discharge",
        line=dict(color="#577590", width=0.8),
    ))
    if not events.empty:
        overlap = events.get(
            "layer1_spike_peak_overlap",
            pd.Series(False, index=events.index),
        ).fillna(False)
        for subset, name, color, symbol in [
            (events[~overlap], "Layer 2 high-flow peaks", COLORS["peak"], "circle-open"),
            (events[overlap], "Spike–peak overlap: check", COLORS["spike_dip"], "x-open"),
        ]:
            if subset.empty:
                continue
            hover_data = subset[["event_id", "event_start", "event_end"]].copy()
            hover_data["selection_note"] = subset[
                "representative_exclusion_reason"
            ].fillna("Eligible for representative-event selection.")
            fig.add_trace(go.Scatter(
                x=subset["peak_date"], y=subset["peak_flow"], mode="markers",
                name=name,
                marker=dict(
                    color=color, size=9, symbol=symbol, opacity=0.78,
                    line=dict(width=2),
                ),
                customdata=hover_data,
                hovertemplate=(
                    "Peak=%{x}<br>Flow=%{y:.3f}<br>Event=%{customdata[0]}"
                    "<br>Window=%{customdata[1]} to %{customdata[2]}"
                    "<br>%{customdata[3]}<extra></extra>"
                ),
            ))
    for kind, symbol in [("spike", "triangle-up"), ("dip", "triangle-down")]:
        rows = crosscheck[crosscheck["type"] == kind] if not crosscheck.empty else pd.DataFrame()
        if rows.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rows["timestamp"], y=rows["candidate_value"], mode="markers",
            name=f"Spike / dip ({kind})",
            marker=dict(color=COLORS["spike_dip"], size=10, symbol=symbol, opacity=0.68),
            text=rows["assessment"],
            hovertemplate=f"{kind.title()} candidate<br>%{{x}}<br>Discharge=%{{y:.3f}}<br>%{{text}}<extra></extra>",
        ))
    fig.update_layout(
        title=f"{station_id} - Layer 1 spike/dip and Layer 2 peak cross-check",
        xaxis_title="Date", yaxis_title="Discharge", template="plotly_white",
        height=650, hovermode="closest",
    )
    return fig


def layer2_typical_rows(diagnostics: dict) -> list[tuple[str, object]]:
    annual = diagnostics["annual_signatures"]
    rows: list[tuple[str, object]] = []
    for column, label in [
        ("mean_flow", "Median annual mean flow"),
        ("median_flow", "Median annual median flow"),
        ("minimum_flow", "Median annual screened minimum flow"),
        ("maximum_flow", "Median annual screened maximum flow"),
        ("flashiness_index", "Median annual flashiness"),
        ("baseflow_index", "Median annual baseflow index"),
        ("seasonality_index", "Median annual seasonality index"),
        ("lag1_autocorrelation", "Median annual lag-1 autocorrelation"),
    ]:
        values = pd.to_numeric(annual.get(column), errors="coerce").dropna()
        rows.append((label, values.median() if not values.empty else np.nan))
    profile = diagnostics["seasonality_profile"].dropna(subset=["median"])
    rows.extend([
        (
            "Typical wettest month",
            profile.loc[profile["median"].idxmax(), "month"]
            if not profile.empty else np.nan,
        ),
        (
            "Typical driest month",
            profile.loc[profile["median"].idxmin(), "month"]
            if not profile.empty else np.nan,
        ),
    ])
    summary = diagnostics["diagnostic_summary"].set_index("diagnostic")
    for diagnostic in [
        "Median event time to peak (days)", "Median event rising slope",
        "Median event recession slope", "Median event duration (days)",
        "Median event peak flow",
    ]:
        value = summary.loc[diagnostic, "median"] if diagnostic in summary.index else np.nan
        rows.append((diagnostic, value))
    references = diagnostics["references"]
    rows.extend([
        ("Q05 low-flow reference", references.get("q05_percentile_low_flow_fdc_q95", np.nan)),
        ("Q95 high-flow reference", references.get("q95_percentile_high_flow_fdc_q05", np.nan)),
        (
            "Spike–peak overlaps requiring manual check",
            diagnostics.get("spike_peak_overlap_count", 0),
        ),
    ])
    return rows


def build_layer2_diagnostic_table(
    diagnostics: dict, station_id: str, source_id: str
) -> go.Figure:
    rows = layer2_typical_rows(diagnostics)
    values = [
        str(int(value)) if (
            "month" in metric.lower() or "overlap" in metric.lower()
        ) and pd.notna(value)
        else f"{float(value):.3f}" if pd.notna(value) else "Not available"
        for metric, value in rows
    ]
    figure = go.Figure(go.Table(
        columnwidth=[0.70, 0.30],
        header=dict(values=["Diagnostic", source_id], fill_color="#E9ECEF", align="left"),
        cells=dict(values=[[metric for metric, _ in rows], values], align="left"),
    ))
    figure.update_layout(
        title=f"{station_id} - {source_id} - Layer 2 diagnostic medians",
        template="plotly_white", height=620, margin=dict(t=70, b=20),
    )
    return figure


def generate_layer2_visuals(
    obs_series: pd.Series,
    diagnostics: dict,
    *,
    station_id: str,
    output_root: str | Path,
    show: bool = False,
    source_id: str | None = None,
    write_tables: bool = True,
    flat_filename: str | None = None,
) -> Path:
    station_dir = Path(output_root) if flat_filename else Path(output_root) / station_id
    if source_id and not flat_filename:
        station_dir = station_dir / source_id
    if not flat_filename:
        station_dir = station_dir / "layer2"
    station_dir.mkdir(parents=True, exist_ok=True)
    overview = build_layer2_overview(obs_series, diagnostics, station_id)
    review = build_peak_review(
        obs_series, diagnostics["hydrograph_events"],
        diagnostics["spike_peak_crosscheck"], station_id,
    )
    figures = [overview]
    if not diagnostics["hydrograph_events"].empty or not diagnostics["spike_peak_crosscheck"].empty:
        figures.append(review)
    figures.append(build_layer2_diagnostic_table(
        diagnostics, station_id, source_id or "source"
    ))
    if show:
        for figure in figures:
            figure.show()
    parts = [
        pio.to_html(
            figure, full_html=False,
            include_plotlyjs="cdn" if index == 0 else False,
        )
        for index, figure in enumerate(figures)
    ]
    (station_dir / (flat_filename or "layer2_overview.html")).write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{station_id} - {source_id or 'source'} - Layer 2</title>"
        "</head><body>" + "\n".join(parts) + "</body></html>",
        encoding="utf-8",
    )
    tables = {
        "layer2_annual_signatures.csv": diagnostics["annual_signatures"],
        "layer2_annual_monthly_profiles.csv": diagnostics["annual_monthly_profiles"],
        "layer2_seasonality_profile.csv": diagnostics["seasonality_profile"],
        "layer2_hydrograph_events.csv": diagnostics["hydrograph_events"],
        "layer2_representative_event.csv": diagnostics["representative_event"],
        "layer2_spike_peak_crosscheck.csv": diagnostics["spike_peak_crosscheck"],
        "layer2_diagnostic_summary.csv": diagnostics["diagnostic_summary"],
    }
    if write_tables:
        for filename, frame in tables.items():
            frame.to_csv(station_dir / filename, index=False, float_format="%.6f")
    return station_dir


# ==========================================================
