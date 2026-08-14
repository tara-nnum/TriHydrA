"""Layer 1 interactive diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trihydra.plotting._shared import (
    CHECK_LABELS_L1,
    DEFAULT_OUTPUT_ROOT,
    FLAG_MARKER_OPACITY,
    FLAG_MARKER_SIZE,
    LAYER1_CHECK_COLORS,
    PLOT_TEMPLATE,
    _save_dashboard,
)
from trihydra.layer1.timeseries_validity import get_valid_record

EPOCH_DRIFT_COLOURS = {
    "rising": "#168A63",
    "stable": "#F28E2B",
    "falling": "#B32646",
}

# ==========================================================
# LAYER 1: CONSOLIDATED CHECK PLOT
# ==========================================================

def plot_layer1_combined(
    series: pd.Series,
    check_results: list[dict],
    series_type: str,
    show_summary_table: bool = True,
    station_id: str | None = None,
) -> go.Figure:
    """Overlay all Layer 1 evidence on one toggleable hydrograph."""
    by_check = {r["check"]: r for r in check_results}
    # Plot a sorted copy of the valid record while preserving duplicate rows.
    series = get_valid_record(series)
    marker_series = series.loc[~series.index.duplicated(keep="first")]

    if show_summary_table:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.74, 0.26],
            specs=[[{"type": "xy"}], [{"type": "table"}]],
            vertical_spacing=0.06,
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    finite = series.dropna()
    ymin = float(finite.min()) if not finite.empty else 0.0
    ymax = float(finite.max()) if not finite.empty else 1.0
    missing_result = by_check.get("missing_values", {})
    missing_intervals = missing_result.get("internal_intervals", []) or []
    for index, interval in enumerate(missing_intervals):
        start = pd.Timestamp(interval["start"])
        end = pd.Timestamp(interval["end"]) + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[start, start, end, end],
            y=[ymin, ymax, ymax, ymin],
            mode="lines",
            line={"width": 0},
            fill="toself",
            fillcolor="rgba(224,49,49,0.10)",
            name="Missing interval",
            legendgroup="missing_intervals",
            showlegend=index == 0,
            hovertemplate=(
                f"Missing interval<br>{start:%Y-%m-%d} to "
                f"{pd.Timestamp(interval['end']):%Y-%m-%d}"
                f"<br>{int(interval['missing_count'])} observation(s)"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color="#ADB5BD", width=1), name="Discharge",
        hovertemplate="%{x|%Y-%m-%d}<br>Discharge=%{y:.4g}<extra></extra>",
    ), row=1, col=1)

    # Descriptor/period checks are drawn only when they have visible evidence.
    zero_result = by_check.get("zero_flow_regime")
    zero_mask = series.round(3).eq(0)
    zero_dates = series.index[zero_mask]
    if zero_result is not None and len(zero_dates):
        fig.add_trace(go.Scatter(
            x=zero_dates, y=series.to_numpy()[zero_mask.to_numpy()],
            mode="markers",
            marker=dict(size=2, opacity=0.40,
                        color=LAYER1_CHECK_COLORS["zero_flow_regime"],
                        line=dict(width=0)),
            name="Zero flow",
        ), row=1, col=1)
    low_result = by_check.get("low_variability", {})
    plateau_periods = low_result.get("plateau_periods", []) or []
    plateau_threshold = int(low_result.get("minimum_plateau_days", 15))
    for i, period in enumerate(plateau_periods):
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        fig.add_trace(go.Scatter(
            x=[start, end], y=[period["plateau_value"]] * 2,
            mode="lines",
            line=dict(color=LAYER1_CHECK_COLORS["low_variability"], width=5),
            name="Non-zero plateau",
            legendgroup="low_variability",
            showlegend=i == 0,
            hovertemplate=(
                f"Non-zero plateau<br>{start:%Y-%m-%d} to {end:%Y-%m-%d}"
                f"<br>Duration: {period['calendar_duration_days']} days"
                f"<br>Value: {period['plateau_value']:.3f}"
                f"<br>Minimum duration: {plateau_threshold} days"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # --- point-style checks: small, semi-transparent markers ---
    for check in ["negative_discharge", "duplicate_timestamps", "timestep_consistency"]:
        result = by_check.get(check)
        if result is None or not result.get("flagged_timestamps"):
            continue
        idx = pd.to_datetime(result["flagged_timestamps"])
        fig.add_trace(go.Scatter(
            x=idx, y=marker_series.reindex(idx).values, mode="markers",
            marker=dict(size=FLAG_MARKER_SIZE, color=LAYER1_CHECK_COLORS[check],
                        opacity=FLAG_MARKER_OPACITY, line=dict(width=0)),
            name=CHECK_LABELS_L1[check],
        ), row=1, col=1)

    # --- spike/dip: same check colour, shape distinguishes spike vs dip ---
    spike_dip_result = by_check.get("spike_dip")
    if spike_dip_result is not None:
        details = {d["timestamp"]: d for d in spike_dip_result.get("candidate_details", [])}
        for kind, symbol in [("spike", "triangle-up"), ("dip", "triangle-down")]:
            ts = [t for t, d in details.items() if d["type"] == kind]
            if not ts:
                continue
            idx = pd.to_datetime(ts)
            hover = [
                (
                    f"{kind.title()} candidate<br>{d['timestamp']}"
                    f"<br>Discharge: {d.get('candidate_value', d.get('flagged_value')):.3f}"
                    f"<br>Recovery score: {d.get('recovery_score', float('nan')):.3f}"
                    f"<br>Candidate score: {d.get('raw_score', float('nan')):.3f}"
                )
                for t, d in details.items() if d["type"] == kind
            ]
            fig.add_trace(go.Scatter(
                x=idx, y=marker_series.reindex(idx).values, mode="markers",
                marker=dict(size=11, color="#D6336C", opacity=0.95,
                            symbol=symbol, line=dict(width=1, color="#7B2C46")),
                name=f"Spike / dip ({kind})",
                text=hover, hovertemplate="%{text}<extra></extra>",
            ), row=1, col=1)

    step_result = by_check.get("step_shift", {})
    for index, regime in enumerate(step_result.get("regime_summary", []) or []):
        start = pd.Timestamp(regime["start"])
        end = pd.Timestamp(regime["end"])
        fig.add_trace(go.Scatter(
            x=[start, end],
            y=[regime["median"], regime["median"]],
            mode="lines",
            line={"color": "#212529", "width": 1.5},
            name="Regime median",
            legendgroup="regime_median",
            showlegend=index == 0,
            hovertemplate=(
                f"Regime {regime.get('regime', index + 1)}"
                f"<br>{start:%Y-%m-%d} to {end:%Y-%m-%d}"
                f"<br>Median: {regime['median']:.3f}<extra></extra>"
            ),
        ), row=1, col=1)

    shown_tiers: set[str] = set()
    for boundary in step_result.get("regime_boundaries", []) or []:
        tier = str(boundary.get("diagnosis", "Tier 3"))
        if tier not in {"Tier 1", "Tier 2"}:
            continue
        date = pd.Timestamp(boundary["boundary_timestamp"])
        color = "#A71930" if tier == "Tier 1" else "#E07A3F"
        fig.add_trace(go.Scatter(
            x=[date, date], y=[ymin, ymax], mode="lines",
            line={"color": color, "width": 1.5, "dash": "dash"},
            name=f"Step shift ({tier})",
            legendgroup=f"step_shift_{tier}",
            showlegend=tier not in shown_tiers,
            hovertemplate=(
                f"{tier} step shift<br>{date:%Y-%m-%d}"
                f"<br>Before median: {boundary['before_median']:.3f}"
                f"<br>After median: {boundary['after_median']:.3f}"
                f"<br>Absolute change: "
                f"{boundary['absolute_median_change']:.3f}<extra></extra>"
            ),
        ), row=1, col=1)
        shown_tiers.add(tier)

    epoch_result = by_check.get("epoch_drift", {})
    overview = pd.DataFrame(
        epoch_result.get("consolidated_overview_slopes", []) or []
    )
    if not overview.empty:
        overview["date"] = pd.to_datetime(
            overview["year"].astype(str) + "-07-01"
        )
        shown_states: set[str] = set()
        for _, continuous in overview.groupby("continuous_run", sort=False):
            previous_tail = None
            for _, period in continuous.groupby("overview", sort=False):
                state = str(period["state"].iloc[0])
                draw = period.copy()
                if previous_tail is not None:
                    draw = pd.concat([previous_tail.to_frame().T, draw])
                fig.add_trace(go.Scatter(
                    x=draw["date"], y=draw["fitted_level"], mode="lines",
                    line={
                        "color": EPOCH_DRIFT_COLOURS[state],
                        "width": 2,
                        "dash": "dot",
                    },
                    name=f"Epoch {state}",
                    legendgroup=f"epoch_{state}",
                    showlegend=state not in shown_states,
                    customdata=np.column_stack([
                        draw["sen_slope_percent_per_year"]
                    ]),
                    hovertemplate=(
                        "%{x|%Y}<br>Epoch level: %{y:.3f}"
                        "<br>Sen slope: %{customdata[0]:+.2f}%/year"
                        "<extra></extra>"
                    ),
                ), row=1, col=1)
                shown_states.add(state)
                previous_tail = period.iloc[-1]

    if show_summary_table:
        compact_rows = []
        simple_order = [
            "missing_values", "long_gaps", "negative_discharge",
            "duplicate_timestamps", "timestep_consistency",
            "zero_flow_regime", "low_variability", "spike_dip",
            "step_shift", "epoch_drift",
        ]
        for check in simple_order:
            result = by_check.get(check)
            if result is None:
                continue
            if result.get("execution_status") == "skipped":
                continue
            has_visible_evidence = bool(result.get("flag", False)) or (
                check == "zero_flow_regime"
                and int(result.get("zero_count", 0)) > 0
            )
            if not has_visible_evidence:
                continue
            value = result.get("value")
            if check == "missing_values":
                count = int(result.get("internal_nan_count", value or 0))
                ratio = float(result.get("internal_missing_percentage", 0.0))
                summary = f"{count:,} internal missing observations ({ratio:.2f}%)"
            elif check == "long_gaps":
                gaps = result.get("long_gap_intervals", []) or []
                summary = f"{len(gaps)} long gap(s); longest {int(value or 0):,} days"
            elif check == "negative_discharge":
                summary = f"{int(value or 0):,} negative observation(s)"
            elif check == "duplicate_timestamps":
                summary = (
                    f"{int(value or 0):,} "
                    f"unique duplicated date(s); "
                    f"{int(result.get('extra_duplicate_rows', 0)):,} extra row(s)"
                )
            elif check == "timestep_consistency":
                summary = (
                    f"{int(result.get('irregular_spacing_count', 0)):,} irregular "
                    f"interval(s); out of order: "
                    f"{'yes' if result.get('out_of_order') else 'no'}"
                )
            elif check == "zero_flow_regime":
                summary = (
                    f"{int(result.get('zero_count', 0)):,} zero-flow observations; "
                    f"{int(result.get('zero_spell_count', 0))} spell(s); longest "
                    f"{int(result.get('longest_zero_spell_days', 0)):,} days"
                )
            elif check == "low_variability":
                periods = result.get("plateau_periods", []) or []
                longest = max(
                    (int(p.get("calendar_duration_days", 0)) for p in periods),
                    default=0,
                )
                summary = (
                    f"{len(periods)} non-zero plateau candidate(s); "
                    f"longest {longest:,} days"
                )
            elif check == "spike_dip":
                details = result.get("candidate_details", []) or []
                spikes = sum(d.get("type") == "spike" for d in details)
                dips = sum(d.get("type") == "dip" for d in details)
                summary = f"{len(details)} candidate(s): {spikes} spike(s), {dips} dip(s)"
            elif check == "step_shift":
                summary = (
                    f"{result.get('composite_tier', 'Tier 3')}; score "
                    f"{float(result.get('step_shift_score', 0.0)):.3f}; "
                    f"{int(result.get('tier_1_count', 0))} Tier 1 and "
                    f"{int(result.get('tier_2_count', 0))} Tier 2 boundary(ies)"
                )
            else:
                summary = (
                    f"{result.get('diagnosis', 'Not assessed')} "
                    f"({result.get('tier', 'Not assessed')}); "
                    f"stable for "
                    f"{100 * float(result.get('stable_year_fraction', 0.0)):.1f}% "
                    "of assessed years"
                )
            compact_rows.append({
                "check": CHECK_LABELS_L1.get(check, check),
                "summary": summary,
            })

        if not compact_rows:
            compact_rows = [{"check": "Clean", "summary": "No Layer 1 checks were flagged."}]

        fig.add_trace(go.Table(
            columnwidth=[0.28, 0.72],
            header=dict(
                values=["Check", "Basic summary"],
                fill_color="#E9ECEF", align="left",
            ),
            cells=dict(
                values=[
                    [row["check"] for row in compact_rows],
                    [row["summary"] for row in compact_rows],
                ],
                align="left",
            ),
        ), row=2, col=1)

    fig.update_layout(
        title=(
            f"Layer 1 combined checks ({series_type})"
            f" - {station_id or 'station'}"
        ),
        template=PLOT_TEMPLATE, height=780 if show_summary_table else 590,
        margin=dict(t=70, b=30),
        legend={"groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Discharge (source units)", row=1, col=1)
    return fig


# ==========================================================
# LAYER 1: MAIN ENTRY POINT
# ==========================================================

def generate_layer1_visuals(
    obs_series: pd.Series,
    sim_series: pd.Series | None,
    layer1_diagnostics: dict,
    station_id: str = "station",
    model_name: str = "model",
    output_root: Path | None = None,
    show: bool = False,
    source_id: str | None = None,
    write_tables: bool = True,
    flat_filename: str | None = None,
) -> Path:
    """Write one Layer 1 dashboard from already calculated diagnostics."""
    output_root = (
        Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    )
    station_dir = output_root if flat_filename else output_root / station_id
    if source_id and not flat_filename:
        station_dir = station_dir / source_id
    if not flat_filename:
        station_dir = station_dir / "layer1"
    station_dir.mkdir(parents=True, exist_ok=True)
    raw_results = layer1_diagnostics["raw_results"]

    inputs = [("obs", obs_series)]
    if sim_series is not None:
        inputs.append((model_name, sim_series))
    figures = [
        plot_layer1_combined(
            series,
            raw_results[series_type],
            source_id if source_id and len(inputs) == 1 else series_type,
            station_id=station_id,
        )
        for series_type, series in inputs
    ]
    _save_dashboard(
        figures,
        station_dir / (flat_filename or "layer1_overview.html"),
        title=f"{station_id} - {source_id or 'Layer 1'}",
        show=show,
    )

    if write_tables:
        table_files = {
            "layer1_eda_summary.csv": layer1_diagnostics["eda_summary"],
            "layer1_summary_all.csv": layer1_diagnostics["summary_all"],
            "layer1_summary_flagged.csv": layer1_diagnostics["summary_flagged"],
        }
        for filename, table in table_files.items():
            table.to_csv(station_dir / filename, index=False)
        for table_name, table in layer1_diagnostics.get("detail_tables", {}).items():
            if not table.empty:
                table.round(3).to_csv(
                    station_dir / f"layer1_{table_name}.csv",
                    index=False,
                )

    return station_dir
