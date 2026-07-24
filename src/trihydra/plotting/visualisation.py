"""
visualisation.py

All plotting logic for Layer 1 and Layer 2, in one file.

Every function below consumes the OUTPUT of diagnostics.py (already-
computed check/signature results) rather than calling a check itself.
This is deliberate: if this file re-ran checks independently, a rename
or signature change in the check layer could make this file crash
without diagnostics.py noticing, or the two files could silently
disagree about what a check found. Feeding this file diagnostics.py's
output means there is exactly one place checks are ever executed.

Typical use, in one notebook cell:

    from trihydra.plotting.diagnostics import run_layer1_diagnostics, run_layer2_diagnostics
    from trihydra.plotting.visualisation import generate_layer1_visuals, generate_layer2_visuals

    l1 = run_layer1_diagnostics(obs, sim, model_name="AIFL")
    generate_layer1_visuals(obs, sim, l1, station_id="GRDC_4123300", model_name="AIFL")

    l2 = run_layer2_diagnostics(obs, sim, model_name="AIFL")
    generate_layer2_visuals(obs, sim, l2, station_id="GRDC_4123300", model_name="AIFL")

Output layout (per the actual TriHydrA repo structure): everything
saves under <project_root>/src/trihydra/io/output/<station_id>/, with
layer1/ and layer2/ subfolders inside that -- station name first, then
layer, not the other way around.

Expects this file to live at <project_root>/src/trihydra/plotting/,
alongside layer1/, layer2/, layer3/, and io/ as siblings under
trihydra/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------
# Path setup.
#
# This file lives at: <project_root>/src/trihydra/plotting/visualisation.py
# TRIHYDRA_DIR       = <project_root>/src/trihydra          (2 parents up)
# PROJECT_ROOT       = <project_root>                       (4 parents up;
#                       the folder that CONTAINS "src", needed on sys.path
#                       for "from src.trihydra..." imports to resolve)
# LAYER2_DIR         = <project_root>/src/trihydra/layer2   (layer2's own
#                       files use flat imports between each other, so that
#                       folder itself needs to be importable too)
# IO_OUTPUT_ROOT     = <project_root>/src/trihydra/io/output (a sibling of
#                       plotting/, this is where every station's results
#                       are saved: IO_OUTPUT_ROOT/<station_id>/layer1/...
#                       and IO_OUTPUT_ROOT/<station_id>/layer2/...)
# ----------------------------------------------------------------
TRIHYDRA_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LAYER2_DIR = TRIHYDRA_DIR / "layer2"
IO_OUTPUT_ROOT = TRIHYDRA_DIR / "io" / "output"

for p in (PROJECT_ROOT, LAYER2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import pymannkendall as mk
except ImportError:
    mk = None

# Only used for the one "temporary imputation, pending redesign" plot
# below -- not used anywhere else in this file.
from layer2_obs_ml_comparison import fill_obs_for_layer2


# ==========================================================
# SHARED SETTINGS
# ==========================================================

PLOT_TEMPLATE = "plotly_white"
OBS_COLOR = "#1971C2"
MODEL_COLOR = "#E8590C"
OBS_SHADE = "rgba(25, 113, 194, 0.15)"
MODEL_SHADE = "rgba(232, 89, 12, 0.15)"

# One unique colour per Layer 1 check (mentor feedback: basic and
# behavioural checks need to be visually distinguishable from each
# other when several are overlaid on the same plot). Basic checks use
# cooler/greyer tones, behavioural checks use warmer/more saturated
# ones, but every one of the 8 is a genuinely distinct hue on its own.
LAYER1_CHECK_COLORS = {
    "missing_values": "#495057",        # basic -- dark grey, fill
    "long_gaps": "#C92A2A",             # basic -- dark red, fill
    "negative_discharge": "#1971C2",    # basic -- blue, points
    "duplicate_timestamps": "#F08C00",  # basic -- amber, points
    "timestep_consistency": "#5F3DC4",  # basic -- purple, points
    "zero_flow_regime": "#0C8599",      # behavioural -- teal, fill
    "low_variability": "#2F9E44",       # behavioural -- green, fill
    "spike_dip": "#E64980",             # behavioural -- pink, points
}

FLAG_MARKER_SIZE = 6
FLAG_MARKER_OPACITY = 0.6

CHECK_LABELS_L1 = {
    "missing_values": "Missing values",
    "long_gaps": "Long gaps",
    "negative_discharge": "Negative discharge",
    "duplicate_timestamps": "Duplicate timestamps",
    "timestep_consistency": "Timestep consistency",
    "zero_flow_regime": "Zero-flow regime",
    "low_variability": "Low variability",
    "spike_dip": "Spike / dip",
}


# ==========================================================
# SHARED HELPERS
# ==========================================================

def _save(fig: go.Figure, path: Path, show: bool = False) -> None:
    """
    Save a figure to disk, and optionally display it first (in a
    notebook, this renders inline exactly where the cell runs -- the
    figure is shown then written, one at a time, not all shown and
    then all saved afterwards).
    """
    if show:
        fig.show()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


def _flow_duration_curve(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    valid = series.dropna()
    sorted_flow = np.sort(valid.to_numpy())[::-1]
    n = len(sorted_flow)
    if n == 0:
        return np.array([]), np.array([])
    ranks = np.arange(1, n + 1)
    return ranks / (n + 1) * 100, sorted_flow


def _group_into_spans(timestamps: list) -> list[tuple]:
    """Group a sorted list of timestamps into (start, end) spans of
    consecutive days, so a run of flagged days can be shaded as one
    band instead of many separate marks."""
    if not timestamps:
        return []
    dates = sorted(pd.to_datetime(timestamps))
    spans, span_start, previous = [], dates[0], dates[0]
    for current in dates[1:]:
        if (current - previous).days > 1:
            spans.append((span_start, previous))
            span_start = current
        previous = current
    spans.append((span_start, previous))
    return spans


def _shapes_for_spans(spans: list[tuple], color: str) -> list[dict]:
    """Build vrect shape dicts for a batch layout update. Adding shapes
    one at a time via add_vrect() is very slow once there are more
    than a few dozen -- some checks (e.g. derivative-sign events) can
    have hundreds, so every span-shading plot in this file batches
    shapes through fig.update_layout(shapes=[...]) instead."""
    return [
        dict(type="rect", xref="x", yref="paper", x0=start, x1=end + pd.Timedelta(days=1),
             y0=0, y1=1, fillcolor=color, line_width=0)
        for start, end in spans
    ]


def _obs_model_lines(obs_x, obs_y, model_x, model_y, title, x_label, y_label,
                      obs_name="OBS", model_name="Model", y_log=False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs_x, y=obs_y, mode="lines+markers", name=obs_name,
                              line=dict(color=OBS_COLOR, width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=model_x, y=model_y, mode="lines+markers", name=model_name,
                              line=dict(color=MODEL_COLOR, width=1.5), marker=dict(size=4)))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                       template=PLOT_TEMPLATE, height=420)
    if y_log:
        fig.update_yaxes(type="log")
    return fig


def _obs_model_bars(categories, obs_values, model_values, title, x_label, y_label,
                     obs_name="OBS", model_name="Model") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=obs_values, name=obs_name, marker_color=OBS_COLOR))
    fig.add_trace(go.Bar(x=categories, y=model_values, name=model_name, marker_color=MODEL_COLOR))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                       barmode="group", template=PLOT_TEMPLATE, height=420)
    return fig


def _obs_model_histogram(obs_values, model_values, title, x_label,
                          obs_name="OBS", model_name="Model") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=obs_values, name=obs_name, marker_color=OBS_COLOR, opacity=0.6))
    fig.add_trace(go.Histogram(x=model_values, name=model_name, marker_color=MODEL_COLOR, opacity=0.6))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Count",
                       barmode="overlay", template=PLOT_TEMPLATE, height=420)
    return fig


# ==========================================================
# LAYER 1: ONE COMBINED PLOT FOR THE 8 "SIMPLE" CHECKS
# (5 basic + zero_flow_regime + low_variability + spike_dip)
# ==========================================================

def plot_layer1_combined(series: pd.Series, check_results: list[dict], series_type: str) -> go.Figure:
    """
    One figure, one discharge line, all 8 non-step-shift/non-drift
    checks overlaid -- each with its own colour (LAYER1_CHECK_COLORS)
    so they stay visually distinguishable when several fire on the
    same date range. Every trace can also be toggled individually via
    the legend, so "one file" and "inspect one check at a time" are
    not actually in tension.

    Gaps (missing_values, long_gaps) are shaded spans, not points --
    a point can't represent a day that has no value to plot at.
    Everything else is small, semi-transparent point markers on top
    of the line (mentor feedback).
    """
    by_check = {r["check"]: r for r in check_results}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                              line=dict(color="#ADB5BD", width=1), name="Discharge"))

    all_shapes = []

    # --- gap-style checks: filled spans ---
    for check in ["missing_values", "long_gaps", "zero_flow_regime", "low_variability"]:
        result = by_check.get(check)
        if result is None:
            continue

        if check == "zero_flow_regime":
            # This check is a whole-record descriptor (ratio, not a
            # per-day list) -- recompute the zero-flow mask directly
            # from the series for plotting purposes only.
            rounded = series.round(3)
            timestamps = list(series.index[rounded == 0])
        else:
            timestamps = result.get("flagged_timestamps", [])

        spans = _group_into_spans(timestamps)
        color = LAYER1_CHECK_COLORS[check]
        all_shapes.extend(_shapes_for_spans(spans, _hex_to_rgba(color, 0.18)))
        # Dummy trace so this check gets its own legend entry (shapes
        # don't appear in the legend on their own).
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                  marker=dict(size=10, color=color, symbol="square"),
                                  name=CHECK_LABELS_L1[check]))

    fig.update_layout(shapes=all_shapes)

    # --- point-style checks: small, semi-transparent markers ---
    for check in ["negative_discharge", "duplicate_timestamps", "timestep_consistency"]:
        result = by_check.get(check)
        if result is None or not result.get("flagged_timestamps"):
            continue
        idx = pd.to_datetime(result["flagged_timestamps"])
        fig.add_trace(go.Scatter(
            x=idx, y=series.reindex(idx).values, mode="markers",
            marker=dict(size=FLAG_MARKER_SIZE, color=LAYER1_CHECK_COLORS[check],
                        opacity=FLAG_MARKER_OPACITY, line=dict(width=0)),
            name=CHECK_LABELS_L1[check],
        ))

    # --- spike/dip: same check colour, shape distinguishes spike vs dip ---
    spike_dip_result = by_check.get("spike_dip")
    if spike_dip_result is not None:
        details = {d["timestamp"]: d for d in spike_dip_result.get("candidate_details", [])}
        for kind, symbol in [("spike", "triangle-up"), ("dip", "triangle-down")]:
            ts = [t for t, d in details.items() if d["type"] == kind]
            if not ts:
                continue
            idx = pd.to_datetime(ts)
            fig.add_trace(go.Scatter(
                x=idx, y=series.reindex(idx).values, mode="markers",
                marker=dict(size=FLAG_MARKER_SIZE, color=LAYER1_CHECK_COLORS["spike_dip"],
                            opacity=FLAG_MARKER_OPACITY, symbol=symbol, line=dict(width=0)),
                name=f"Spike / dip ({kind})",
            ))

    fig.update_layout(
        title=f"Layer 1 combined checks ({series_type})",
        xaxis_title="Date", yaxis_title="Discharge",
        template=PLOT_TEMPLATE, height=480,
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


# ==========================================================
# LAYER 1: STEP SHIFT (separate plot)
# ==========================================================

def plot_step_shift(series: pd.Series, step_shift_result: dict, series_type: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                              line=dict(color="#4C6EF5", width=1), name="Discharge"))

    for regime in step_shift_result.get("regime_summary", []):
        start, end = pd.Timestamp(regime["start"]), pd.Timestamp(regime["end"])
        fig.add_trace(go.Scatter(x=[start, end], y=[regime["mean_flow"]] * 2, mode="lines",
                                  line=dict(color="#2F9E44", width=3), showlegend=False))

    for boundary in step_shift_result.get("regime_boundaries", []):
        boundary_date = pd.Timestamp(boundary["boundary_timestamp"])
        fig.add_vline(x=boundary_date, line_dash="dash",
                       line_color="#E8590C" if boundary.get("robust") else "grey")
        fig.add_annotation(x=boundary_date, y=series.max(), text=f"p={boundary['p_value']:.2e}",
                            showarrow=False, yshift=10, font=dict(size=9))

    fig.update_layout(title=f"Step shift ({series_type})<br><sub>{step_shift_result.get('message', '')}</sub>",
                       xaxis_title="Date", yaxis_title="Discharge", template=PLOT_TEMPLATE, height=440)
    return fig


# ==========================================================
# LAYER 1: GRADUAL DRIFT (separate, 2 plots -- trend + FDC)
# ==========================================================

def _recompute_gradual_drift_trend(series: pd.Series, min_daily_values_per_month: int = 20):
    """
    Recreate the monthly series + Seasonal Sen trend line for plotting.
    check_gradual_drift() only returns scalar diagnostics, not the full
    monthly series, so this repeats the same lightweight aggregation
    for plotting purposes only -- it does not change or re-decide
    anything the check already decided.
    """
    daily_flow = series.groupby(series.index).median().sort_index()
    daily_index = pd.date_range(daily_flow.index.min(), daily_flow.index.max(), freq="D")
    daily_flow = daily_flow.reindex(daily_index)

    monthly_flow = daily_flow.resample("MS").median()
    monthly_valid_count = daily_flow.resample("MS").count()
    monthly_flow = monthly_flow.where(monthly_valid_count >= min_daily_values_per_month)

    first_valid, last_valid = monthly_flow.first_valid_index(), monthly_flow.last_valid_index()
    if first_valid is None or last_valid is None:
        return daily_flow, pd.Series(dtype=float), pd.Series(dtype=float)

    monthly_flow = monthly_flow.loc[first_valid:last_valid]
    monthly_analysis = monthly_flow.interpolate(method="time", limit=2, limit_area="inside")

    if monthly_analysis.isna().any() or len(monthly_analysis) < 36 or mk is None:
        return daily_flow, monthly_analysis, pd.Series(dtype=float)

    seasonal_mk = mk.seasonal_test(monthly_analysis.values, period=12)
    elapsed_years = np.arange(len(monthly_analysis)) / 12
    intercept = getattr(seasonal_mk, "intercept",
                         monthly_analysis.median() - seasonal_mk.slope * np.median(elapsed_years))
    sen_line = pd.Series(intercept + seasonal_mk.slope * elapsed_years, index=monthly_analysis.index)
    return daily_flow, monthly_analysis, sen_line


def plot_gradual_drift_trend(series: pd.Series, gradual_drift_result: dict, series_type: str) -> go.Figure:
    daily_flow, monthly_analysis, sen_line = _recompute_gradual_drift_trend(series)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_flow.index, y=daily_flow.values, mode="lines",
                              line=dict(color="#4C6EF5", width=0.6), opacity=0.4, name="Daily discharge"))
    if not monthly_analysis.empty:
        fig.add_trace(go.Scatter(x=monthly_analysis.index, y=monthly_analysis.values, mode="lines+markers",
                                  line=dict(color="#2F9E44", width=1), marker=dict(size=3), name="Monthly median"))
    if not sen_line.empty:
        fig.add_trace(go.Scatter(x=sen_line.index, y=sen_line.values, mode="lines",
                                  line=dict(color="#E8590C", width=3), name="Seasonal Sen drift line"))

    fig.update_layout(title=f"Gradual drift -- trend ({series_type})<br><sub>{gradual_drift_result.get('message', '')}</sub>",
                       xaxis_title="Date", yaxis_title="Discharge", template=PLOT_TEMPLATE, height=420)
    return fig


def plot_gradual_drift_fdc(series: pd.Series, series_type: str) -> go.Figure:
    valid = series.dropna()
    midpoint = valid.index.min() + (valid.index.max() - valid.index.min()) / 2
    first_half, second_half = valid.loc[:midpoint], valid.loc[midpoint:]
    exceed_1, flow_1 = _flow_duration_curve(first_half)
    exceed_2, flow_2 = _flow_duration_curve(second_half)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exceed_1, y=flow_1, mode="lines", line=dict(color="#2F9E44", width=2),
                              name=f"First half (to {first_half.index.max().date()})"))
    fig.add_trace(go.Scatter(x=exceed_2, y=flow_2, mode="lines", line=dict(color="#E8590C", width=2),
                              name=f"Second half (from {second_half.index.min().date()})"))
    fig.update_layout(title=f"Gradual drift -- flow duration curve, first half vs. second half ({series_type})",
                       xaxis_title="Percent of time flow equalled or exceeded (%)",
                       yaxis_title="Discharge (log scale)", yaxis_type="log",
                       template=PLOT_TEMPLATE, height=420)
    return fig


# ==========================================================
# LAYER 1: MAIN ENTRY POINT
# ==========================================================

def generate_layer1_visuals(
    obs_series: pd.Series,
    sim_series: Optional[pd.Series],
    layer1_diagnostics: dict,
    station_id: str = "station",
    model_name: str = "AIFL",
    output_root: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Build every Layer 1 visual from diagnostics.py's output and save
    them into <IO_OUTPUT_ROOT>/<station_id>/layer1/ -- station name
    first, then layer, matching the actual repo layout. Takes `layer1_diagnostics`
    (the dict returned by diagnostics.run_layer1_diagnostics) rather
    than re-running any check.

    show=True displays each figure inline (e.g. in a notebook) right
    before it's saved. Defaults to False here since this file is also
    meant to run unattended (a batch across many stations); the
    notebook-facing run_layer1() in layer1/layer1.py sets show=True.
    """
    output_root = Path(output_root) if output_root is not None else IO_OUTPUT_ROOT
    station_dir = output_root / station_id / "layer1"
    station_dir.mkdir(parents=True, exist_ok=True)

    raw_results = layer1_diagnostics["raw_results"]

    for series_type, series in [("obs", obs_series)] + ([(model_name, sim_series)] if sim_series is not None else []):
        by_check = {r["check"]: r for r in raw_results[series_type]}

        _save(plot_layer1_combined(series, raw_results[series_type], series_type),
              station_dir / f"{series_type}_combined_checks.html", show=show)

        if "step_shift" in by_check:
            _save(plot_step_shift(series, by_check["step_shift"], series_type),
                  station_dir / f"{series_type}_step_shift.html", show=show)

        if "gradual_drift" in by_check:
            _save(plot_gradual_drift_trend(series, by_check["gradual_drift"], series_type),
                  station_dir / f"{series_type}_gradual_drift_trend.html", show=show)
            _save(plot_gradual_drift_fdc(series, series_type),
                  station_dir / f"{series_type}_gradual_drift_fdc.html", show=show)

    layer1_diagnostics["eda_summary"].to_csv(station_dir / "layer1_eda_summary.csv", index=False)
    layer1_diagnostics["summary_all"].to_csv(station_dir / "layer1_summary_all.csv", index=False)
    layer1_diagnostics["summary_flagged"].to_csv(station_dir / "layer1_summary_flagged.csv", index=False)

    return station_dir


# ==========================================================
# LAYER 2: DATA-QUALITY CONTEXT PLOTS
# (missing-value markers, temporary-fill highlight)
# ==========================================================

def plot_missing_values(obs_series: pd.Series) -> go.Figure:
    """Raw OBS hydrograph with every missing timestamp marked, matching
    Layer2_demo1_OBS-ML-Comparison.ipynb's original visual."""
    missing_dates = obs_series.index[obs_series.isna()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs_series.index, y=obs_series.values, mode="lines",
                              line=dict(color=OBS_COLOR, width=0.8), name="Observed"))

    # Batched shapes, not one add_vline() call per date -- calling a
    # shape-adding method once per point is very slow once there are
    # more than a few dozen (same issue already hit and fixed for
    # event shading further down this file).
    shapes = [
        dict(type="line", xref="x", yref="paper", x0=d, x1=d, y0=0, y1=1,
             line=dict(color="rgba(200,0,0,0.15)", width=1))
        for d in missing_dates
    ]
    fig.update_layout(shapes=shapes)
    fig.update_layout(title=f"OBS missing values ({len(missing_dates)} day(s))",
                       xaxis_title="Date", yaxis_title="Discharge", template=PLOT_TEMPLATE, height=420)
    return fig


def plot_temporary_fill_highlight(obs_series: pd.Series) -> go.Figure:
    """
    Raw OBS hydrograph with temporarily-filled values highlighted.

    NOTE -- temporary stand-in: this uses the EXISTING seasonal-
    climatology fill (fill_obs_for_layer2, day-of-year +/- 15 days,
    minimum 5 samples) purely so this plot exists and works today.
    It is not the tiered (gap-length-aware, dry-season-vs-event-aware)
    imputation approach discussed separately -- swap the call below
    once that function exists. Nothing here modifies obs_series itself;
    the fill is computed on a copy, for this plot only.
    """
    filled = fill_obs_for_layer2(obs_series, method="seasonal_climatology")
    # fill_obs_for_layer2 trims to the valid OBS record internally (it
    # will not invent values before the first or after the last real
    # observation), so its output can be shorter than obs_series if
    # there's a trailing/leading stretch with no OBS at all (e.g. a
    # model that forecasts further than OBS actually runs). Reindex
    # back onto the original index before comparing, so this doesn't
    # crash on a length mismatch.
    filled = filled.reindex(obs_series.index)
    filled_mask = obs_series.isna() & filled.notna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs_series.index, y=obs_series.values, mode="lines",
                              line=dict(color=OBS_COLOR, width=0.8), name="Observed"))
    fig.add_trace(go.Scatter(x=filled.index[filled_mask], y=filled[filled_mask].values, mode="markers",
                              marker=dict(size=5, color="#2F9E44", opacity=0.7),
                              name="Temporarily filled (calculation only)"))
    fig.update_layout(title=f"OBS with temporary fill highlighted ({int(filled_mask.sum())} day(s) filled)"
                             "<br><sub>Provisional method -- see docstring</sub>",
                       xaxis_title="Date", yaxis_title="Discharge", template=PLOT_TEMPLATE, height=420)
    return fig


# ==========================================================
# LAYER 2: ONE VISUAL PER SIGNATURE CHECK
# ==========================================================

def _hydrograph_with_events(obs_series, model_series, obs_events, model_events, title,
                             model_name="Model", start_col="start_date", end_col="end_date",
                             peak_col="maximum_flow", zoom_years: Optional[float] = None) -> go.Figure:
    """
    Both series plotted together; OBS events shaded, model event peaks
    marked as small semi-transparent dots. `zoom_years`, if given,
    restricts the plotted window to the most recent N years so a
    check with hundreds of events (e.g. derivative-sign) stays
    readable instead of becoming one solid shaded block.
    """
    if zoom_years is not None:
        end = min(obs_series.index.max(), model_series.index.max())
        start = end - pd.DateOffset(years=zoom_years)
        obs_series = obs_series.loc[start:end]
        model_series = model_series.loc[start:end]
        if obs_events is not None and not obs_events.empty:
            obs_events = obs_events[(obs_events[end_col] >= start) & (obs_events[start_col] <= end)]
        if model_events is not None and not model_events.empty:
            model_events = model_events[(model_events[end_col] >= start) & (model_events[start_col] <= end)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs_series.index, y=obs_series.values, mode="lines",
                              name="OBS", line=dict(color=OBS_COLOR, width=0.8)))
    fig.add_trace(go.Scatter(x=model_series.index, y=model_series.values, mode="lines",
                              name=model_name, line=dict(color=MODEL_COLOR, width=0.8), opacity=0.8))

    if obs_events is not None and not obs_events.empty:
        shapes = [dict(type="rect", xref="x", yref="paper", x0=row[start_col], x1=row[end_col],
                        y0=0, y1=1, fillcolor=OBS_SHADE, line_width=0) for _, row in obs_events.iterrows()]
        fig.update_layout(shapes=shapes)

    if model_events is not None and not model_events.empty and start_col in model_events:
        mid_dates = pd.to_datetime(model_events[start_col]) + (
            pd.to_datetime(model_events[end_col]) - pd.to_datetime(model_events[start_col])) / 2
        peak_values = model_events[peak_col] if peak_col in model_events else [model_series.max()] * len(model_events)
        fig.add_trace(go.Scatter(x=mid_dates, y=peak_values, mode="markers",
                                  marker=dict(size=5, color=MODEL_COLOR, opacity=0.55),
                                  name=f"{model_name} event midpoints"))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Discharge",
                       template=PLOT_TEMPLATE, height=440)
    return fig


def generate_layer2_visuals(
    obs_series: pd.Series,
    ml_series: pd.Series,
    layer2_diagnostics: dict,
    station_id: str = "station",
    model_name: str = "AIFL",
    output_root: Optional[Path] = None,
    event_zoom_years: float = 2.0,
    show: bool = False,
) -> Path:
    """
    Build every Layer 2 visual from diagnostics.py's output and save
    them into <IO_OUTPUT_ROOT>/<station_id>/layer2/ -- station name
    first, then layer, matching the actual repo layout. Takes `layer2_diagnostics`
    (the dict returned by diagnostics.run_layer2_diagnostics) rather
    than recomputing any signature.

    show=True displays each figure inline (e.g. in a notebook) right
    before it's saved. Defaults to False here since this file is also
    meant to run unattended (a batch across many stations); the
    notebook-facing run_layer2() in layer2/layer2.py sets show=True.
    """
    output_root = Path(output_root) if output_root is not None else IO_OUTPUT_ROOT
    station_dir = output_root / station_id / "layer2"
    station_dir.mkdir(parents=True, exist_ok=True)

    obs_results = layer2_diagnostics["obs_results"]
    model_results = layer2_diagnostics["model_results"]
    obs_clean, model_clean = obs_series.dropna(), ml_series.dropna()

    figures = {}

    # --- data-quality context (new, from the notebook audit) ---
    figures["missing_values"] = plot_missing_values(obs_series)
    figures["temporary_fill_highlight"] = plot_temporary_fill_highlight(obs_series)

    # --- flow magnitude: FDC ---
    obs_x, obs_y = _flow_duration_curve(obs_clean)
    model_x, model_y = _flow_duration_curve(model_clean)
    figures["flow_magnitude"] = _obs_model_lines(
        obs_x, obs_y, model_x, model_y, title="Flow magnitude -- flow duration curve",
        x_label="Percent of time flow equalled or exceeded (%)", y_label="Discharge (log scale)",
        model_name=model_name, y_log=True,
    )

    # --- low / high flow ---
    figures["low_flow"] = _hydrograph_with_events(
        obs_clean, model_clean, obs_results["low_flow"].tables["low_flow_events"],
        model_results["low_flow"].tables["low_flow_events"],
        title=f"Low flow (OBS threshold = {obs_results['low_flow'].metrics.get('low_flow_threshold', float('nan')):.2f})",
        model_name=model_name,
    )
    figures["high_flow"] = _hydrograph_with_events(
        obs_clean, model_clean, obs_results["high_flow"].tables["high_flow_events"],
        model_results["high_flow"].tables["high_flow_events"],
        title=f"High flow (OBS threshold = {obs_results['high_flow'].metrics.get('high_flow_threshold', float('nan')):.2f})",
        model_name=model_name,
    )

    # --- annual maximum / zero flow ---
    obs_am, model_am = obs_results["annual_maximum"].tables["annual_maxima"], model_results["annual_maximum"].tables["annual_maxima"]
    figures["annual_maximum"] = _obs_model_bars(obs_am["year"], obs_am["annual_maximum"], model_am["annual_maximum"],
                                                 "Annual maximum discharge", "Year", "Discharge", model_name=model_name)

    obs_zf = obs_results["zero_flow"].tables["annual_zero_flow_frequency"]
    model_zf = model_results["zero_flow"].tables["annual_zero_flow_frequency"]
    figures["zero_flow"] = _obs_model_bars(obs_zf["year"], obs_zf["zero_flow_days"], model_zf["zero_flow_days"],
                                            "Zero-flow days per year", "Year", "Zero-flow days", model_name=model_name)

    # --- baseflow ---
    obs_bf = obs_results["baseflow"].tables["baseflow_series"]["baseflow"]
    model_bf = model_results["baseflow"].tables["baseflow_series"]["baseflow"]
    bf_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(f"OBS (BFI={obs_results['baseflow'].metrics['baseflow_index']:.2f})",
                                            f"{model_name} (BFI={model_results['baseflow'].metrics['baseflow_index']:.2f})"))
    bf_fig.add_trace(go.Scatter(x=obs_clean.index, y=obs_clean.values, mode="lines",
                                 line=dict(color=OBS_COLOR, width=0.6), name="Total flow", opacity=0.5), row=1, col=1)
    bf_fig.add_trace(go.Scatter(x=obs_bf.index, y=obs_bf.values, mode="lines",
                                 line=dict(color=OBS_COLOR, width=1.5), name="Baseflow"), row=1, col=1)
    bf_fig.add_trace(go.Scatter(x=model_clean.index, y=model_clean.values, mode="lines",
                                 line=dict(color=MODEL_COLOR, width=0.6), name="Total flow", opacity=0.5), row=2, col=1)
    bf_fig.add_trace(go.Scatter(x=model_bf.index, y=model_bf.values, mode="lines",
                                 line=dict(color=MODEL_COLOR, width=1.5), name="Baseflow"), row=2, col=1)
    bf_fig.update_layout(title="Baseflow separation (Lyne-Hollick)", template=PLOT_TEMPLATE, height=560)
    figures["baseflow"] = bf_fig

    # --- seasonality ---
    obs_season = obs_results["seasonality"].tables["monthly_climatology_mean"]
    model_season = model_results["seasonality"].tables["monthly_climatology_mean"]
    obs_si = obs_results["seasonality"].metrics.get("walsh_lawler_seasonality_index", float("nan"))
    model_si = model_results["seasonality"].metrics.get("walsh_lawler_seasonality_index", float("nan"))
    season_fig = _obs_model_lines(
        obs_season["month"], obs_season["climatological_monthly_mean"],
        model_season["month"], model_season["climatological_monthly_mean"],
        title=f"Seasonality -- monthly climatology (Walsh-Lawler SI: OBS={obs_si:.2f}, {model_name}={model_si:.2f})",
        x_label="Month", y_label="Mean discharge", model_name=model_name,
    )
    season_fig.update_xaxes(tickmode="linear", dtick=1)
    figures["seasonality"] = season_fig

    # --- flashiness / autocorrelation ---
    obs_fl, model_fl = obs_results["flashiness"].tables["annual_flashiness"], model_results["flashiness"].tables["annual_flashiness"]
    figures["flashiness"] = _obs_model_lines(obs_fl["year"], obs_fl["flashiness_index"], model_fl["year"], model_fl["flashiness_index"],
                                              "Annual flashiness index (Richards-Baker)", "Year", "Flashiness index", model_name=model_name)

    obs_acf, model_acf = obs_results["autocorrelation"].tables["autocorrelation_function"], model_results["autocorrelation"].tables["autocorrelation_function"]
    acf_fig = _obs_model_lines(obs_acf["lag"], obs_acf["autocorrelation"], model_acf["lag"], model_acf["autocorrelation"],
                                "Autocorrelation function", "Lag (days)", "Autocorrelation", model_name=model_name)
    acf_fig.add_hline(y=1 / np.e, line_dash="dot", line_color="grey", annotation_text="1/e decorrelation threshold")
    figures["autocorrelation"] = acf_fig

    # --- rising / recession limb ---
    figures["rising_limb"] = _obs_model_histogram(
        obs_results["rising_limb"].tables["rising_rates"]["rising_rate"].dropna(),
        model_results["rising_limb"].tables["rising_rates"]["rising_rate"].dropna(),
        "Rising-limb rate distribution", "Rising rate (discharge/day)", model_name=model_name,
    )
    figures["recession_limb"] = _obs_model_histogram(
        obs_results["recession_limb"].tables["recession_rates"]["recession_rate"].dropna(),
        model_results["recession_limb"].tables["recession_rates"]["recession_rate"].dropna(),
        "Recession-limb rate distribution", "Recession rate (discharge/day)", model_name=model_name,
    )

    # --- peaks (zoomed to recent years, plus peak markers) ---
    end = min(obs_clean.index.max(), model_clean.index.max())
    start = end - pd.DateOffset(years=event_zoom_years)
    peaks_fig = go.Figure()
    peaks_fig.add_trace(go.Scatter(x=obs_clean.loc[start:end].index, y=obs_clean.loc[start:end].values,
                                    mode="lines", name="OBS", line=dict(color=OBS_COLOR, width=0.9)))
    peaks_fig.add_trace(go.Scatter(x=model_clean.loc[start:end].index, y=model_clean.loc[start:end].values,
                                    mode="lines", name=model_name, line=dict(color=MODEL_COLOR, width=0.9), opacity=0.8))
    obs_peaks = obs_results["peaks"].tables["peaks"]
    model_peaks = model_results["peaks"].tables["peaks"]
    if not obs_peaks.empty:
        p = obs_peaks[(obs_peaks["peak_date"] >= start) & (obs_peaks["peak_date"] <= end)]
        peaks_fig.add_trace(go.Scatter(x=p["peak_date"], y=p["peak_value"], mode="markers",
                                        marker=dict(size=7, color=OBS_COLOR, symbol="circle"), name="OBS peaks"))
    if not model_peaks.empty:
        p = model_peaks[(model_peaks["peak_date"] >= start) & (model_peaks["peak_date"] <= end)]
        peaks_fig.add_trace(go.Scatter(x=p["peak_date"], y=p["peak_value"], mode="markers",
                                        marker=dict(size=7, color=MODEL_COLOR, symbol="x"), name=f"{model_name} peaks"))
    peaks_fig.update_layout(title=f"Detected peaks (latest {event_zoom_years:.0f} years)",
                             xaxis_title="Date", yaxis_title="Discharge", template=PLOT_TEMPLATE, height=440)
    figures["peaks"] = peaks_fig

    # --- threshold-based / derivative-sign events (zoomed) ---
    figures["threshold_event_hydrographs"] = _hydrograph_with_events(
        obs_clean, model_clean, obs_results["threshold_event_hydrographs"].tables["events"],
        model_results["threshold_event_hydrographs"].tables["events"],
        title="Threshold-based (Q95) event hydrographs", model_name=model_name,
        start_col="event_start", end_col="event_end", peak_col="peak_flow",
        zoom_years=event_zoom_years,
    )
    figures["derivative_event_hydrographs"] = _hydrograph_with_events(
        obs_clean, model_clean, obs_results["derivative_event_hydrographs"].tables["events"],
        model_results["derivative_event_hydrographs"].tables["events"],
        title="Derivative-sign event hydrographs", model_name=model_name,
        start_col="event_start", end_col="event_end", peak_col="peak_flow",
        zoom_years=event_zoom_years,
    )

    # --- flashiness persistence ---
    obs_fp = obs_results["flashiness_persistence"].tables["rolling_flashiness_autocorrelation"]
    model_fp = model_results["flashiness_persistence"].tables["rolling_flashiness_autocorrelation"]
    fp_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Rolling flashiness", "Rolling lag-1 autocorrelation"))
    fp_fig.add_trace(go.Scatter(x=obs_fp.index, y=obs_fp["rolling_flashiness"], mode="lines",
                                 line=dict(color=OBS_COLOR, width=1), name="OBS"), row=1, col=1)
    fp_fig.add_trace(go.Scatter(x=model_fp.index, y=model_fp["rolling_flashiness"], mode="lines",
                                 line=dict(color=MODEL_COLOR, width=1), name=model_name), row=1, col=1)
    fp_fig.add_trace(go.Scatter(x=obs_fp.index, y=obs_fp["rolling_ac1"], mode="lines",
                                 line=dict(color=OBS_COLOR, width=1), showlegend=False), row=2, col=1)
    fp_fig.add_trace(go.Scatter(x=model_fp.index, y=model_fp["rolling_ac1"], mode="lines",
                                 line=dict(color=MODEL_COLOR, width=1), showlegend=False), row=2, col=1)
    fp_fig.update_layout(title="Flashiness persistence over time", template=PLOT_TEMPLATE, height=560)
    figures["flashiness_persistence"] = fp_fig

    for name, fig in figures.items():
        _save(fig, station_dir / f"{name}.html", show=show)

    layer2_diagnostics["compact_comparison"].to_csv(station_dir / "layer2_compact_comparison.csv", index=False)
    layer2_diagnostics["full_comparison"].to_csv(station_dir / "layer2_summary_all.csv", index=False)
    layer2_diagnostics["full_comparison_flagged"].to_csv(station_dir / "layer2_summary_flagged.csv", index=False)
    layer2_diagnostics["percentile_diagnostics"].to_csv(station_dir / "layer2_percentile_diagnostics.csv", index=False)

    return station_dir


# ==========================================================
# LAYER 3: CONTEXT COMPARISON PLOT
# ==========================================================

def plot_context_period(
    target_series: pd.Series,
    candidate_series: dict[str, pd.Series],
    station_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> go.Figure:
    """
    Robust-normalised target vs. context candidates over one
    manageable period (plotting 40+ years of daily data on one axis is
    technically possible and visually close to spilled noodles --
    pick a flood period, a suspicious interval, or one representative
    year instead). All series are lightly smoothed (3-day centered
    rolling median) purely to make the comparison readable.
    """
    from src.trihydra.layer3.discharge_comparison import robust_normalise

    fig = go.Figure()

    def _prepare(series: pd.Series) -> pd.Series:
        sliced = series.loc[start:end] if (start or end) else series
        normalised = robust_normalise(sliced)
        return normalised.rolling(3, center=True, min_periods=1).median()

    fig.add_trace(go.Scatter(
        x=target_series.index, y=_prepare(target_series).values, mode="lines",
        line=dict(color=OBS_COLOR, width=2), name=f"Target: {station_id}",
    ))

    palette = ["#2F9E44", "#F08C00", "#5F3DC4", "#0C8599", "#C92A2A"]
    for i, (candidate_id, series) in enumerate(candidate_series.items()):
        smoothed = _prepare(series)
        fig.add_trace(go.Scatter(
            x=smoothed.index, y=smoothed.values, mode="lines",
            line=dict(color=palette[i % len(palette)], width=1.3),
            name=f"Context: {candidate_id}",
        ))

    fig.update_layout(
        title=f"{station_id}: robust-normalised target and context gauges",
        xaxis_title="Date", yaxis_title="(Q - median) / IQR",
        template=PLOT_TEMPLATE, height=440,
    )
    return fig


def generate_layer3_visuals(
    target_series: pd.Series,
    candidate_series: dict[str, pd.Series],
    layer3_diagnostics: dict,
    station_id: str = "station",
    output_root: Optional[Path] = None,
    show: bool = False,
    period: Optional[tuple] = None,
) -> Path:
    """
    Build the Layer 3 comparison-period plot from already-computed
    results and save it into <IO_OUTPUT_ROOT>/<station_id>/layer3/.
    Takes `layer3_diagnostics` (the dict returned by
    diagnostics.run_layer3_diagnostics) rather than recomputing
    anything, and `candidate_series` (already-loaded candidate
    discharge, from nc_loader.py via layer3.py) since this file never
    loads discharge data itself.

    The gauge-network maps are a separate concern -- see mapviz.py's
    generate_layer3_maps -- since they're a different rendering
    technology (folium/Leaflet) from everything else in this file.
    """
    output_root = Path(output_root) if output_root is not None else IO_OUTPUT_ROOT
    station_dir = output_root / station_id / "layer3"
    station_dir.mkdir(parents=True, exist_ok=True)

    start, end = period if period is not None else (None, None)

    if candidate_series:
        fig = plot_context_period(target_series, candidate_series, station_id, start=start, end=end)
        _save(fig, station_dir / "context_comparison.html", show=show)

    layer3_diagnostics["context_summary"].to_csv(station_dir / "layer3_context_summary.csv", index=False)
    layer3_diagnostics["comparison_table"].to_csv(station_dir / "layer3_comparison_table.csv", index=False)
    (station_dir / "layer3_interpretation.txt").write_text(layer3_diagnostics["interpretation"])

    return station_dir


if __name__ == "__main__":
    print(
        "This module is meant to be imported. Call "
        "generate_layer1_visuals(obs, sim, layer1_diagnostics, station_id=..., model_name=...), "
        "generate_layer2_visuals(obs, ml, layer2_diagnostics, station_id=..., model_name=...), "
        "and/or generate_layer3_visuals(target_series, candidate_series, layer3_diagnostics, station_id=...) "
        "using the output of diagnostics.run_layer1_diagnostics / run_layer2_diagnostics / run_layer3_diagnostics."
    )
