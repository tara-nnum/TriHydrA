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
from display_format import DISPLAY_DECIMALS, format_display_number


# ==========================================================
# SHARED SETTINGS
# ==========================================================

PLOT_TEMPLATE = "plotly_white"
OBS_COLOR = "#1971C2"
MODEL_COLOR = "#E8590C"
OBS_SHADE = "rgba(25, 113, 194, 0.15)"
MODEL_SHADE = "rgba(232, 89, 12, 0.15)"
def _format_display_number(value) -> str:
    """Compatibility alias for the shared Layer 2 presentation formatter."""
    return format_display_number(value)


def _csv_display_number(value: float) -> str:
    """Pandas ``float_format`` adapter using the shared display rule."""
    return _format_display_number(value)


def _apply_layer2_number_format(fig: go.Figure) -> go.Figure:
    """Limit visible axis and hover numbers without altering source values."""
    fig.update_xaxes(tickformat=".3~g", hoverformat=".3~g")
    fig.update_yaxes(tickformat=".3~g", hoverformat=".3~g")
    return fig

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


def _save_csv_resilient(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV, using ``_updated`` when the existing file is open."""
    try:
        frame.to_csv(path, index=False, float_format=_csv_display_number)
        return path
    except PermissionError:
        updated_path = path.with_name(f"{path.stem}_updated{path.suffix}")
        frame.to_csv(
            updated_path, index=False, float_format=_csv_display_number
        )
        return updated_path


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


def _threshold_comparison_figure(
    obs_series: pd.Series,
    model_series: pd.Series,
    obs_result,
    model_result,
    flow_kind: str,
    threshold: float,
    unit: str,
    model_name: str,
) -> go.Figure:
    """Show a fixed raw-OBS threshold without ambiguous event shading."""
    is_low = flow_kind == "low"
    comparison = (
        (lambda values: values <= threshold)
        if is_low
        else (lambda values: values >= threshold)
    )
    obs_mask = comparison(obs_series) & obs_series.notna()
    model_mask = comparison(model_series) & model_series.notna()
    metric_prefix = "low_flow" if is_low else "high_flow"
    relation = "at or below" if is_low else "at or above"
    percentile = "Q05" if is_low else "Q95"

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.78, 0.22],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=obs_series.index, y=obs_series.values, mode="lines",
            name="OBS", line=dict(color=OBS_COLOR, width=0.8),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=model_series.index, y=model_series.values, mode="lines",
            name=model_name, line=dict(color=MODEL_COLOR, width=0.8),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=obs_series.index[obs_mask],
            y=obs_series.loc[obs_mask],
            mode="markers",
            name=f"OBS {flow_kind}-flow days",
            marker=dict(color=OBS_COLOR, size=4, opacity=0.65),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=model_series.index[model_mask],
            y=model_series.loc[model_mask],
            mode="markers",
            name=f"{model_name} {flow_kind}-flow days",
            marker=dict(color=MODEL_COLOR, size=4, opacity=0.65),
        ),
        row=1, col=1,
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#5F3DC4",
        annotation_text=f"Raw OBS {percentile} = {threshold:.3f} {unit}",
        row=1,
        col=1,
    )
    metric_names = [
        f"{metric_prefix}_days",
        f"{metric_prefix}_event_count",
        f"median_{metric_prefix}_duration",
        f"maximum_{metric_prefix}_duration",
    ]
    labels = ["Days", "Consecutive events", "Median duration (days)", "Longest duration (days)"]
    fig.add_trace(
        go.Table(
            header=dict(values=["Summary", "OBS", model_name]),
            cells=dict(
                values=[
                    labels,
                    [
                        _format_display_number(obs_result.metrics.get(name))
                        for name in metric_names
                    ],
                    [
                        _format_display_number(model_result.metrics.get(name))
                        for name in metric_names
                    ],
                ],
            ),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=(
            f"{flow_kind.title()} flow — daily discharge {relation} the raw "
            f"OBS {percentile} threshold"
            f"<br><sub>One fixed OBS threshold is applied to both series; "
            "temporary Layer 2 fills did not define it.</sub>"
        ),
        yaxis_title=f"Discharge ({unit})",
        template=PLOT_TEMPLATE,
        height=650,
    )
    return fig


def _combined_limb_violin_figure(
    obs_rising,
    model_rising,
    obs_recession,
    model_recession,
    unit: str,
    model_name: str,
) -> go.Figure:
    """Compare rising and recession rates in one horizontal violin figure."""
    series = {
        "OBS rising": obs_rising.tables["rising_rates"]["rising_rate"].dropna(),
        f"{model_name} rising": model_rising.tables["rising_rates"][
            "rising_rate"
        ].dropna(),
        "OBS recession": obs_recession.tables["recession_rates"][
            "recession_rate"
        ].dropna(),
        f"{model_name} recession": model_recession.tables["recession_rates"][
            "recession_rate"
        ].dropna(),
    }
    # Display the central 98% so a handful of extremes cannot flatten the
    # useful distribution. Full values remain in calculation tables.
    displayed = {}
    for name, values in series.items():
        if values.empty:
            displayed[name] = values
        else:
            lower, upper = values.quantile([0.01, 0.99])
            displayed[name] = values.loc[values.between(lower, upper)]

    obs_rise_median = obs_rising.metrics.get("median_rising_rate", np.nan)
    model_rise_median = model_rising.metrics.get("median_rising_rate", np.nan)
    obs_recession_median = obs_recession.metrics.get(
        "median_recession_rate", np.nan
    )
    model_recession_median = model_recession.metrics.get(
        "median_recession_rate", np.nan
    )
    rise_difference = model_rise_median - obs_rise_median
    recession_difference = model_recession_median - obs_recession_median
    rise_word = (
        "faster" if rise_difference > 0
        else "slower" if rise_difference < 0
        else "at the same rate"
    )
    recession_word = (
        "faster" if model_recession_median < obs_recession_median
        else "slower" if model_recession_median > obs_recession_median
        else "at the same rate"
    )
    rate_unit = f"{unit}/day"
    if unit.replace(" ", "") in {"mm/day", "mmday"}:
        rate_unit = "mm/day²"

    fig = go.Figure()
    rows = [
        ("OBS recession", "Recession — OBS", OBS_COLOR),
        (f"{model_name} recession", f"Recession — {model_name}", MODEL_COLOR),
        ("OBS rising", "Rising — OBS", OBS_COLOR),
        (f"{model_name} rising", f"Rising — {model_name}", MODEL_COLOR),
    ]
    for source_name, row_name, color in rows:
        fig.add_trace(
            go.Violin(
                x=displayed[source_name],
                y=[row_name] * len(displayed[source_name]),
                customdata=[
                    _format_display_number(value)
                    for value in displayed[source_name]
                ],
                name=row_name,
                orientation="h",
                side="both",
                width=0.8,
                line_color=color,
                fillcolor=color,
                opacity=0.55,
                points=False,
                meanline_visible=False,
                hovertemplate=(
                    f"{row_name}<br>Rate=%{{customdata}} {rate_unit}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    medians = [
        (obs_recession_median, "Recession — OBS", OBS_COLOR),
        (model_recession_median, f"Recession — {model_name}", MODEL_COLOR),
        (obs_rise_median, "Rising — OBS", OBS_COLOR),
        (model_rise_median, f"Rising — {model_name}", MODEL_COLOR),
    ]
    for median, row_name, color in medians:
        fig.add_trace(
            go.Scatter(
                x=[median], y=[row_name], mode="markers",
                marker=dict(color=color, size=11, symbol="diamond",
                            line=dict(color="white", width=1)),
                name=f"{row_name} typical rate",
                hovertemplate=(
                    f"{row_name}<br>Typical rate="
                    f"{_format_display_number(median)} {rate_unit}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=0, line_color="#343A40", line_width=1.5,
        annotation_text="No daily change",
    )
    rise_text = (
        f"Rising: OBS {_format_display_number(obs_rise_median)}, "
        f"{model_name} {_format_display_number(model_rise_median)} {rate_unit}; "
        f"{model_name} rises {rise_word} by "
        f"{_format_display_number(abs(rise_difference))}."
    )
    recession_text = (
        f"Recession: OBS {_format_display_number(obs_recession_median)}, "
        f"{model_name} {_format_display_number(model_recession_median)} {rate_unit}; "
        f"{model_name} recedes {recession_word} by "
        f"{_format_display_number(abs(recession_difference))}."
    )
    fig.update_layout(
        title=(
            "Rising and recession limb rates"
            f"<br><sub>{rise_text} {recession_text}</sub>"
            "<br><sub>Left of zero = recession; right of zero = rising. "
            "Wider violin sections mean that rate occurred more often. "
            "Diamonds mark typical (median) rates. Display limited to the "
            "central 98%; calculations retain all values.</sub>"
        ),
        xaxis_title=(
            f"Daily discharge-rate change ({rate_unit}) — "
            "faster recession ← 0 → faster rise"
        ),
        yaxis_title="",
        violinmode="overlay",
        template=PLOT_TEMPLATE,
        height=600,
        margin=dict(t=150),
    )
    return _apply_layer2_number_format(fig)


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

    gap_rows = []
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.78, 0.22],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.06,
    )
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color="#ADB5BD", width=1), name="Discharge",
        hovertemplate="%{x|%Y-%m-%d}<br>Discharge=%{y:.4g}<extra></extra>",
    ), row=1, col=1)

    finite = series.dropna()
    ymin = float(finite.min()) if not finite.empty else 0.0
    ymax = float(finite.max()) if not finite.empty else 1.0
    missing_result = by_check.get("missing_values", {})
    long_result = by_check.get("long_gaps", {})
    long_intervals = long_result.get("long_gap_intervals", []) or []
    long_keys = {(x["start"], x["end"]) for x in long_intervals}

    # Isolated/short missing intervals: hoverable black vertical lines.
    first_missing = True
    for interval in missing_result.get("internal_intervals", []) or []:
        if (interval["start"], interval["end"]) in long_keys:
            continue
        start, end = pd.Timestamp(interval["start"]), pd.Timestamp(interval["end"])
        for timestamp in pd.date_range(start, end, freq="D"):
            fig.add_trace(go.Scatter(
                x=[timestamp, timestamp], y=[ymin, ymax], mode="lines",
                line=dict(color="rgba(33,37,41,0.65)", width=1),
                name="Missing observation",
                legendgroup="missing",
                showlegend=first_missing,
                hovertemplate=(
                    f"Missing observation<br>{timestamp:%Y-%m-%d}<extra></extra>"
                ),
            ), row=1, col=1)
            first_missing = False
        gap_rows.append({
            "type": "Missing interval", **interval,
        })

    # Long gaps: calm transparent-red spans plus a hover target.
    first_long = True
    for interval in long_intervals:
        start, end = pd.Timestamp(interval["start"]), pd.Timestamp(interval["end"])
        fig.add_vrect(
            x0=start, x1=end + pd.Timedelta(days=1),
            fillcolor="rgba(214,51,108,0.16)", line_width=0,
            row=1, col=1,
        )
        midpoint = start + (end - start) / 2
        fig.add_trace(go.Scatter(
            x=[midpoint], y=[ymax], mode="markers",
            marker=dict(
                size=10, color="rgba(214,51,108,0.35)", symbol="square",
            ),
            name="Long gap", legendgroup="long_gap", showlegend=first_long,
            hovertemplate=(
                f"Long gap<br>{start:%Y-%m-%d} to {end:%Y-%m-%d}"
                f"<br>{interval['missing_count']} missing observation(s)"
                "<extra></extra>"
            ),
        ), row=1, col=1)
        first_long = False
        gap_rows.append({"type": "Long gap", **interval})

    # Descriptor/period checks are drawn only when they have visible evidence.
    zero_result = by_check.get("zero_flow_regime", {})
    zero_dates = series.index[series.round(3).eq(0)]
    if len(zero_dates):
        fig.add_trace(go.Scatter(
            x=zero_dates, y=series.reindex(zero_dates), mode="markers",
            marker=dict(size=7, color=LAYER1_CHECK_COLORS["zero_flow_regime"]),
            name="Zero flow",
        ), row=1, col=1)
    low_result = by_check.get("low_variability", {})
    for i, period in enumerate(low_result.get("low_variability_periods", []) or []):
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=_hex_to_rgba(LAYER1_CHECK_COLORS["low_variability"], 0.20),
            line_width=0, row=1, col=1,
        )
        dates = series.loc[start:end].index
        fig.add_trace(go.Scatter(
            x=dates,
            y=series.reindex(dates).values,
            mode="markers",
            marker=dict(
                size=7,
                color=LAYER1_CHECK_COLORS["low_variability"],
                line=dict(width=0.7, color="#1B5E2F"),
            ),
            name="Low-variability period",
            legendgroup="low_variability",
            showlegend=i == 0,
            hovertemplate=(
                f"Low-variability period<br>{start:%Y-%m-%d} to {end:%Y-%m-%d}"
                f"<br>Duration: {period['calendar_duration_days']} days"
                f"<br>Mean rolling CV: {period.get('mean_rolling_cv', float('nan')):.3f}"
                f"<br>Mean rolling range: {period.get('mean_rolling_range', float('nan')):.3f}"
                "<br>Discharge: %{y:.3f}<extra></extra>"
            ),
        ), row=1, col=1)
        gap_rows.append({"type": "Low variability", **period})

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
                x=idx, y=series.reindex(idx).values, mode="markers",
                marker=dict(size=11, color="#D6336C", opacity=0.95,
                            symbol=symbol, line=dict(width=1, color="#7B2C46")),
                name=f"Spike / dip ({kind})",
                text=hover, hovertemplate="%{text}<extra></extra>",
            ), row=1, col=1)

    if gap_rows:
        table = pd.DataFrame(gap_rows)
        headers = ["Type", "Start", "End", "Missing", "Calendar days"]
        cells = [
            table.get("type", pd.Series(dtype=object)),
            pd.to_datetime(table.get("start", pd.Series(dtype=object))).dt.strftime("%Y-%m-%d"),
            pd.to_datetime(table.get("end", pd.Series(dtype=object))).dt.strftime("%Y-%m-%d"),
            table.get("missing_count", pd.Series(dtype=object)),
            table.get("calendar_duration_days", pd.Series(dtype=object)),
        ]
    else:
        headers = ["Missing-data information"]
        cells = [["No internal missing intervals."]]
    fig.add_trace(go.Table(
        header=dict(values=headers, fill_color="#E9ECEF", align="left"),
        cells=dict(values=cells, align="left"),
    ), row=2, col=1)

    fig.update_layout(
        title=f"Layer 1 combined checks ({series_type})",
        template=PLOT_TEMPLATE, height=690,
        margin=dict(t=70, b=30),
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Discharge (source units)", row=1, col=1)
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _format_p_value(value) -> str:
    """Format human-facing p-values without displaying false numerical zero."""
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


# ==========================================================
# LAYER 1: STEP SHIFT (separate plot)
# ==========================================================

def plot_step_shift(series: pd.Series, step_shift_result: dict, series_type: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.68, 0.32],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.07,
    )
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                              line=dict(color="#4C6EF5", width=1), name="Discharge"),
                  row=1, col=1)

    regime_colors = ["#2F9E44", "#1971C2", "#7048E8", "#0B7285", "#5C940D"]
    for i, regime in enumerate(step_shift_result.get("regime_summary", [])):
        start, end = pd.Timestamp(regime["start"]), pd.Timestamp(regime["end"])
        color = regime_colors[i % len(regime_colors)]
        fig.add_vrect(
            x0=start, x1=end + pd.offsets.MonthEnd(1),
            fillcolor=_hex_to_rgba(color, 0.055), line_width=0,
            row=1, col=1,
        )
        fig.add_trace(go.Scatter(
            x=[start, end], y=[regime["median"]] * 2, mode="lines",
            line=dict(color=color, width=3),
            name=(
                f"Regime {regime.get('continuous_period', 1)}."
                f"{regime.get('regime', i + 1)}"
            ),
            showlegend=False,
            hovertemplate=(
                f"Regime {regime.get('continuous_period', 1)}."
                f"{regime.get('regime', i + 1)}"
                f"<br>{start:%Y-%m-%d} to {end:%Y-%m-%d}"
                f"<br>Calendar months: {regime['calendar_month_count']}"
                f"<br>Valid days: {regime['valid_day_count']}"
                f"<br>Q25: {regime['q25']:.3f}"
                f"<br>Median: {regime['median']:.3f}"
                f"<br>Q75: {regime['q75']:.3f}<extra></extra>"
            ),
        ), row=1, col=1)
        midpoint = start + (end - start) / 2
        fig.add_trace(go.Scatter(
            x=[midpoint], y=[regime["median"]], mode="markers",
            marker=dict(size=16, color=_hex_to_rgba(color, 0.01)),
            showlegend=False,
            hovertemplate=(
                f"Regime {regime.get('continuous_period', 1)}."
                f"{regime.get('regime', i + 1)}"
                f"<br>{start:%Y-%m-%d} to {end:%Y-%m-%d}"
                f"<br>Q25 / median / Q75: {regime['q25']:.3f} / "
                f"{regime['median']:.3f} / {regime['q75']:.3f}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    for boundary in step_shift_result.get("regime_boundaries", []):
        boundary_date = pd.Timestamp(boundary["boundary_timestamp"])
        color = "#8B1E3F" if boundary.get("confirmed") else "rgba(108,117,125,0.45)"
        fig.add_vline(x=boundary_date, line_width=1.2, line_color=color,
                      row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[boundary_date], y=[series.max()],
            mode="markers",
            marker=dict(size=10, color=color, symbol="line-ns-open"),
            showlegend=False,
            hovertemplate=(
                f"Monthly regime-boundary candidate<br>{boundary_date:%Y-%m-%d}"
                f"<br>Median effect: {boundary['standardised_median_effect']:.3f}"
                f"<br>p-value: {_format_p_value(boundary.get('p_value'))}"
                f"<br>Decision: {boundary['decision']}<extra></extra>"
            ),
        ), row=1, col=1)

    boundaries = step_shift_result.get("regime_boundaries", [])
    if boundaries:
        fig.add_trace(go.Table(
            header=dict(values=[
                "Boundary", "Continuous period", "Before median", "After median",
                "Effect", "p-value", "Decision",
            ], fill_color="#E9ECEF", align="left"),
            cells=dict(values=[
                [pd.Timestamp(x["boundary_timestamp"]).strftime("%Y-%m-%d") for x in boundaries],
                [x.get("continuous_period") for x in boundaries],
                [f"{x['before_median']:.3f}" for x in boundaries],
                [f"{x['after_median']:.3f}" for x in boundaries],
                [f"{x['standardised_median_effect']:.3f}" for x in boundaries],
                [_format_p_value(x.get("p_value")) for x in boundaries],
                [x["decision"] for x in boundaries],
            ], align="left"),
        ), row=2, col=1)
    else:
        fig.add_trace(go.Table(
            header=dict(values=["Step-shift result"], fill_color="#E9ECEF"),
            cells=dict(values=[[step_shift_result.get("message", "No boundaries.")]]),
        ), row=2, col=1)

    fig.update_layout(
        title=(
            f"Step shift ({series_type}) — "
            f"{step_shift_result.get('confirmed_boundary_count', 0)} confirmed / "
            f"{step_shift_result.get('consolidated_candidate_count', 0)} consolidated candidates"
        ),
        template=PLOT_TEMPLATE, height=760,
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Discharge (source units)", row=1, col=1)
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


def plot_gradual_drift(series: pd.Series, result: dict, series_type: str) -> go.Figure:
    """Combine segmented trends, segment FDCs, gap context, and a result table."""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy", "colspan": 2}, None],
               [{"type": "xy"}, {"type": "table"}]],
        row_heights=[0.62, 0.38],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
        subplot_titles=("Discharge and segment trends", "Segment flow-duration curves", "Trend results"),
    )
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color="#A5B4FC", width=0.7), opacity=0.55,
        name="Daily discharge",
    ), row=1, col=1)
    colors = ["#2F9E44", "#1971C2", "#7048E8", "#E8590C", "#0B7285"]
    completed = []
    for i, segment in enumerate(result.get("drift_segments", []) or []):
        color = colors[i % len(colors)]
        dates = pd.to_datetime(segment.get("monthly_dates", []))
        values = segment.get("monthly_values", [])
        if len(dates):
            fig.add_trace(go.Scatter(
                x=dates, y=values, mode="markers",
                marker=dict(size=4, color=color, opacity=0.45),
                name=f"Monthly medians — continuous record period {i + 1}",
                visible="legendonly",
            ), row=1, col=1)
        sen_dates = pd.to_datetime(segment.get("sen_line_dates", []))
        if len(sen_dates):
            fig.add_trace(go.Scatter(
                x=sen_dates, y=segment["sen_line_values"], mode="lines",
                line=dict(color=color, width=4),
                name=f"Straight Sen trend — continuous record period {i + 1}",
                hovertemplate=(
                    f"Continuous record period {i + 1}"
                    f"<br>Annual Sen slope: {segment['sen_slope_per_year']:.3f}"
                    f"<br>Estimated total change: {segment['estimated_total_change']:.3f}"
                    f"<br>Relative total change: "
                    f"{segment.get('estimated_relative_change_percent', float('nan')):.3f}%"
                    f"<br>{segment.get('interpretation', '')}<extra></extra>"
                ),
            ), row=1, col=1)
        start, end = pd.Timestamp(segment["start"]), pd.Timestamp(segment["end"])
        segment_flow = series.loc[start:end].dropna()
        if not segment_flow.empty:
            exceedance, flow = _flow_duration_curve(segment_flow)
            fig.add_trace(go.Scatter(
                x=exceedance, y=flow, mode="lines",
                line=dict(color=color, width=2),
                name=f"FDC — continuous record period {i + 1}",
                legendgroup=f"segment-{i}",
            ), row=2, col=1)
        if segment.get("execution_status") == "completed":
            completed.append(segment)
    for gap in result.get("unresolved_gaps", []) or []:
        fig.add_vrect(
            x0=pd.Timestamp(gap["start"]), x1=pd.Timestamp(gap["end"]),
            fillcolor="rgba(214,51,108,0.16)", line_width=0,
            row=1, col=1,
        )
    segments = result.get("drift_segments", []) or []
    if segments:
        fig.add_trace(go.Table(
            header=dict(values=[
                "Record period", "Dates", "Years", "Interpretation",
                "Annual Sen slope", "Total change", "Reference median",
                "Relative total change (%)", "p-value",
            ], fill_color="#E9ECEF", align="left"),
            cells=dict(values=[
                [f"Continuous period {i}" for i in range(1, len(segments) + 1)],
                [
                    f"{pd.Timestamp(x['start']):%Y-%m-%d} – "
                    f"{pd.Timestamp(x['end']):%Y-%m-%d}"
                    for x in segments
                ],
                [f"{x.get('record_years', 0):.3f}" for x in segments],
                [x.get("interpretation", "not calculated") for x in segments],
                [
                    f"{x['sen_slope_per_year']:.3f}" if x.get("sen_slope_per_year") is not None else "n/a"
                    for x in segments
                ],
                [
                    f"{x['estimated_total_change']:.3f}"
                    if x.get("estimated_total_change") is not None else "n/a"
                    for x in segments
                ],
                [
                    f"{x['reference_flow']:.3f}"
                    if x.get("reference_flow") is not None else "n/a"
                    for x in segments
                ],
                [
                    f"{x['estimated_relative_change_percent']:.3f}"
                    if x.get("estimated_relative_change_percent") is not None else "n/a"
                    for x in segments
                ],
                [_format_p_value(x.get("p_value")) for x in segments],
            ], align="left"),
        ), row=2, col=2)
    else:
        fig.add_trace(go.Table(
            header=dict(values=["Gradual-drift result"], fill_color="#E9ECEF"),
            cells=dict(values=[[result.get("message", "Not calculated")]]),
        ), row=2, col=2)
    fig.update_layout(
        title=f"Gradual drift ({series_type}) — segmented at unresolved gaps",
        template=PLOT_TEMPLATE, height=850,
    )
    fig.update_yaxes(title_text="Discharge (source units)", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Exceedance (%)", row=2, col=1)
    fig.update_yaxes(title_text="Discharge (log scale)", type="log", row=2, col=1)
    return fig


def plot_gradual_drift_trend(series: pd.Series, gradual_drift_result: dict, series_type: str) -> go.Figure:
    """Compatibility wrapper returning the new combined gradual-drift figure."""
    return plot_gradual_drift(series, gradual_drift_result, series_type)


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
    # Remove obsolete split drift products after migration to one combined
    # gradual-drift figure. Targets are explicit station-output files.
    for series_label in ["obs", model_name]:
        for legacy_name in [
            f"{series_label}_gradual_drift_trend.html",
            f"{series_label}_gradual_drift_fdc.html",
        ]:
            legacy_path = station_dir / legacy_name
            if legacy_path.is_file():
                legacy_path.unlink()
    # Detail tables are conditional. Remove previous versions before writing
    # this run so a now-empty finding cannot leave a stale CSV behind.
    for table_name in layer1_diagnostics.get("detail_tables", {}):
        stale_table = station_dir / f"layer1_{table_name}.csv"
        if stale_table.is_file():
            stale_table.unlink()

    raw_results = layer1_diagnostics["raw_results"]

    for series_type, series in [("obs", obs_series)] + ([(model_name, sim_series)] if sim_series is not None else []):
        by_check = {r["check"]: r for r in raw_results[series_type]}

        _save(plot_layer1_combined(series, raw_results[series_type], series_type),
              station_dir / f"{series_type}_combined_checks.html", show=show)

        if "step_shift" in by_check:
            _save(plot_step_shift(series, by_check["step_shift"], series_type),
                  station_dir / f"{series_type}_step_shift.html", show=show)

        if "gradual_drift" in by_check:
            _save(plot_gradual_drift(series, by_check["gradual_drift"], series_type),
                  station_dir / f"{series_type}_gradual_drift.html", show=show)

    layer1_diagnostics["eda_summary"].to_csv(station_dir / "layer1_eda_summary.csv", index=False)
    layer1_diagnostics["summary_all"].to_csv(station_dir / "layer1_summary_all.csv", index=False)
    layer1_diagnostics["summary_flagged"].to_csv(station_dir / "layer1_summary_flagged.csv", index=False)
    for table_name, table in layer1_diagnostics.get("detail_tables", {}).items():
        if not table.empty:
            table.round(3).to_csv(
                station_dir / f"layer1_{table_name}.csv",
                index=False,
            )

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
    ml_series: Optional[pd.Series],
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
    obs_clean = layer2_diagnostics.get("obs_analysis", obs_series).dropna()
    unit = layer2_diagnostics.get("threshold_provenance", {}).get(
        "unit", "source units"
    )

    # OBS-only is a supported Layer 2 mode. Comparison plots have no honest
    # model trace to draw, so save an OBS overview plus the 13-signature table
    # and imputation audit instead of fabricating or duplicating a model.
    if model_results is None:
        overview = go.Figure()
        overview.add_trace(
            go.Scatter(
                x=obs_clean.index,
                y=obs_clean.values,
                mode="lines",
                name="OBS temporary analysis copy",
                line=dict(color=OBS_COLOR, width=0.8),
            )
        )
        imputation_log = layer2_diagnostics.get("imputation_log", pd.DataFrame())
        obs_fills = (
            imputation_log[imputation_log["series"] == "obs"]
            if not imputation_log.empty
            else pd.DataFrame()
        )
        if not obs_fills.empty:
            overview.add_trace(
                go.Scatter(
                    x=obs_fills["timestamp"],
                    y=obs_fills["temporary_value"],
                    mode="markers",
                    name="Temporarily imputed for Layer 2",
                    marker=dict(color="#C92A2A", size=5, symbol="x"),
                    customdata=obs_fills[["method"]],
                    hovertemplate=(
                        "Date=%{x}<br>Temporary value=%{y:.3f}"
                        "<br>Method=%{customdata[0]}<extra></extra>"
                    ),
                )
            )
        overview.update_layout(
            title="Layer 2 OBS analysis copy (temporary fills explicitly marked)",
            xaxis_title="Date",
            yaxis_title="Discharge",
            template=PLOT_TEMPLATE,
            height=480,
        )
        _save(overview, station_dir / "obs_analysis_overview.html", show=show)
        _save_csv_resilient(
            layer2_diagnostics["signature_comparison"],
            station_dir / "layer2_signature_diagnostics.csv",
        )
        _save_csv_resilient(
            imputation_log,
            station_dir / "layer2_temporary_imputation_log.csv",
        )
        _save_csv_resilient(
            layer2_diagnostics["percentile_diagnostics"],
            station_dir / "layer2_percentile_diagnostics.csv",
        )
        return station_dir

    model_clean = layer2_diagnostics.get("model_analysis", ml_series).dropna()

    figures = {}

    # --- data-quality context (new, from the notebook audit) ---
    figures["missing_values"] = plot_missing_values(
        layer2_diagnostics.get("obs_aligned", obs_series)
    )
    figures["temporary_fill_highlight"] = plot_temporary_fill_highlight(
        layer2_diagnostics.get("obs_aligned", obs_series)
    )

    # --- flow magnitude: FDC ---
    obs_x, obs_y = _flow_duration_curve(obs_clean)
    model_x, model_y = _flow_duration_curve(model_clean)
    figures["flow_magnitude"] = _obs_model_lines(
        obs_x, obs_y, model_x, model_y, title="Flow magnitude -- flow duration curve",
        x_label="Exceedance probability: percent of days this flow was equalled or exceeded",
        y_label=f"Discharge ({unit}, log scale)",
        model_name=model_name, y_log=True,
    )
    figures["flow_magnitude"].update_layout(
        title=(
            "Flow-duration comparison — distribution of high, typical and low discharge"
            "<br><sub>Left: rare high flows | Centre: typical flows | "
            "Right: frequently equalled low flows. Vertical separation shows magnitude bias.</sub>"
        )
    )
    figures["flow_magnitude"].add_vrect(
        x0=0, x1=10, fillcolor="rgba(214, 51, 108, 0.07)", line_width=0,
        annotation_text="High flows", annotation_position="top left",
    )
    figures["flow_magnitude"].add_vrect(
        x0=10, x1=90, fillcolor="rgba(25, 113, 194, 0.04)", line_width=0,
        annotation_text="Typical flows", annotation_position="top left",
    )
    figures["flow_magnitude"].add_vrect(
        x0=90, x1=100, fillcolor="rgba(47, 158, 68, 0.07)", line_width=0,
        annotation_text="Low flows", annotation_position="top left",
    )

    # --- low / high flow ---
    figures["low_flow"] = _threshold_comparison_figure(
        obs_clean, model_clean, obs_results["low_flow"],
        model_results["low_flow"], "low",
        obs_results["low_flow"].metrics["low_flow_threshold"],
        unit, model_name,
    )
    figures["high_flow"] = _threshold_comparison_figure(
        obs_clean, model_clean, obs_results["high_flow"],
        model_results["high_flow"], "high",
        obs_results["high_flow"].metrics["high_flow_threshold"],
        unit, model_name,
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
    obs_rbi = obs_results["flashiness"].metrics["whole_record_flashiness_index"]
    model_rbi = model_results["flashiness"].metrics["whole_record_flashiness_index"]
    rbi_difference = model_rbi - obs_rbi
    rbi_percent = (rbi_difference / obs_rbi * 100) if obs_rbi else np.nan
    rbi_direction = "more" if rbi_difference > 0 else "less" if rbi_difference < 0 else "the same"
    figures["flashiness"] = _obs_model_lines(
        obs_fl["year"], obs_fl["flashiness_index"],
        model_fl["year"], model_fl["flashiness_index"],
        (
            "Annual Richards–Baker flashiness index"
            f"<br><sub>Whole record: OBS={obs_rbi:.3f}, {model_name}={model_rbi:.3f}. "
            f"{model_name} represents {abs(rbi_percent):.3f}% {rbi_direction} "
            "day-to-day flashiness. Higher values mean larger daily changes "
            "relative to total flow.</sub>"
        ),
        "Year", "Richards–Baker flashiness index", model_name=model_name,
    )

    obs_acf, model_acf = obs_results["autocorrelation"].tables["autocorrelation_function"], model_results["autocorrelation"].tables["autocorrelation_function"]
    acf_fig = _obs_model_lines(obs_acf["lag"], obs_acf["autocorrelation"], model_acf["lag"], model_acf["autocorrelation"],
                                "Autocorrelation function", "Lag (days)", "Autocorrelation", model_name=model_name)
    acf_fig.add_hline(y=1 / np.e, line_dash="dot", line_color="grey", annotation_text="1/e decorrelation threshold")
    obs_decay = obs_results["autocorrelation"].metrics.get("decorrelation_lag")
    model_decay = model_results["autocorrelation"].metrics.get("decorrelation_lag")
    acf_fig.update_layout(
        title=(
            "Discharge memory — autocorrelation by lag"
            f"<br><sub>First lag at or below 1/e: OBS={obs_decay} days, "
            f"{model_name}={model_decay} days. This reference estimates how "
            "long discharge retains temporal memory; it is not a warning threshold.</sub>"
        )
    )
    figures["autocorrelation"] = acf_fig

    # --- rising and recession limbs: one zero-centred horizontal comparison ---
    figures["limb_rates"] = _combined_limb_violin_figure(
        obs_results["rising_limb"],
        model_results["rising_limb"],
        obs_results["recession_limb"],
        model_results["recession_limb"],
        unit,
        model_name,
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

    # High-flow threshold events are already communicated in the redesigned
    # high-flow figure. Retain their calculated table but do not duplicate the
    # same information in another ambiguous hydrograph.

    for name, fig in figures.items():
        _apply_layer2_number_format(fig)
        _save(fig, station_dir / f"{name}.html", show=show)

    _save_csv_resilient(layer2_diagnostics["compact_comparison"], station_dir / "layer2_compact_comparison.csv")
    _save_csv_resilient(layer2_diagnostics["signature_comparison"], station_dir / "layer2_signature_diagnostics.csv")
    _save_csv_resilient(layer2_diagnostics["imputation_log"], station_dir / "layer2_temporary_imputation_log.csv")
    _save_csv_resilient(layer2_diagnostics["full_comparison"], station_dir / "layer2_summary_all.csv")
    _save_csv_resilient(layer2_diagnostics["percentile_diagnostics"], station_dir / "layer2_percentile_diagnostics.csv")

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
    anything, and `candidate_series` already loaded through
    ``trihydra.io``. This file never loads discharge data itself.

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
        _apply_layer2_number_format(fig)
        _save(fig, station_dir / "context_comparison.html", show=show)

    _save_csv_resilient(
        layer3_diagnostics["context_summary"],
        station_dir / "layer3_context_summary.csv",
    )
    _save_csv_resilient(
        layer3_diagnostics["comparison_table"],
        station_dir / "layer3_comparison_table.csv",
    )
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
