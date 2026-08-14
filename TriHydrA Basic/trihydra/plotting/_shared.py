
"""Internal plotting implementation retained during the domain split.

Every function below consumes the OUTPUT of diagnostics.py (already-
computed check/signature results) rather than calling a check itself.
This is deliberate: if this file re-ran checks independently, a rename
or signature change in the check layer could make this file crash
without diagnostics.py noticing, or the two files could silently
disagree about what a check found. Feeding this file diagnostics.py's
output means there is exactly one place checks are ever executed.

Public plotting imports belong to the layer-owned visualisation modules or the
``trihydra.plotting.visualisation`` facade. This module is private.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "outputs"

from trihydra.settings.defaults import DISPLAY_DECIMALS


# ==========================================================
# SHARED SETTINGS
# ==========================================================

PLOT_TEMPLATE = "plotly_white"
OBS_COLOR = "#1971C2"
MODEL_COLOR = "#E8590C"
OBS_SHADE = "rgba(25, 113, 194, 0.15)"
MODEL_SHADE = "rgba(232, 89, 12, 0.15)"
def _format_display_number(value) -> str:
    """Format finite display values to the configured precision."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(numeric) else f"{numeric:.{DISPLAY_DECIMALS}f}"


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
    "low_variability": "Non-zero plateau / flatline",
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


def _save_dashboard(
    figures: list[go.Figure], path: Path, *, title: str, show: bool = False
) -> None:
    """Save several conditional Plotly panels in one self-contained HTML."""
    if show:
        for figure in figures:
            figure.show()
    parts = []
    for index, figure in enumerate(figures):
        parts.append(pio.to_html(
            figure, full_html=False,
            include_plotlyjs="cdn" if index == 0 else False,
        ))
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>" + "\n".join(parts) + "</body></html>",
        encoding="utf-8",
    )


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
