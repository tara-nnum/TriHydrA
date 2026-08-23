
"""Constants and HTML helpers shared by diagnostic plots."""

from __future__ import annotations

from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "outputs"

PLOT_TEMPLATE = "plotly_white"

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
