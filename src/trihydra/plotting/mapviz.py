"""
mapviz.py

Gauge-network map visualisations: a world map of the whole network,
and a zoomed local map per target station showing its context
candidates. Kept separate from visualisation.py deliberately -- these
use folium (Leaflet.js web maps), a genuinely different rendering
technology from the Plotly figures everywhere else in this project,
not because the content is different in kind.

Like visualisation.py, every function here takes already-computed
results (gauge network metadata, a candidate_result dict from
gauge_network.find_context_candidates) rather than finding candidates
itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import folium
from folium.plugins import MarkerCluster

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRIHYDRA_DIR = Path(__file__).resolve().parent.parent
IO_OUTPUT_ROOT = TRIHYDRA_DIR / "io" / "output"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _station_popup(row) -> str:
    area_text = (
        f"{row['area_km2']:,.1f} km\u00b2" if row.get("area_km2") == row.get("area_km2") else "missing"
    )
    return (
        f"<b>{row.get('gauge_id', '')}</b><br>"
        f"Station: {row.get('StationName', '')}<br>"
        f"Catchment: {row.get('Catchment', '')}<br>"
        f"River: {row.get('River', '')}<br>"
        f"Area: {area_text}"
    )


def make_world_map(meta, target_ids: list[str]) -> folium.Map:
    """
    Every gauge in the network, clustered (883+ individual markers
    would make an unreadable, slow-loading map otherwise), with target
    stations shown larger and in red.
    """
    world_map = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron", control_scale=True)
    cluster = MarkerCluster(name="All outlet stations").add_to(world_map)

    for _, row in meta.iterrows():
        is_target = row["gauge_id"] in target_ids
        folium.CircleMarker(
            location=[row["StationLat"], row["StationLon"]],
            radius=7 if is_target else 2,
            color="red" if is_target else "grey",
            fill=True,
            fill_opacity=1.0 if is_target else 0.45,
            popup=folium.Popup(_station_popup(row), max_width=350),
            tooltip=row["gauge_id"],
        ).add_to(cluster)

    folium.LayerControl().add_to(world_map)
    return world_map


def make_target_map(meta, candidate_result: dict) -> folium.Map:
    """
    A zoomed map centred on one target: the adaptive search radius as
    a circle, every other station within that radius in grey, and the
    selected context candidates highlighted -- blue for same-river,
    orange for same-catchment-only, with the target itself marked as
    a red star on top.
    """
    target_id = candidate_result["target_id"]
    target = candidate_result["target"]
    radius = candidate_result["radius_km"]
    candidates = candidate_result["candidates"]
    selected_ids = set(candidates["gauge_id"])

    local_map = folium.Map(
        location=[target["StationLat"], target["StationLon"]],
        zoom_start=6, tiles="CartoDB positron", control_scale=True,
    )

    folium.Circle(
        location=[target["StationLat"], target["StationLon"]],
        radius=radius * 1000, color="red", weight=1.5, fill=False,
        tooltip=f"Search radius: {radius:.1f} km",
    ).add_to(local_map)

    from src.trihydra.layer3.gauge_network import haversine_km

    all_other = meta.loc[~meta["gauge_id"].eq(target_id)].copy()
    all_other["distance_km"] = haversine_km(
        target["StationLat"], target["StationLon"],
        all_other["StationLat"].to_numpy(), all_other["StationLon"].to_numpy(),
    )
    nearby = all_other.loc[all_other["distance_km"].le(radius)]

    for _, row in nearby.loc[~nearby["gauge_id"].isin(selected_ids)].iterrows():
        folium.CircleMarker(
            location=[row["StationLat"], row["StationLon"]], radius=3, color="grey",
            fill=True, fill_opacity=0.4,
            popup=folium.Popup(_station_popup(row) + f"<br>Distance: {row['distance_km']:.1f} km", max_width=350),
            tooltip=row["gauge_id"],
        ).add_to(local_map)

    for _, row in candidates.iterrows():
        marker_color = "blue" if row["same_river"] else "orange"
        folium.CircleMarker(
            location=[row["StationLat"], row["StationLon"]], radius=7, color=marker_color,
            fill=True, fill_opacity=0.9,
            popup=folium.Popup(
                _station_popup(row) + f"<br>Distance: {row['distance_km']:.1f} km<br>Same river: {row['same_river']}",
                max_width=350,
            ),
            tooltip=f"{row['gauge_id']} | {row['distance_km']:.1f} km",
        ).add_to(local_map)

    folium.Marker(
        location=[target["StationLat"], target["StationLon"]],
        icon=folium.Icon(color="red", icon="star"),
        popup=folium.Popup(_station_popup(target), max_width=350),
        tooltip=f"TARGET: {target_id}",
    ).add_to(local_map)

    title_html = (
        f"<h4 style='position: fixed; top: 10px; left: 50px; z-index:9999; "
        f"background:white; padding:8px;'>{target_id}: context {candidate_result['status']}</h4>"
    )
    local_map.get_root().html.add_child(folium.Element(title_html))

    return local_map


def generate_layer3_maps(
    meta,
    candidate_result: dict,
    station_id: str = "station",
    output_root: Optional[Path] = None,
    show: bool = False,
    include_world_map: bool = False,
) -> dict:
    """
    Build the Layer 3 maps and save them.

    The per-target local map always goes to
    <IO_OUTPUT_ROOT>/<station_id>/layer3/context_map.html.

    include_world_map=False by default: the world map covers the whole
    network, not just this one station, so it's not regenerated on
    every single run_layer3() call unless asked for. When it is, it
    saves to a shared location (<IO_OUTPUT_ROOT>/_gauge_network_map.html),
    not inside any one station's folder.

    show=True displays each map inline (e.g. in a notebook) via
    IPython's display(), the way folium maps normally render.
    """
    output_root = Path(output_root) if output_root is not None else IO_OUTPUT_ROOT
    station_dir = output_root / station_id / "layer3"
    station_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    target_map = make_target_map(meta, candidate_result)
    target_map_path = station_dir / "context_map.html"
    target_map.save(str(target_map_path))
    saved_paths["context_map"] = target_map_path
    if show:
        from IPython.display import display
        display(target_map)

    if include_world_map:
        world_map = make_world_map(meta, [candidate_result["target_id"]])
        world_map_path = output_root / "_gauge_network_map.html"
        world_map.save(str(world_map_path))
        saved_paths["world_map"] = world_map_path
        if show:
            from IPython.display import display
            display(world_map)

    return saved_paths


if __name__ == "__main__":
    print(
        "This module is meant to be imported. Call "
        "generate_layer3_maps(meta, candidate_result, station_id=...) "
        "using the gauge-network metadata (gauge_network.load_gauge_network) "
        "and a candidate_result (gauge_network.find_context_candidates)."
    )
