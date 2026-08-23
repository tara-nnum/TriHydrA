"""Public plotting API for completed in-memory results."""

from __future__ import annotations

from pathlib import Path

from trihydra.comparison.visualisation import generate_comparison_visuals
from trihydra.layer1.visualisation import generate_layer1_visuals
from trihydra.layer2.visualisation import generate_layer2_visuals
from trihydra.layer3.visualisation import write_layer3_overview
from trihydra.result import TriHydrANetworkResult, TriHydrAResult


def plot_results(
    result: TriHydrAResult | TriHydrANetworkResult,
    output_directory: str | Path = "outputs",
    *,
    show: bool = False,
    layer3_plot_mode: str = "recommended",
) -> dict[str, Path]:
    """Write available HTML diagnostics without recalculating assessments."""
    if isinstance(result, TriHydrANetworkResult):
        written: dict[str, Path] = {}
        for station_id, station_result in result.station_results.items():
            for name, path in plot_results(
                station_result, output_directory, show=show,
                layer3_plot_mode=layer3_plot_mode,
            ).items():
                written[f"{station_id}:{name}"] = path
            layer3_result = result.layer3_run.station_results.get(station_id)
            if layer3_result is not None:
                path = Path(output_directory) / station_id / "layer3.html"
                saved = write_layer3_overview(
                    layer3_result, result.layer3_run.evidence_cache,
                    result.series_by_station, path, plot_mode=layer3_plot_mode,
                )
                if saved is not None:
                    written[f"{station_id}:layer3"] = saved
        return written
    if not isinstance(result, TriHydrAResult):
        raise TypeError("result must be a TriHydrAResult or TriHydrANetworkResult.")
    station_directory = Path(output_directory) / result.station_id
    station_directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if result.layer1 is not None:
        generate_layer1_visuals(
            result.station.obs, None, result.layer1,
            station_id=result.station_id, output_root=station_directory,
            show=show, source_id=result.station.series1_name, write_tables=False,
            flat_filename="layer1.html",
        )
        written["layer1"] = station_directory / "layer1.html"
    if result.layer2 is not None:
        generate_layer2_visuals(
            result.station.obs, result.layer2,
            station_id=result.station_id, output_root=station_directory,
            show=show, source_id=result.station.series1_name, write_tables=False,
            flat_filename="layer2.html",
        )
        written["layer2"] = station_directory / "layer2.html"

    comparison = result.comparison
    if comparison is not None and result.station.ml is not None:
        candidate_name = str(comparison["candidate_name"])
        generate_layer1_visuals(
            result.station.ml, None, comparison["candidate_layer1"],
            station_id=result.station_id, output_root=station_directory,
            show=show, source_id=candidate_name, write_tables=False,
            flat_filename="candidate_layer1.html",
        )
        generate_layer2_visuals(
            result.station.ml, comparison["candidate_native_layer2"],
            station_id=result.station_id, output_root=station_directory,
            show=show, source_id=candidate_name, write_tables=False,
            flat_filename="candidate_layer2.html",
        )
        written["candidate_layer1"] = station_directory / "candidate_layer1.html"
        written["candidate_layer2"] = station_directory / "candidate_layer2.html"
        written["comparison"] = generate_comparison_visuals(
            comparison, station_directory, show=show,
            flat_filename="comparison.html",
        )
    return written


__all__ = ["plot_results"]
