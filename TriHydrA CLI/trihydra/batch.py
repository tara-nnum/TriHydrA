"""TOML-driven batch orchestration for the ECMWF TriHydrA workflow."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from trihydra.io import StationData, iter_netcdf_stations, load_model_result_pickle
from trihydra.network import attach_layer3_to_results
from trihydra.outputs import save_results, write_netcdf_results
from trihydra.pipeline import run_trihydra
from trihydra.plotting import plot_results
from trihydra.layer3.visualisation import write_layer3_overview
from trihydra.result import TriHydrANetworkResult, TriHydrAResult
from trihydra.settings import TriHydrAConfig, load_toml_config


def _logger(output_directory: Path, write_log: bool) -> logging.Logger:
    logger = logging.getLogger("trihydra.batch")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if write_log:
        output_directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            output_directory / "trihydra_run.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _attach_aifl_simulation(
    observation: StationData, config: TriHydrAConfig
) -> StationData:
    """Attach only the trusted AIFL simulation to the CARAVAN observation."""
    source = config.simulation.path
    if source is None:
        raise ValueError("simulation.path is required for an enabled simulation.")
    pickle_path = source / f"{observation.station_id}_results.p" if source.is_dir() else source
    supplied = load_model_result_pickle(
        pickle_path,
        station_id=observation.station_id,
        unit=config.simulation.units,
        time_step=config.simulation.time_step,
        observation_variable=config.simulation.observation_variable,
        simulation_variable=config.simulation.simulation_variable,
    )
    metadata = dict(observation.metadata)
    metadata["model_metadata"] = supplied.metadata
    return StationData(
        station_id=observation.station_id,
        obs=observation.obs,
        unit=observation.unit,
        obs_provenance=observation.obs_provenance,
        ml=supplied.ml,
        ml_provenance=supplied.ml_provenance,
        metadata=metadata,
    )


def _review_required(result: TriHydrAResult) -> bool:
    """Return whether any completed station-series assessment requests review."""
    columns = [name for name in result.summary.columns if name.endswith("_class")]
    values = {
        str(value).strip().casefold()
        for name in columns
        for value in result.summary[name].dropna().tolist()
    }
    return bool(values.intersection({"needs review", "review"}))


def _resolve_html_mode(config: TriHydrAConfig) -> str:
    """Resolve the optional terminal prompt without blocking unattended jobs."""
    mode = config.output.html_mode
    if mode != "ask":
        return mode
    if not sys.stdin.isatty():
        return config.output.non_interactive_html_mode
    prompt = (
        "\nHTML diagnostics: [1] all stations, [2] review-only, [3] none "
        "(default 2): "
    )
    try:
        choice = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return config.output.non_interactive_html_mode
    return {"1": "all", "2": "review_only", "3": "none", "": "review_only"}.get(
        choice, "review_only"
    )


def _selected_for_plotting(
    results: Iterable[TriHydrAResult], mode: str
) -> list[TriHydrAResult]:
    if mode == "all":
        return list(results)
    if mode == "review_only":
        return [result for result in results if _review_required(result)]
    return []


def run_from_config(path: str | Path = "trihydra.toml") -> pd.DataFrame:
    """Run selected stations from one validated TOML configuration."""
    started = time.perf_counter()
    public = load_toml_config(path)
    base_directory = Path(path).expanduser().resolve().parent
    runtime = public.runtime_overrides(base_directory)
    output_directory = public.output.directory
    output_directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(output_directory, public.output.write_log)
    selection = public.station_selection()
    selected_label = "all" if selection == "all" else str(len(selection))
    logger.info("TriHydrA started | selected stations: %s", selected_label)

    completed: list[TriHydrAResult] = []
    rows: list[dict[str, object]] = []
    layer3_run = None
    iterator = iter_netcdf_stations(
        public.observation.path,
        selection,
        unit=public.observation.units,
        variable=public.observation.variable,
        time_coordinate=public.observation.time_coordinate,
        station_coordinate=public.observation.station_coordinate,
        # The batch manifest must retain empty requested stations so they can
        # be reported as skipped rather than silently disappearing.
        include_empty=True,
    )
    for number, observation in enumerate(iterator, start=1):
        station_started = time.perf_counter()
        station_id = observation.station_id
        logger.info("[%d] processing station=%s", number, station_id)
        row: dict[str, object] = {"station_id": station_id}
        if len(observation.obs) == 0 or not observation.obs.notna().any():
            elapsed = time.perf_counter() - station_started
            row.update(
                status="skipped",
                simulation_status="not_attempted",
                review_required=False,
                elapsed_seconds=elapsed,
                error_type="NoValidObservations",
                error_message="No valid dated streamflow observations are available.",
            )
            rows.append(row)
            logger.warning(
                "skipped station=%s | no valid dated streamflow observations | %.2f s",
                station_id,
                elapsed,
            )
            continue
        try:
            station = observation
            row["simulation_status"] = "disabled"
            if public.simulation.enabled:
                try:
                    station = _attach_aifl_simulation(observation, public)
                    row["simulation_status"] = "available"
                except FileNotFoundError:
                    # A Layer 3 peer may have observations without a matching
                    # model result. Keep it as observation-only context rather
                    # than discarding the gauge from the network assessment.
                    row["simulation_status"] = "not_available"
                    logger.warning(
                        "simulation unavailable station=%s | continuing observation-only",
                        station_id,
                    )
            result = run_trihydra(
                station, config=runtime, model_name=public.simulation.model_name
            )
            completed.append(result)
            review = _review_required(result)
            elapsed = time.perf_counter() - station_started
            row.update(status="completed", review_required=review, elapsed_seconds=elapsed)
            logger.info(
                "completed station=%s | review=%s | %.2f s",
                station_id, "yes" if review else "no", elapsed,
            )
        except Exception as error:
            elapsed = time.perf_counter() - station_started
            row.update(
                status="failed",
                review_required=False,
                elapsed_seconds=elapsed,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            logger.exception("failed station=%s | %.2f s", station_id, elapsed)
            if not public.run.continue_on_station_error:
                rows.append(row)
                raise
        rows.append(row)

    if public.layers.layer3:
        if len(completed) < 2:
            logger.warning(
                "Layer 3 not assessed: at least two stations must complete Layers 1/2."
            )
        else:
            context_path = runtime["layer3"]["metadata"]["context_path"]
            logger.info("Layer 3 context assessment started | stations=%d", len(completed))
            layer3_run = attach_layer3_to_results(
                {result.station_id: result for result in completed},
                context_path=context_path,
                target_station_ids=[result.station_id for result in completed],
                config=runtime,
            )
            layer3_status = {
                station_id: (
                    "assessed"
                    if station_result.summary.combined_classification != "Not assessed"
                    else "not_assessed"
                )
                for station_id, station_result in layer3_run.station_results.items()
            }
            for row in rows:
                row["layer3_status"] = layer3_status.get(
                    str(row["station_id"]), "not_assessed"
                )
            assessed = sum(value == "assessed" for value in layer3_status.values())
            logger.info(
                "Layer 3 context assessment finished | assessed=%d not_assessed=%d",
                assessed, len(completed) - assessed,
            )

    if public.output.write_text:
        if completed:
            network_result = TriHydrANetworkResult(
                station_results={result.station_id: result for result in completed},
                layer3_run=layer3_run,
                summary=pd.concat(
                    [result.summary for result in completed],
                    ignore_index=True,
                    sort=False,
                ),
                series_by_station={
                    result.station_id: result.station.obs for result in completed
                },
                configuration_used=public.model_dump(mode="json"),
            )
            save_results(network_result, output_directory)

    html_mode = _resolve_html_mode(public)
    plot_targets = _selected_for_plotting(completed, html_mode)
    logger.info("HTML mode=%s | plotting %d station(s)", html_mode, len(plot_targets))
    for result in plot_targets:
        plot_results(result, output_directory, show=public.output.show_figures)
        if layer3_run is not None:
            layer3_result = layer3_run.station_results.get(result.station_id)
            if layer3_result is not None:
                write_layer3_overview(
                    layer3_result,
                    layer3_run.evidence_cache,
                    {item.station_id: item.station.obs for item in completed},
                    output_directory / result.station_id / "layer3.html",
                    plot_mode=runtime["layer3"]["plotting"]["mode"],
                )

    manifest = pd.DataFrame(rows)
    if public.output.write_netcdf:
        netcdf_path = write_netcdf_results(
            completed,
            manifest,
            output_directory / "trihydra_results.nc",
            configuration=public.model_dump(mode="json"),
        )
        logger.info("NetCDF summary written: %s", netcdf_path)
    elapsed = time.perf_counter() - started
    completed_count = int((manifest.get("status") == "completed").sum()) if not manifest.empty else 0
    failed_count = int((manifest.get("status") == "failed").sum()) if not manifest.empty else 0
    logger.info(
        "TriHydrA finished | completed=%d failed=%d | elapsed %.2f s",
        completed_count, failed_count, elapsed,
    )
    return manifest


__all__ = ["run_from_config"]
