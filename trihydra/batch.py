"""TOML-driven batch orchestration for file-based TriHydrA workflows."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from trihydra.io import load_stations
from trihydra.network import run_trihydra_batch
from trihydra.outputs import save_results, write_netcdf_results
from trihydra.plotting import plot_results
from trihydra.layer3.visualisation import write_layer3_overview
from trihydra.reporting import station_requires_review
from trihydra.result import TriHydrABatchResult, TriHydrAResult
from trihydra.settings import (
    TriHydrAConfig,
    build_runtime_config,
    load_toml_config,
    resolve_station_selection,
)


def _logger(output_directory: Path, write_log: bool) -> logging.Logger:
    logger = logging.getLogger("trihydra.batch")
    # Repeated notebook runs must release the previous log file on Windows.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
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


def _resolve_html_mode(config: TriHydrAConfig) -> str:
    """Resolve the optional terminal prompt without blocking unattended jobs."""
    mode = config.output.html_mode
    if mode != "ask":
        return mode
    if not sys.stdin.isatty():
        return config.output.non_interactive_html_mode
    prompt = (
        "\nHTML diagnostics: [1] all, [2] minor concerns + needs review, "
        "[3] needs review only, [4] none (default 3): "
    )
    try:
        choice = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return config.output.non_interactive_html_mode
    return {
        "1": "all", "2": "concerns_and_review", "3": "needs_review",
        "4": "none", "": "needs_review",
    }.get(
        choice, "needs_review"
    )


def _selected_for_plotting(
    results: Iterable[TriHydrAResult], mode: str
) -> list[TriHydrAResult]:
    if mode == "all":
        return list(results)
    if mode in {"needs_review", "review_only"}:
        return [result for result in results if station_requires_review(result)]
    if mode == "concerns_and_review":
        return [
            result for result in results
            if any(
                str(value).strip().casefold() in {
                    "minor concerns", "needs review", "review"
                }
                for name in result.summary.columns if name.endswith("_class")
                for value in result.summary[name].dropna()
            )
        ]
    return []


def _execute_config(path: str | Path = "trihydra.toml") -> TriHydrABatchResult:
    """Execute the shared validated workflow used by Python and the CLI."""
    started = time.perf_counter()
    public = load_toml_config(path)
    base_directory = Path(path).expanduser().resolve().parent
    runtime = build_runtime_config(public, base_directory)
    output_directory = public.output.directory
    if not output_directory.is_absolute():
        output_directory = (base_directory / output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(output_directory, public.output.write_log)
    selection = resolve_station_selection(public)
    selected_label = "all" if selection == "all" else str(len(selection))
    logger.info("TriHydrA started | selected stations: %s", selected_label)
    load_failures: list[dict[str, object]] = []

    def report_load_progress(event: str, row: dict[str, object]) -> None:
        if event != "failed":
            return
        load_failures.append(dict(row))
        logger.error(
            "failed loading station=%s | %s: %s | %.2f s",
            row["station_id"], row["error_type"], row["error_message"],
            row["elapsed_seconds"],
        )

    stations = load_stations(
        path,
        continue_on_station_error=public.run.continue_on_station_error,
        progress=report_load_progress,
    )
    for station in stations:
        if station.metadata.get("series2_status") == "not_available":
            logger.warning(
                "series2 unavailable station=%s | continuing series1-only",
                station.station_id,
            )

    def report_progress(event: str, row: dict[str, object]) -> None:
        station_id = row["station_id"]
        if event == "started":
            logger.info(
                "[%d/%d] processing station=%s",
                row["station_number"], row["station_count"], station_id,
            )
        elif event == "completed":
            logger.info(
                "completed station=%s | review=%s | %.2f s",
                station_id, "yes" if row["review_required"] else "no",
                row["elapsed_seconds"],
            )
        elif event == "skipped":
            logger.warning(
                "skipped station=%s | %s | %.2f s",
                station_id, row["error_message"], row["elapsed_seconds"],
            )
        elif event == "failed":
            logger.error(
                "failed station=%s | %s: %s | %.2f s",
                station_id, row["error_type"], row["error_message"],
                row["elapsed_seconds"],
            )

    context_path = (
        runtime["layer3"]["metadata"]["context_path"]
        if public.layers.layer3 else None
    )
    if stations:
        batch_result = run_trihydra_batch(
            stations,
            config=runtime,
            context_path=context_path,
            target_station_ids=[station.station_id for station in stations],
            continue_on_station_error=public.run.continue_on_station_error,
            progress=report_progress,
        )
    else:
        batch_result = TriHydrABatchResult(
            manifest=pd.DataFrame(), network=None, output_directory=None
        )
    if load_failures:
        batch_result.manifest = pd.concat(
            [pd.DataFrame(load_failures), batch_result.manifest],
            ignore_index=True,
            sort=False,
        )
    manifest = batch_result.manifest
    network_result = batch_result.network
    completed = [] if network_result is None else list(network_result.station_results.values())
    layer3_run = None if network_result is None else network_result.layer3_run
    if network_result is not None:
        network_result.configuration_used = public.model_dump(mode="json")
        if public.output.write_text:
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

    if public.output.write_netcdf:
        netcdf_path = write_netcdf_results(
            completed,
            manifest,
            output_directory / "trihydra_network_summary.nc",
            configuration=public.model_dump(mode="json"),
        )
        logger.info(
            "NetCDF results written: network=%s stations=%s",
            netcdf_path,
            netcdf_path.parent / "stations",
        )
    elapsed = time.perf_counter() - started
    completed_count = int((manifest.get("status") == "completed").sum()) if not manifest.empty else 0
    failed_count = int((manifest.get("status") == "failed").sum()) if not manifest.empty else 0
    logger.info(
        "TriHydrA finished | completed=%d failed=%d | elapsed %.2f s",
        completed_count, failed_count, elapsed,
    )
    batch_result.output_directory = output_directory
    return batch_result


def run_batch(path: str | Path = "trihydra.toml") -> TriHydrABatchResult:
    """Run the validated file workflow and return explorable results."""
    return _execute_config(path)


__all__ = ["run_batch"]
