"""Public persistence API for completed in-memory results."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from trihydra.result import TriHydrANetworkResult, TriHydrAResult
from trihydra.outputs.reports import (
    render_evidence_report,
    render_network_summary,
    render_station_summary,
)
from trihydra.outputs.network_diagnostics import diagnostic_trigger_summary


def _json_value(value):
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable.")


def _label_evidence(
    table: pd.DataFrame, station_id: str, series_name: str, role: str
) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    frame = table.copy()
    frame.insert(0, "series_role", role)
    frame.insert(0, "series_name", series_name)
    frame.insert(0, "station_id", station_id)
    return frame


def _domain_evidence(result: TriHydrAResult, domain: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    source = getattr(result, domain)
    if source is not None:
        frames.append(_label_evidence(
            source.get("evidence"), result.station_id,
            result.station.series1_name, result.station.series1_role,
        ))
    if result.comparison is not None and domain == "layer1":
        frames.append(_label_evidence(
            result.comparison["candidate_layer1"].get("evidence"), result.station_id,
            result.comparison["candidate_name"], result.station.series2_role,
        ))
    if result.comparison is not None and domain == "layer2":
        frames.append(_label_evidence(
            result.comparison["candidate_native_layer2"].get("evidence"), result.station_id,
            result.comparison["candidate_name"], result.station.series2_role,
        ))
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _available_station_files(directory: Path, station_id: str) -> list[str]:
    """List files available for one station after all configured outputs finish."""
    paths = {path.name for path in directory.iterdir() if path.is_file()}
    paths.add("summary.txt")
    root = directory.parent
    for shared_name in (
        "network_summary.txt",
        "station_assessment_status.txt",
        "trihydra_network_summary.nc",
    ):
        if (root / shared_name).is_file():
            paths.add(f"../{shared_name}")
    station_netcdf = root / "stations" / f"{station_id}.nc"
    if station_netcdf.is_file():
        paths.add(f"../stations/{station_id}.nc")
    return sorted(paths, key=lambda item: (item.startswith("../"), item.casefold()))


def save_results(
    result: TriHydrAResult | TriHydrANetworkResult,
    output_directory: str | Path = "outputs",
    *,
    save_configuration: bool = False,
) -> dict[str, Path]:
    """Write readable TXT summaries and evidence for completed domains."""
    if isinstance(result, TriHydrANetworkResult):
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        network_summary = root / "network_summary.txt"
        network_summary.write_text(
            render_network_summary(
                result.summary,
                diagnostic_trigger_summary(result.station_results.values()),
            ),
            encoding="utf-8",
        )
        written = {"network_summary": network_summary}
        for station_id, station_result in result.station_results.items():
            for name, path in save_results(
                station_result, root,
                save_configuration=False,
            ).items():
                written[f"{station_id}:{name}"] = path
        if save_configuration:
            path = root / "configuration_used.json"
            path.write_text(
                json.dumps(result.configuration_used, indent=2, default=_json_value),
                encoding="utf-8",
            )
            written["configuration"] = path
        return written
    if not isinstance(result, TriHydrAResult):
        raise TypeError("result must be a TriHydrAResult or TriHydrANetworkResult.")
    directory = Path(output_directory) / result.station_id
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if result.layer1 is not None:
        path = directory / "layer1_evidence.txt"
        path.write_text(render_evidence_report(
            result.station_id, "Layer 1", _domain_evidence(result, "layer1")
        ), encoding="utf-8")
        written["layer1_evidence"] = path
    if result.layer2 is not None:
        path = directory / "layer2_evidence.txt"
        path.write_text(render_evidence_report(
            result.station_id, "Layer 2", _domain_evidence(result, "layer2")
        ), encoding="utf-8")
        written["layer2_evidence"] = path
    if result.comparison is not None:
        path = directory / "comparison_evidence.txt"
        frame = result.comparison["evidence"].copy()
        frame["candidate_name"] = result.comparison["candidate_name"]
        frame["reference_name"] = result.comparison["reference_name"]
        frame["station_id"] = result.station_id
        frame = frame[["station_id", "reference_name", "candidate_name"] + [
            column for column in frame.columns
            if column not in {"station_id", "reference_name", "candidate_name"}
        ]]
        path.write_text(render_evidence_report(
            result.station_id, "Comparison", frame
        ), encoding="utf-8")
        written["comparison_evidence"] = path
    if result.layer3 is not None:
        path = directory / "layer3_evidence.txt"
        frame = result.layer3["evidence"].copy()
        frame["station_id"] = result.station_id
        frame = frame[["station_id"] + [
            column for column in frame.columns if column != "station_id"
        ]]
        path.write_text(render_evidence_report(
            result.station_id, "Layer 3", frame
        ), encoding="utf-8")
        written["layer3_evidence"] = path
    if save_configuration:
        path = directory / "configuration_used.json"
        path.write_text(
            json.dumps(result.configuration_used, indent=2, default=_json_value),
            encoding="utf-8",
        )
        written["configuration"] = path
    summary_path = directory / "summary.txt"
    summary_path.write_text(
        render_station_summary(
            result,
            available_files=_available_station_files(directory, result.station_id),
        ),
        encoding="utf-8",
    )
    written["summary"] = summary_path
    return written


__all__ = ["save_results"]
