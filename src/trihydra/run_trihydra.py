"""Run TriHydrA Layers 1-2 for raw Caravan observations.

This module is both a Python API and a small command-line runner. It accepts
one station, a list of stations, or all non-empty stations in one Caravan
NetCDF. Layer 1 always receives the untouched source series. Layer 2 receives
the same raw series and creates its own explicitly logged temporary analysis
copy where continuity is required.

Examples
--------
From the TriHydrA repository root::

    python -m src.trihydra.run_trihydra caravan_observations.nc \
        --station GRDC_5868050

    python -m src.trihydra.run_trihydra caravan_observations.nc \
        --station GRDC_5304140 --station GRDC_4123300

    python -m src.trihydra.run_trihydra caravan_observations.nc --all
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.trihydra.io.caravan_ingestion import (
    StationSelection,
    iter_caravan_stations,
    station_availability,
)
from src.trihydra.layer1.behaviour_profile import calculate_profile
from src.trihydra.layer1.config import DEFAULT_CONFIG, merge_config
from src.trihydra.layer1.layer1 import run_layer1
from src.trihydra.layer2.layer2 import run_layer2


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable.")


def run_caravan_layers12(
    netcdf_path: str | Path,
    stations: StationSelection = "all",
    *,
    output_root: str | Path = "trihydra_outputs",
    config: Mapping[str, Any] | None = None,
    show: bool | None = None,
) -> pd.DataFrame:
    """Run Layers 1-2 station-by-station and return a batch manifest."""
    effective = merge_config(DEFAULT_CONFIG, config)
    run_settings = effective["run"]
    layer1_settings = effective["layer1"]
    layer2_settings = effective["layer2"]
    input_settings = run_settings["input"]
    output_settings = run_settings["output"]
    layer_settings = run_settings["layers"]

    output_path = Path(output_root).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    show_figures = (
        bool(output_settings["show_figures"]) if show is None else bool(show)
    )
    continue_on_error = bool(output_settings["continue_on_station_error"])

    (output_path / "resolved_config.json").write_text(
        json.dumps(effective, indent=2, default=_json_default),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for station in iter_caravan_stations(
        netcdf_path,
        stations,
        unit=str(input_settings["observation_unit"]),
    ):
        availability = station_availability(station)
        raw_before = station.obs.copy(deep=True)
        row: dict[str, Any] = {
            **availability,
            "status": "running",
            "layer1_completed": False,
            "layer2_completed": False,
            "error_type": None,
            "error_message": None,
        }
        try:
            layer1_result = None
            if bool(layer_settings["run_layer1"]):
                layer1_result = run_layer1(
                    obs_series=station.obs,
                    sim_series=None,
                    station_id=station.station_id,
                    show=show_figures,
                    output_root=output_path,
                    config=layer1_settings,
                )
                row["layer1_completed"] = True
                row["layer1_flagged_checks"] = int(
                    len(layer1_result["summary_flagged"])
                )

            pd.testing.assert_series_equal(station.obs, raw_before)

            if bool(layer_settings["run_layer2"]):
                temporary = layer2_settings["temporary_imputation"]
                signature_kwargs = dict(layer2_settings["signatures"])
                layer2_result = run_layer2(
                    obs_series=station.obs,
                    ml_series=None,
                    station_id=station.station_id,
                    show=show_figures,
                    output_root=output_path,
                    fill_method=str(temporary["method"]),
                    fill_window_days=int(temporary["seasonal_window_days"]),
                    fill_min_samples=int(
                        temporary["minimum_seasonal_samples"]
                    ),
                    signature_kwargs=signature_kwargs,
                    layer1_obs_profile=calculate_profile(
                        station.obs, series_name="obs"
                    ),
                    discharge_unit=station.unit,
                )
                row["layer2_completed"] = True
                row["layer2_signature_count"] = int(
                    len(layer2_result["signature_comparison"])
                )
                row["layer2_temporarily_filled"] = int(
                    layer2_result["coverage"]["obs_temporarily_filled"]
                )

            pd.testing.assert_series_equal(station.obs, raw_before)
            row["status"] = "completed"
        except Exception as error:
            row["status"] = "failed"
            row["error_type"] = type(error).__name__
            row["error_message"] = str(error)
            error_dir = output_path / station.station_id
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "run_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            if not continue_on_error:
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    output_path / "run_manifest.csv", index=False
                )
                raise
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_path / "run_manifest.csv", index=False)
    if manifest.empty:
        raise ValueError(
            "No stations with dated streamflow observations were selected."
        )
    return manifest


def _parse_station_selection(
    station_args: Sequence[str] | None,
    all_stations: bool,
) -> StationSelection:
    if all_stations:
        return "all"
    if not station_args:
        raise ValueError("Provide --station at least once, or use --all.")
    values: list[str] = []
    for item in station_args:
        values.extend(value.strip() for value in item.split(",") if value.strip())
    return values[0] if len(values) == 1 else values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TriHydrA Layers 1-2 on Caravan observations."
    )
    parser.add_argument("netcdf", type=Path, help="Caravan observation NetCDF.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--station",
        action="append",
        help="Station ID; repeat or supply comma-separated IDs.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Process every station containing valid dated streamflow.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("trihydra_outputs")
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = _parse_station_selection(args.station, args.all)
    manifest = run_caravan_layers12(
        args.netcdf,
        selected,
        output_root=args.output_root,
        show=args.show,
    )
    print(manifest.to_string(index=False))
    print(f"\nManifest: {(args.output_root.resolve() / 'run_manifest.csv')}")
    return 0 if (manifest["status"] == "completed").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_caravan_layers12"]
