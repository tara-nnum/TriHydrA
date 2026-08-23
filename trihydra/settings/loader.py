"""Read and validate TriHydrA TOML files."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from trihydra.settings.defaults import DEFAULT_CONFIG, merge_config
from trihydra.settings.models import TriHydrAConfig


def resolve_path(value: str | Path, base_directory: Path) -> Path:
    """Resolve one user path relative to its TOML file."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_directory / path).resolve()


def resolve_station_selection(config: TriHydrAConfig) -> str | list[str]:
    """Return ``all`` or the de-duplicated station IDs selected by the user."""
    if config.run.all_stations:
        return "all"
    if config.run.station_ids:
        return config.run.station_ids
    if config.run.station_file is None:
        raise RuntimeError("Validated configuration has no station-selection mode.")
    lines = config.run.station_file.read_text(encoding="utf-8").splitlines()
    selected = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError(
            f"Station file contains no station IDs: {config.run.station_file}"
        )
    return selected


def build_runtime_config(
    config: TriHydrAConfig,
    base_directory: Path,
) -> dict[str, Any]:
    """Translate the public TOML model into the scientific runtime settings."""
    layer3 = dict(config.layer3)
    metadata = layer3.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        for key in ("context_path", "climate_raster", "climate_legend"):
            if key in metadata and metadata[key] is not None:
                metadata[key] = str(resolve_path(metadata[key], base_directory))
        layer3["metadata"] = metadata
    overrides = {
        "run": {
            "layers": {
                "run_layer1": config.layers.layer1,
                "run_layer2": config.layers.layer2,
                "run_comparison": config.layers.comparison,
            },
        },
        "comparison": {
            "mode": config.comparison.mode,
            "daily_metrics": {
                "calculate": config.comparison.calculate_daily_metrics
            },
            "provided_metrics": {
                "include": config.comparison.include_provided_metrics
            },
        },
        "layer1": config.layer1,
        "layer2": config.layer2,
        "layer3": layer3,
    }
    return merge_config(DEFAULT_CONFIG, overrides)


def load_toml_config(path: str | Path, *, check_paths: bool = True) -> TriHydrAConfig:
    """Return one validated configuration with paths resolved from its file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("rb") as handle:
        config = TriHydrAConfig.model_validate(tomllib.load(handle))
    config.series1.path = resolve_path(config.series1.path, config_path.parent)
    config.output.directory = resolve_path(config.output.directory, config_path.parent)
    if config.run.station_file is not None:
        config.run.station_file = resolve_path(config.run.station_file, config_path.parent)
    if config.series2.path is not None:
        config.series2.path = resolve_path(config.series2.path, config_path.parent)
    if check_paths:
        if not config.series1.path.exists():
            raise ValueError(
                "series1.path must exist: "
                f"{config.series1.path}"
            )
        if config.series1.format in {"netcdf", "csv"} and not config.series1.path.is_file():
            raise ValueError(
                f"series1.path must be a file for {config.series1.format}: "
                f"{config.series1.path}"
            )
        if config.series2.enabled and not config.series2.path.exists():
            raise ValueError(
                "series2.path must be an existing file or supported directory: "
                f"{config.series2.path}"
            )
        if (
            config.series2.enabled
            and config.series2.format in {"netcdf", "csv"}
            and not config.series2.path.is_file()
        ):
            raise ValueError(
                f"series2.path must be a file for {config.series2.format}: "
                f"{config.series2.path}"
            )
        if config.run.station_file is not None and not config.run.station_file.is_file():
            raise ValueError(
                f"run.station_file must be an existing text file: {config.run.station_file}"
            )
        if config.layers.layer3:
            context_value = config.layer3.get("metadata", {}).get(
                "context_path", "data/context.csv"
            )
            context_path = resolve_path(context_value, config_path.parent)
            if not context_path.is_file():
                raise ValueError(
                    "layer3.metadata.context_path must be an existing CSV file "
                    f"when Layer 3 is enabled: {context_path}"
                )
    return config


__all__ = [
    "build_runtime_config",
    "load_toml_config",
    "resolve_path",
    "resolve_station_selection",
]
