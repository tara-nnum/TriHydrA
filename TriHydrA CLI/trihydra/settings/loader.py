"""Read and validate TriHydrA TOML files."""

from __future__ import annotations

import tomllib
from pathlib import Path

from trihydra.settings.models import TriHydrAConfig, _resolve_path


def load_toml_config(path: str | Path, *, check_paths: bool = True) -> TriHydrAConfig:
    """Return one validated configuration with paths resolved from its file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("rb") as handle:
        config = TriHydrAConfig.model_validate(tomllib.load(handle))
    config.observation.path = _resolve_path(config.observation.path, config_path.parent)
    config.output.directory = _resolve_path(config.output.directory, config_path.parent)
    if config.run.station_file is not None:
        config.run.station_file = _resolve_path(config.run.station_file, config_path.parent)
    if config.simulation.path is not None:
        config.simulation.path = _resolve_path(config.simulation.path, config_path.parent)
    if check_paths:
        if not config.observation.path.is_file():
            raise ValueError(
                f"observation.path must be an existing NetCDF file: {config.observation.path}"
            )
        if config.simulation.enabled and not config.simulation.path.exists():
            raise ValueError(
                "simulation.path must be an existing trusted result pickle or directory: "
                f"{config.simulation.path}"
            )
        if config.run.station_file is not None and not config.run.station_file.is_file():
            raise ValueError(
                f"run.station_file must be an existing text file: {config.run.station_file}"
            )
        if config.layers.layer3:
            context_value = config.layer3.get("metadata", {}).get(
                "context_path", "data/context.csv"
            )
            context_path = _resolve_path(context_value, config_path.parent)
            if not context_path.is_file():
                raise ValueError(
                    "layer3.metadata.context_path must be an existing CSV file "
                    f"when Layer 3 is enabled: {context_path}"
                )
    return config


__all__ = ["load_toml_config"]
