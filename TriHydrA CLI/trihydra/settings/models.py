"""Validated TOML configuration for the supported TriHydrA inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trihydra.settings.defaults import DEFAULT_CONFIG, merge_config


class StrictModel(BaseModel):
    """Reject misspelled or unsupported configuration fields."""

    model_config = ConfigDict(extra="forbid")


class RunSelection(StrictModel):
    """Choose inline IDs, a text file of IDs, or every available station."""

    station_ids: list[str] = Field(default_factory=list)
    station_file: Path | None = None
    all_stations: bool = False
    continue_on_station_error: bool = True

    @model_validator(mode="after")
    def selection_is_unambiguous(self) -> "RunSelection":
        cleaned = list(dict.fromkeys(value.strip() for value in self.station_ids if value.strip()))
        self.station_ids = cleaned
        modes = int(self.all_stations) + int(bool(cleaned)) + int(self.station_file is not None)
        if modes != 1:
            raise ValueError(
                "Choose exactly one station mode: provide run.station_ids, "
                "provide run.station_file, or set run.all_stations=true."
            )
        return self


class LayerSelection(StrictModel):
    layer1: bool = True
    layer2: bool = True
    layer3: bool = False
    comparison: bool = False


class NetCDFObservation(StrictModel):
    """One continuous station-by-time observation NetCDF."""

    format: Literal["netcdf"] = "netcdf"
    path: Path
    name: str = "observation"
    variable: str | None = None
    station_coordinate: str | None = None
    time_coordinate: str | None = None
    units: str = Field(min_length=1)


class AIFLPickleSimulation(StrictModel):
    """The trusted mentor-provided long-term AIFL result schema."""

    enabled: bool = False
    format: Literal["aifl_pickle"] = "aifl_pickle"
    path: Path | None = None
    model_name: str = "AIFL"
    units: str = "mm/day"
    trusted: bool = False
    time_step: Literal[0] = 0
    observation_variable: str = "streamflow_obs"
    simulation_variable: str = "streamflow_sim"

    @model_validator(mode="after")
    def enabled_pickle_is_explicitly_trusted(self) -> "AIFLPickleSimulation":
        if self.enabled and self.path is None:
            raise ValueError("simulation.path is required when simulation.enabled=true.")
        if self.enabled and not self.trusted:
            raise ValueError(
                "Pickle loading can execute code. Set simulation.trusted=true "
                "only for the supplied trusted long-term result files."
            )
        return self


class OutputConfig(StrictModel):
    directory: Path = Path("outputs")
    html_mode: Literal["ask", "all", "review_only", "none"] = "ask"
    non_interactive_html_mode: Literal["all", "review_only", "none"] = "review_only"
    show_figures: bool = False
    write_text: bool = True
    write_netcdf: bool = True
    write_log: bool = True
    display_decimals: int = Field(default=3, ge=0, le=10)


class ComparisonConfig(StrictModel):
    calculate_daily_metrics: bool = False
    include_provided_metrics: bool = True


def _reject_unknown_keys(overrides: dict[str, Any], defaults: dict[str, Any], path: str) -> None:
    """Catch threshold-name typos while retaining the established dictionaries."""
    unknown = sorted(set(overrides).difference(defaults))
    if unknown:
        location = path or "configuration"
        raise ValueError(f"Unknown field(s) under {location}: {', '.join(unknown)}")
    for key, value in overrides.items():
        default = defaults[key]
        if isinstance(value, dict) and isinstance(default, dict):
            _reject_unknown_keys(value, default, f"{path}.{key}".strip("."))


class TriHydrAConfig(StrictModel):
    """Complete supported configuration before it reaches the runner."""

    run: RunSelection
    layers: LayerSelection = Field(default_factory=LayerSelection)
    observation: NetCDFObservation
    simulation: AIFLPickleSimulation = Field(default_factory=AIFLPickleSimulation)
    output: OutputConfig = Field(default_factory=OutputConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    layer1: dict[str, Any] = Field(default_factory=dict)
    layer2: dict[str, Any] = Field(default_factory=dict)
    layer3: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def configuration_is_consistent(self) -> "TriHydrAConfig":
        _reject_unknown_keys(self.layer1, DEFAULT_CONFIG["layer1"], "layer1")
        _reject_unknown_keys(self.layer2, DEFAULT_CONFIG["layer2"], "layer2")
        _reject_unknown_keys(self.layer3, DEFAULT_CONFIG["layer3"], "layer3")
        layer1 = merge_config(DEFAULT_CONFIG["layer1"], self.layer1)
        spike = layer1["spike_dip"]
        if not 0 <= float(spike["minimum_recovery"]) <= 1:
            raise ValueError("layer1.spike_dip.minimum_recovery must be between 0 and 1.")
        for key in ("absolute_change_reference_quantile", "score_reference_quantile"):
            if not 0 <= float(spike[key]) <= 1:
                raise ValueError(f"layer1.spike_dip.{key} must be between 0 and 1.")
        for key in ("minimum_score", "robust_mad_multiplier", "minimum_outer_change_multiplier"):
            if float(spike[key]) < 0:
                raise ValueError(f"layer1.spike_dip.{key} cannot be negative.")
        composite = layer1["composite"]
        missing = composite["missing_values"]
        if not 0 <= float(missing["tier_2_minimum_percent"]) <= float(missing["tier_1_above_percent"]) <= 100:
            raise ValueError(
                "Layer 1 missingness thresholds must satisfy 0 <= Tier 2 "
                "minimum <= Tier 1 above-threshold <= 100."
            )
        gaps = composite["long_gaps"]
        if not 0 <= int(gaps["long_gap_definition_days"]) < int(gaps["tier_2_minimum_days"]) <= int(gaps["tier_1_minimum_days"]):
            raise ValueError(
                "Layer 1 gap thresholds must increase from the evidence "
                "definition through Tier 2 and Tier 1."
            )
        for key in ("tier_2_missing_share", "tier_1_missing_share"):
            if not 0 <= float(gaps[key]) <= 1:
                raise ValueError(f"layer1.composite.long_gaps.{key} must be between 0 and 1.")
        negative = composite["negative_discharge"]
        if not 0 <= float(negative["low_flow_reference_quantile"]) <= 1:
            raise ValueError(
                "layer1.composite.negative_discharge.low_flow_reference_quantile "
                "must be between 0 and 1."
            )
        if float(negative["tier_1_reference_multiplier"]) < 0:
            raise ValueError(
                "layer1.composite.negative_discharge."
                "tier_1_reference_multiplier cannot be negative."
            )
        if int(composite["low_variability"]["tier_1_minimum_days"]) < 2:
            raise ValueError(
                "layer1.composite.low_variability.tier_1_minimum_days must be "
                "at least 2."
            )
        if int(composite["spike_dip"]["tier_1_minimum_unresolved_count"]) < 1:
            raise ValueError(
                "layer1.composite.spike_dip.tier_1_minimum_unresolved_count "
                "must be at least 1."
            )
        step_shift = layer1["step_shift"]
        step_points = [
            float(step_shift["tier_3_points"]),
            float(step_shift["tier_2_points"]),
            float(step_shift["tier_1_points"]),
        ]
        if step_points != sorted(step_points):
            raise ValueError(
                "Layer 1 step-shift points must be ordered Tier 3 <= Tier 2 <= Tier 1."
            )
        if not (
            step_points[0]
            <= float(step_shift["composite_tier_2_minimum_score"])
            < float(step_shift["composite_tier_1_above_score"])
            <= step_points[2]
        ):
            raise ValueError(
                "Layer 1 step-shift composite cutoffs must increase within the "
                "configured boundary-tier point range."
            )
        if not 0 < float(step_shift["refinement_block_fraction"]) <= 1:
            raise ValueError(
                "layer1.step_shift.refinement_block_fraction must be greater "
                "than 0 and at most 1."
            )
        epoch = layer1["epoch_drift"]
        if not (
            0 <= float(epoch["tier_2_minimum_stable_fraction"])
            < float(epoch["tier_3_minimum_stable_fraction"]) <= 1
        ):
            raise ValueError(
                "Layer 1 epoch-drift stable-fraction cutoffs must satisfy "
                "0 <= Tier 2 minimum < Tier 3 minimum <= 1."
            )
        layer2 = merge_config(DEFAULT_CONFIG["layer2"], self.layer2)
        if float(layer2["events"]["spike_crosscheck_minimum_event_duration_days"]) < 1:
            raise ValueError(
                "layer2.events.spike_crosscheck_minimum_event_duration_days "
                "must be at least 1."
            )
        if any(not isinstance(value, int) or value < 0 for value in composite["weights"].values()):
            raise ValueError("Layer 1 composite weights cannot be negative.")
        points = composite["tier_points"]
        point_values = [points["tier_3"], points["tier_2"], points["tier_1"]]
        if (any(not isinstance(value, int) or value < 0 for value in point_values)
                or point_values != sorted(point_values)):
            raise ValueError(
                "Layer 1 tier points must be non-negative integers ordered "
                "Tier 3 <= Tier 2 <= Tier 1."
            )
        classification = composite["classification"]
        if int(classification["minor_concerns_minimum_score"]) >= int(classification["needs_review_minimum_score"]):
            raise ValueError(
                "The minor-concerns score must be below the needs-review score."
            )
        if self.layers.comparison and not self.simulation.enabled:
            raise ValueError(
                "layers.comparison=true requires simulation.enabled=true."
            )
        if self.simulation.enabled and self.simulation.units != self.observation.units:
            raise ValueError(
                "Observation and simulation units must match before comparison."
            )
        if self.layers.layer3 and not self.layers.layer1:
            raise ValueError("Layer 3 currently requires Layer 1 evidence.")
        if self.layers.layer3 and not self.layers.layer2:
            raise ValueError("Layer 3 currently requires Layer 2 evidence.")
        return self

    def station_selection(self) -> str | list[str]:
        if self.run.all_stations:
            return "all"
        if self.run.station_ids:
            return self.run.station_ids
        if self.run.station_file is None:
            raise RuntimeError("Validated configuration has no station-selection mode.")
        lines = self.run.station_file.read_text(encoding="utf-8").splitlines()
        selected = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
        selected = list(dict.fromkeys(selected))
        if not selected:
            raise ValueError(f"Station file contains no station IDs: {self.run.station_file}")
        return selected

    def runtime_overrides(self, base_directory: Path) -> dict[str, Any]:
        """Translate public TOML names into the existing internal configuration."""
        layer3 = dict(self.layer3)
        metadata = layer3.get("metadata")
        if isinstance(metadata, dict):
            metadata = dict(metadata)
            for key in ("context_path", "climate_raster", "climate_legend"):
                if key in metadata and metadata[key] is not None:
                    metadata[key] = str(_resolve_path(metadata[key], base_directory))
            layer3["metadata"] = metadata
        overrides = {
            "run": {
                "layers": {
                    "run_layer1": self.layers.layer1,
                    "run_layer2": self.layers.layer2,
                    "run_comparison": self.layers.comparison,
                },
            },
            "comparison": {
                "daily_metrics": {"calculate": self.comparison.calculate_daily_metrics},
                "provided_metrics": {"include": self.comparison.include_provided_metrics},
            },
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": layer3,
        }
        return merge_config(DEFAULT_CONFIG, overrides)


def _resolve_path(value: str | Path, base_directory: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_directory / path).resolve()


__all__ = [
    "AIFLPickleSimulation",
    "NetCDFObservation",
    "TriHydrAConfig",
]
