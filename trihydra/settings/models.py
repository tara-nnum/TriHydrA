"""Validated TOML configuration for the supported TriHydrA inputs."""

from __future__ import annotations

from pathlib import Path
from datetime import date
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


class TimespanConfig(StrictModel):
    """Use a complete input series or one explicit inclusive date range."""

    mode: Literal["full", "range"] = "full"
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def range_dates_are_complete_and_ordered(self) -> "TimespanConfig":
        if self.mode == "full":
            if self.start_date is not None or self.end_date is not None:
                raise ValueError(
                    "timespan start_date/end_date require mode='range'."
                )
            return self
        if self.start_date is None or self.end_date is None:
            raise ValueError(
                "timespan mode='range' requires start_date and end_date."
            )
        if self.start_date > self.end_date:
            raise ValueError("timespan start_date must not be after end_date.")
        return self


class Series1Config(StrictModel):
    """Required primary/reference input series."""

    format: Literal["netcdf", "csv", "aifl_pickle"]
    path: Path
    name: str = "series1"
    role: Literal["observation", "simulation", "historical_observation", "other"] = (
        "observation"
    )
    units: str = Field(min_length=1)
    timespan: TimespanConfig = Field(default_factory=TimespanConfig)
    # NetCDF fields. Equivalent station-by-time files are supported regardless
    # of their common NetCDF filename extension.
    variable: str | None = None
    station_coordinate: str | None = None
    time_coordinate: str | None = None
    engine: Literal["auto", "netcdf4", "h5netcdf"] = "auto"
    # Wide CSV field.
    date_column: str = "date"
    # Trusted AIFL-pickle fields.
    trusted: bool = False
    time_step: Literal[0] = 0
    observation_variable: str = "streamflow_obs"
    simulation_variable: str = "streamflow_sim"

    @model_validator(mode="after")
    def source_definition_is_complete_and_safe(self) -> "Series1Config":
        if self.format == "netcdf" and getattr(self, "enabled", True):
            missing = [
                name for name in (
                    "variable", "station_coordinate", "time_coordinate"
                )
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(
                    "NetCDF inputs require: " + ", ".join(missing) + "."
                )
        if (
            self.format == "aifl_pickle"
            and getattr(self, "enabled", True)
            and not self.trusted
        ):
            raise ValueError(
                "Pickle loading can execute code. Set series1.trusted=true "
                "only for the supplied trusted long-term result files."
            )
        return self


class Series2Config(Series1Config):
    """Optional comparison series using the same supported input formats."""

    enabled: bool = False
    path: Path | None = None
    format: Literal["netcdf", "csv", "aifl_pickle"] = "netcdf"
    name: str = "series2"
    role: Literal["observation", "simulation", "historical_observation", "other"] = (
        "simulation"
    )
    units: str = "source units"

    @model_validator(mode="after")
    def enabled_input_is_complete_and_safe(self) -> "Series2Config":
        if self.enabled and self.path is None:
            raise ValueError("series2.path is required when series2.enabled=true.")
        if self.enabled and self.format == "aifl_pickle" and not self.trusted:
            raise ValueError(
                "Pickle loading can execute code. Set series2.trusted=true "
                "only for the supplied trusted long-term result files."
            )
        return self

class OutputConfig(StrictModel):
    directory: Path = Path("outputs")
    html_mode: Literal[
        "ask", "all", "concerns_and_review", "needs_review", "review_only", "none"
    ] = "ask"
    non_interactive_html_mode: Literal[
        "all", "concerns_and_review", "needs_review", "review_only", "none"
    ] = "needs_review"
    show_figures: bool = False
    write_text: bool = True
    write_netcdf: bool = True
    write_log: bool = True


class ComparisonConfig(StrictModel):
    mode: Literal["paired_overlap", "independent_timespans"] = "paired_overlap"
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
    series1: Series1Config
    series2: Series2Config = Field(default_factory=Series2Config)
    output: OutputConfig = Field(default_factory=OutputConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    layer1: dict[str, Any] = Field(default_factory=dict)
    layer2: dict[str, Any] = Field(default_factory=dict)
    layer3: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def layer1_settings_are_consistent(self) -> "TriHydrAConfig":
        """Validate Layer 1 thresholds and composite settings."""
        _reject_unknown_keys(self.layer1, DEFAULT_CONFIG["layer1"], "layer1")
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
        if any(
            not isinstance(value, int) or value < 0
            for value in composite["weights"].values()
        ):
            raise ValueError("Layer 1 composite weights cannot be negative.")
        points = composite["tier_points"]
        point_values = [points["tier_3"], points["tier_2"], points["tier_1"]]
        if (
            any(not isinstance(value, int) or value < 0 for value in point_values)
            or point_values != sorted(point_values)
        ):
            raise ValueError(
                "Layer 1 tier points must be non-negative integers ordered "
                "Tier 3 <= Tier 2 <= Tier 1."
            )
        classification = composite["classification"]
        minor_percent = float(classification["minor_concerns_minimum_percent"])
        review_percent = float(classification["needs_review_minimum_percent"])
        if not 0 <= minor_percent < review_percent <= 100:
            raise ValueError(
                "Layer 1 percentage cutoffs must satisfy "
                "0 <= minor < review <= 100."
            )
        return self

    @model_validator(mode="after")
    def layer2_settings_are_consistent(self) -> "TriHydrAConfig":
        """Validate Layer 2 event and comparison settings."""
        _reject_unknown_keys(self.layer2, DEFAULT_CONFIG["layer2"], "layer2")
        layer2 = merge_config(DEFAULT_CONFIG["layer2"], self.layer2)
        events = layer2["events"]
        boundary = float(events["boundary_percentile"])
        trigger = float(events["trigger_percentile"])
        if not 0 < boundary < trigger < 1:
            raise ValueError(
                "Layer 2 event percentiles must satisfy "
                "0 < boundary_percentile < trigger_percentile < 1."
            )
        if float(events["spike_crosscheck_minimum_event_duration_days"]) < 1:
            raise ValueError(
                "layer2.events.spike_crosscheck_minimum_event_duration_days "
                "must be at least 1."
            )
        layer2_comparison = layer2["comparison"]
        tier2_similarity = float(layer2_comparison["similarity_tier2_minimum"])
        tier3_similarity = float(layer2_comparison["similarity_tier3_minimum"])
        if not 0 <= tier2_similarity < tier3_similarity <= 1:
            raise ValueError(
                "Layer 2 similarity cutoffs must satisfy "
                "0 <= Tier 2 minimum < Tier 3 minimum <= 1."
            )
        if not (
            0 <= int(layer2_comparison["seasonal_timing_tier3_max_months"])
            <= int(layer2_comparison["seasonal_timing_tier2_max_months"])
            <= 6
        ):
            raise ValueError(
                "Layer 2 seasonal timing cutoffs must increase from Tier 3 "
                "to Tier 2 and cannot exceed six months."
            )
        if not (
            0 <= float(layer2_comparison["time_to_peak_tier3_max_days"])
            < float(layer2_comparison["time_to_peak_tier1_min_days"])
        ):
            raise ValueError(
                "Layer 2 time-to-peak cutoffs must satisfy "
                "0 <= Tier 3 maximum < Tier 1 minimum."
            )
        if not (
            0 <= float(layer2_comparison["event_duration_tier3_max_days"])
            < float(layer2_comparison["event_duration_tier2_max_days"])
        ):
            raise ValueError(
                "Layer 2 event-duration cutoffs must satisfy "
                "0 <= Tier 3 maximum < Tier 2 maximum."
            )
        component_count = len(layer2_comparison["components"])
        minimum_components = int(layer2_comparison["minimum_assessable_components"])
        if not 1 <= minimum_components <= component_count:
            raise ValueError(
                "layer2.comparison.minimum_assessable_components must be "
                f"between 1 and {component_count}."
            )
        similarity_max = float(layer2_comparison["similar_maximum_percent"])
        review_max = float(layer2_comparison["review_maximum_percent"])
        if not 0 <= similarity_max < review_max <= 100:
            raise ValueError(
                "Layer 2 percentage cutoffs must satisfy "
                "0 <= similar < review <= 100."
            )
        if any(
            not isinstance(value, bool)
            for value in layer2_comparison["components"].values()
        ):
            raise ValueError("Layer 2 comparison component switches must be booleans.")
        if self.layers.comparison and not any(
            layer2_comparison["components"].values()
        ):
            raise ValueError(
                "Enable at least one Layer 2 comparison component when "
                "layers.comparison=true."
            )
        if any(
            not isinstance(value, int) or value < 0
            for value in layer2_comparison["weights"].values()
        ):
            raise ValueError("Layer 2 comparison weights cannot be negative.")
        enabled_components = layer2_comparison["components"]
        if self.layers.comparison and not any(
            layer2_comparison["weights"][name] > 0
            for name, enabled in enabled_components.items() if enabled
        ):
            raise ValueError(
                "At least one enabled Layer 2 comparison component must have "
                "a positive weight."
            )
        return self

    @model_validator(mode="after")
    def workflow_is_consistent(self) -> "TriHydrAConfig":
        """Validate dependencies between inputs, layers, and comparison mode."""
        if self.layers.comparison and not self.series2.enabled:
            raise ValueError(
                "layers.comparison=true requires series2.enabled=true."
            )
        if self.series2.enabled and self.series2.units != self.series1.units:
            raise ValueError(
                "series1 and series2 units must match before comparison."
            )
        if self.series2.enabled and self.series2.name == self.series1.name:
            raise ValueError(
                "series1.name and series2.name must differ so their results "
                "remain distinguishable."
            )
        if (
            self.comparison.mode == "independent_timespans"
            and self.comparison.calculate_daily_metrics
        ):
            raise ValueError(
                "Daily paired metrics are unavailable for independent_timespans."
            )
        if self.layers.layer3 and not self.layers.layer1:
            raise ValueError("Layer 3 currently requires Layer 1 evidence.")
        if self.layers.layer3 and not self.layers.layer2:
            raise ValueError("Layer 3 currently requires Layer 2 evidence.")
        return self

    @model_validator(mode="after")
    def layer3_settings_are_consistent(self) -> "TriHydrAConfig":
        """Validate Layer 3 peer selection and contextual thresholds."""
        _reject_unknown_keys(self.layer3, DEFAULT_CONFIG["layer3"], "layer3")
        layer3 = merge_config(DEFAULT_CONFIG["layer3"], self.layer3)
        for group_name in ("local_peers", "analogue_peers"):
            group = layer3[group_name]
            minimum = int(group["minimum_peers"])
            maximum = int(group["maximum_peers"])
            if not 1 <= minimum <= maximum:
                raise ValueError(
                    f"layer3.{group_name} peer limits must satisfy "
                    "1 <= minimum_peers <= maximum_peers."
                )
            if float(group["maximum_search_radius_km"]) <= 0:
                raise ValueError(
                    f"layer3.{group_name}.maximum_search_radius_km must be positive."
                )
        if float(layer3["analogue_peers"]["maximum_catchment_area_ratio"]) < 1:
            raise ValueError(
                "layer3.analogue_peers.maximum_catchment_area_ratio must be "
                "at least 1."
            )
        context = layer3["comparison"]
        for key in (
            "minimum_peak_timing_similarity",
            "minimum_step_shift_timing_similarity",
            "analogue_similarity_minimum",
            "peer_consensus_fraction",
        ):
            if not 0 <= float(context[key]) <= 1:
                raise ValueError(f"layer3.comparison.{key} must be between 0 and 1.")
        for key in (
            "local_context_weight",
            "comparable_catchment_weight",
            "peak_tolerance_days",
            "step_shift_tolerance_days",
            "minimum_epoch_overlap_years",
            "event_time_to_peak_tolerance_days",
            "event_duration_tolerance_days",
        ):
            if float(context[key]) < 0:
                raise ValueError(f"layer3.comparison.{key} cannot be negative.")
        if (
            float(context["local_context_weight"])
            + float(context["comparable_catchment_weight"])
            <= 0
        ):
            raise ValueError("At least one Layer 3 context weight must be positive.")
        partial = float(context["partial_minimum_percent"])
        similar = float(context["similar_minimum_percent"])
        if not 0 <= partial < similar <= 100:
            raise ValueError(
                "Layer 3 agreement cutoffs must satisfy "
                "0 <= partial < similar <= 100."
            )
        report_minimum = float(context["report_minimum_similarity_percent"])
        if not 0 <= report_minimum <= 100:
            raise ValueError(
                "layer3.comparison.report_minimum_similarity_percent must be "
                "between 0 and 100."
            )
        if int(context["minimum_profile_points"]) < 2:
            raise ValueError(
                "layer3.comparison.minimum_profile_points must be at least 2."
            )
        if layer3["plotting"]["mode"] not in {"recommended", "all", "none"}:
            raise ValueError(
                "layer3.plotting.mode must be 'recommended', 'all', or 'none'."
            )
        return self

__all__ = [
    "ComparisonConfig",
    "LayerSelection",
    "OutputConfig",
    "RunSelection",
    "Series1Config",
    "Series2Config",
    "TimespanConfig",
    "TriHydrAConfig",
]
