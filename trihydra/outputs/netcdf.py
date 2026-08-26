"""Write compact, self-documenting multi-station NetCDF summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from trihydra.io.models import SourceProvenance
from trihydra.result import TriHydrAResult
from trihydra.outputs.network_diagnostics import (
    diagnostic_trigger_summary,
    network_assessment_counts,
    station_attention_ranking,
)

try:
    SOFTWARE_VERSION = version("trihydra")
except PackageNotFoundError:
    SOFTWARE_VERSION = "unknown"


# User-facing variables need domain descriptions rather than descriptions
# mechanically copied from their Python names. Less prominent evidence fields
# still receive the consistent fallback assembled by ``_variable_attributes``.
_VARIABLE_METADATA: dict[str, dict[str, str]] = {
    "latitude": {"units": "degrees_north", "description": "Gauge latitude."},
    "longitude": {"units": "degrees_east", "description": "Gauge longitude."},
    "catchment_area_km2": {
        "units": "km2",
        "description": "Upstream catchment area supplied in the context metadata.",
    },
    "series_available": {
        "description": "Indicates whether this station-series combination was available and assessed.",
        "interpretation": "1 means available; 0 means the series was not supplied for that station.",
    },
    "units": {
        "description": "Discharge unit declared for the input series; TriHydrA does not silently convert it.",
        "interpretation": "Observation and simulation units must match before comparison.",
    },
    "input_dataset": {
        "description": "Filename or source label from which the station series was read.",
        "interpretation": "Use with input_path, input_format and input_variable for provenance.",
    },
    "valid_observation_count": {
        "description": "Number of non-missing dated discharge values in the native series.",
        "method": "Count of values that are not NaN; no imputation is performed.",
    },
    "requested_timespan_mode": {
        "description": "Configured timespan mode for this input series.",
        "interpretation": "full selects the available record; range applies the inclusive requested start and end dates.",
    },
    "requested_start": {
        "description": "Inclusive start date requested by the user; blank when the full record was requested.",
    },
    "requested_end": {
        "description": "Inclusive end date requested by the user; blank when the full record was requested.",
    },
    "selected_calendar_start": {
        "description": "First timestamp retained after applying the configured timespan.",
    },
    "selected_calendar_end": {
        "description": "Last timestamp retained after applying the configured timespan.",
    },
    "first_valid_date": {
        "description": "First retained timestamp with a non-missing discharge value.",
    },
    "last_valid_date": {
        "description": "Last retained timestamp with a non-missing discharge value.",
    },
    "selected_row_count": {
        "units": "count",
        "description": "Number of source rows retained by the configured timespan, including missing values and duplicate timestamps.",
    },
    "layer1_missing_percentage": {
        "description": "Percentage of expected internal daily dates without a valid discharge value.",
        "method": "Missing internal calendar dates divided by the expected daily calendar length.",
        "interpretation": "Higher values indicate less complete temporal coverage.",
    },
    "layer1_longest_gap_days": {
        "description": "Duration of the longest continuous internal missing-data interval.",
        "interpretation": "Long gaps are assessed jointly with overall missingness in Layer 1.",
    },
    "layer1_duplicate_timestamp_count": {
        "description": "Number of calendar dates represented by more than one input row.",
        "interpretation": "A positive value indicates ambiguous duplicate observations.",
    },
    "layer1_irregular_timestep_count": {
        "description": "Number of consecutive unique-date intervals that are not one day.",
        "interpretation": "A positive value indicates an irregular native sampling calendar.",
    },
    "layer1_negative_discharge_count": {
        "description": "Number of materially negative discharge observations after applying numerical tolerance.",
        "interpretation": "Negative streamflow values can require source-data review.",
    },
    "layer1_zero_flow_percentage": {
        "description": "Percentage of valid observations classified as zero flow.",
        "method": "Zero-flow count divided by valid observation count; missing values are excluded.",
        "interpretation": "This is a flow-regime descriptor, not by itself an error flag.",
    },
    "layer1_unresolved_spike_dip_count": {
        "description": "Number of short-lived spike or dip candidates retained after recovery checks.",
        "interpretation": "Candidates are propagated to Layer 2 for peak-event cross-checking.",
    },
    "layer1_step_shift_score": {
        "description": "Mean tier-point score across retained persistent step-shift boundaries.",
        "method": "Boundary tier points are summed and divided by the retained boundary count.",
        "interpretation": "Values nearer 2 indicate more consistently material regime shifts.",
    },
    "layer1_epoch_stable_fraction": {
        "description": "Fraction of assessed annual epochs classified as stable.",
        "interpretation": "Higher values indicate greater long-term station stability.",
    },
    "layer1_score": {
        "description": "Weighted sum of assessable Layer 1 diagnostic contributions.",
        "interpretation": "Classification cutoffs are stored in the corresponding threshold variables.",
    },
    "layer1_score_percent": {
        "units": "%",
        "description": "Layer 1 score as a percentage of the maximum possible score from enabled, assessable checks.",
        "interpretation": "This remains comparable when users disable checks or run one check only.",
    },
    "layer1_class": {
        "description": "Final intrinsic time-series screening class derived strictly from the Layer 1 score.",
        "interpretation": "A review class requests inspection and does not prove that data are erroneous.",
    },
    "layer1_assessment_scope": {
        "description": "Whether the Layer 1 composite used every composite check or a user-selected subset.",
        "interpretation": "Full is an overall Layer 1 assessment; Focused applies only to the enabled checks.",
    },
    "layer1_scope_conclusion": {
        "description": "Scope-aware plain-language interpretation of the Layer 1 classification.",
        "interpretation": "Focused conclusions must not be interpreted as complete station-quality verdicts.",
    },
    "layer1_enabled_check_count": {
        "units": "count",
        "description": "Number of Layer 1 composite checks deliberately enabled by the user.",
    },
    "layer1_assessable_check_count": {
        "units": "count",
        "description": "Number of enabled Layer 1 composite checks with sufficient evidence.",
    },
    "layer1_total_composite_check_count": {
        "units": "count",
        "description": "Total number of checks available to the Layer 1 composite.",
    },
    "layer1_evidence_coverage_percent": {
        "units": "%",
        "description": "Assessable enabled checks as a percentage of enabled Layer 1 composite checks.",
    },
    "layer1_enabled_checks": {
        "description": "Semicolon-separated names of Layer 1 checks included in the composite scope.",
    },
    "layer1_disabled_checks": {
        "description": "Semicolon-separated names of Layer 1 checks deliberately excluded from the composite scope.",
    },
    "layer2_mean_discharge": {
        "description": "Arithmetic mean of valid native discharge observations.",
        "method": "Mean over valid values only; the raw record is not imputed.",
        "interpretation": "Descriptive hydrological signature; not directly a QC flag.",
    },
    "layer2_median_discharge": {
        "description": "Median of valid native discharge observations.",
        "method": "Median over valid values only; the raw record is not imputed.",
        "interpretation": "Descriptive hydrological signature; not directly a QC flag.",
    },
    "layer2_low_flow_q05": {
        "description": "Fifth percentile discharge, equivalent to conventional flow-duration Q95.",
        "interpretation": "Low-flow reference exceeded by approximately 95 percent of valid observations.",
    },
    "layer2_high_flow_q95": {
        "description": "Ninety-fifth percentile discharge, equivalent to conventional flow-duration Q5.",
        "interpretation": "High-flow reference exceeded by approximately 5 percent of valid observations.",
    },
    "layer2_median_annual_flashiness": {
        "description": "Median annual Richards-Baker flashiness index.",
        "interpretation": "Higher dimensionless values indicate greater day-to-day flow variability.",
    },
    "layer2_median_annual_baseflow_index": {
        "description": "Median annual Lyne-Hollick baseflow index.",
        "interpretation": "Higher dimensionless values indicate a larger slowly varying flow component.",
    },
    "layer2_median_annual_seasonality_index": {
        "description": "Median annual Walsh-Lawler seasonality index.",
        "interpretation": "Higher dimensionless values indicate stronger concentration of flow within the year.",
    },
    "layer2_high_flow_event_count": {
        "description": "Number of retained threshold-defined high-flow events.",
        "interpretation": "Spike/dip candidates are excluded from representative-event selection.",
    },
    "layer2_representative_event_duration_days": {
        "description": "Calendar duration of the observed representative high-flow event.",
        "interpretation": "The event is selected from retained observed events, not a synthetic average curve.",
    },
    "layer2_representative_event_rising_slope": {
        "units": "source discharge units day-1",
        "description": "Median rising-limb discharge change per day for the representative event.",
    },
    "layer2_representative_event_recession_slope": {
        "units": "source discharge units day-1",
        "description": "Median recession-limb discharge change per day for the representative event.",
    },
    "layer2_spike_peak_overlap_count": {
        "description": "Number of Layer 1 spike/dip candidates coinciding with Layer 2 high-flow peaks.",
        "interpretation": "A positive value identifies events requiring spike-versus-peak interpretation.",
    },
    "comparison_score": {
        "description": "Weighted Layer 2 dissimilarity score for the reference and candidate series.",
        "interpretation": "Lower scores indicate closer hydrological agreement; review cutoffs are threshold variables.",
    },
    "comparison_score_percent": {
        "units": "%",
        "description": "Layer 2 dissimilarity score normalized by the maximum possible score from enabled, assessable components.",
        "interpretation": "Lower percentages indicate closer hydrological agreement.",
    },
    "comparison_class": {
        "description": "Layer 2 pairwise comparison classification derived from assessable components.",
        "interpretation": "Review indicates hydrological disagreement requiring inspection.",
    },
    "layer3_context_agreement_score_percent": {
        "description": "Weighted contextual agreement across assessed nearby-gauge and comparable-catchment checks.",
        "method": "Configured nearby and comparable-catchment weights are applied only to assessed evidence groups.",
        "interpretation": "Higher percentages indicate stronger contextual support from the selected gauge network.",
    },
    "layer3_context_agreement_class": {
        "description": "Categorical network-context agreement derived from the Layer 3 agreement score.",
        "interpretation": "Reported as low, moderate, strong, or not assessed agreement.",
    },
    "processing_status": {
        "description": "Batch-processing outcome for the requested station.",
        "interpretation": "Completed, skipped, or failed; consult processing_message for details.",
    },
    "elapsed_seconds": {
        "units": "s",
        "description": "Elapsed processing time recorded for this station.",
    },
    "review_required": {
        "description": "Whether any completed Layer 1 or pairwise comparison classification requests review.",
        "interpretation": "1 requests inspection; 0 means no configured review rule was triggered.",
    },
    "flagged_diagnostic_count": {
        "description": "Number of Layer 1 diagnostics with a positive composite contribution.",
        "interpretation": "Clean diagnostics are omitted from the sparse flag_record table.",
    },
    "diagnostic_status": {
        "description": "Compact Layer 1 diagnostic status for each available station-series pair.",
        "interpretation": "Clean means no positive contribution; Concerns found means at least one flag record exists.",
    },
    "flagged_diagnostic_weight": {
        "units": "1",
        "description": "Configured dimensionless weight of the flagged Layer 1 diagnostic.",
    },
    "flagged_diagnostic_tier_points": {
        "units": "points",
        "description": "Tier points assigned to the flagged Layer 1 diagnostic.",
    },
    "flagged_diagnostic_contribution": {
        "units": "points",
        "description": "Tier points multiplied by diagnostic weight.",
    },
}


def _plain_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def _text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _provenance_values(provenance: SourceProvenance | None) -> dict[str, str]:
    if provenance is None:
        return {name: "" for name in ("input_dataset", "input_path", "input_format", "input_variable")}
    path = provenance.path
    source = provenance.details.get("source") if provenance.details else None
    return {
        "input_dataset": _text(source or (path.name if path else provenance.format)),
        "input_path": "" if path is None else str(path),
        "input_format": _text(provenance.format),
        "input_variable": _text(provenance.variable),
    }


def _summary_rows(results: Iterable[TriHydrAResult]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        frame = result.summary.copy()
        if frame.empty:
            continue
        for index, row in frame.iterrows():
            is_primary = row.get("series_name") == result.station.series1_name
            provenance = result.station.series1_provenance if is_primary else result.station.series2_provenance
            for name, value in _provenance_values(provenance).items():
                frame.at[index, name] = value
            series = result.station.series1 if is_primary else result.station.series2
            if series is not None and len(series):
                dates = pd.DatetimeIndex(series.index)
                frame.at[index, "record_start"] = dates.min()
                frame.at[index, "record_end"] = dates.max()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    summary = pd.concat(frames, ignore_index=True, sort=False)
    summary = summary.rename(columns={
        "unit": "units",
        "comparison_layer2_comparison_score": "comparison_score",
        "comparison_layer2_comparison_score_percent": "comparison_score_percent",
        "comparison_layer2_comparison_class": "comparison_class",
    })
    summary.columns = [
        name.replace("threshold_comparison_layer2_comparison_", "threshold_comparison_")
        for name in summary.columns
    ]
    return summary


def _layer1_components(result: TriHydrAResult, series_name: str) -> pd.DataFrame:
    if series_name == result.station.series1_name:
        composite = result.layer1_composite
    elif result.comparison is not None and series_name == result.comparison.get("candidate_name"):
        composite = result.comparison.get("candidate_layer1_composite")
    else:
        composite = None
    if not composite:
        return pd.DataFrame()
    components = composite.get("components", pd.DataFrame())
    return components if isinstance(components, pd.DataFrame) else pd.DataFrame()


def _variable_attributes(name: str) -> dict[str, str]:
    layer = (
        "Layer 1" if name.startswith(("layer1_", "threshold_layer1_"))
        else "Layer 2" if name.startswith(("layer2_", "threshold_layer2_"))
        else "Comparison" if name.startswith(("comparison_", "threshold_comparison_"))
        else "Layer 3" if name.startswith(("layer3_", "threshold_layer3_"))
        else "Metadata"
    )
    lowered = name.casefold()
    tokens = lowered.split("_")
    units = (
        # Both ``percentage`` and ``percent`` occur in the established result
        # contract. ``percentile`` is deliberately excluded because those
        # configured quantiles are stored as fractions from 0 to 1.
        "%" if "percentage" in tokens or "percent" in tokens
        else "1" if "percentile" in tokens or "quantile" in tokens
        else "days" if lowered.endswith("_days") or "duration_days" in lowered or "gap_days" in lowered
        else "count" if any(token in lowered for token in ("count", "year_count", "assessable_component"))
        else "source discharge units" if "discharge" in lowered or "flow" in lowered
        else "points" if any(token in lowered for token in ("score", "contribution", "weight"))
        else "1"
    )
    attrs = {
        "long_name": name.replace("_", " ").capitalize(),
        "description": f"TriHydrA {layer.lower()} output: {name.replace('_', ' ')}.",
        "units": units,
        "layer": layer,
    }
    attrs.update(_VARIABLE_METADATA.get(name, {}))
    if name.startswith("threshold_"):
        attrs.update(
            threshold_definition="Actual configured or station-derived threshold used for this assessment.",
            interpretation="Use with the corresponding diagnostic value.",
        )
    return attrs


def _normalise_column(values: pd.Series, name: str = "") -> tuple[np.ndarray, str]:
    nonmissing = values.dropna()
    if nonmissing.empty:
        if name in {
            "series_role", "units", "river_name", "catchment_name", "input_dataset",
            "climate_code", "climate_description", "input_path", "input_format",
            "input_variable", "processing_status",
            "error_type", "processing_message",
        } or name.endswith(("_class", "_tier", "_regime", "_concerns", "_components")):
            return np.full(len(values), "", dtype=str), "text"
        return np.full(len(values), np.nan), "numeric"
    if all(isinstance(value, (pd.Timestamp, datetime, np.datetime64)) for value in nonmissing):
        return pd.to_datetime(values, errors="coerce").to_numpy(), "datetime"
    if all(isinstance(value, (bool, np.bool_)) for value in nonmissing):
        array = np.full(len(values), -1, dtype=np.int8)
        present = values.notna().to_numpy()
        array[present] = values[present].astype(bool).astype(np.int8)
        return array, "boolean"
    numeric = pd.to_numeric(values, errors="coerce")
    if int(numeric.notna().sum()) == len(nonmissing):
        return numeric.to_numpy(dtype=float), "numeric"
    return values.map(_text).to_numpy(dtype=str), "text"


def _build_flat_dataset(
    results: Iterable[TriHydrAResult],
    manifest: pd.DataFrame,
    *,
    configuration: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Build the complete internal table before arranging user-facing groups."""
    completed = list(results)
    summary = _summary_rows(completed)
    manifest = manifest.copy()
    stations = list(dict.fromkeys(manifest.get("station_id", pd.Series(dtype=str)).astype(str)))
    for station_id in summary.get("station_id", pd.Series(dtype=str)).astype(str):
        if station_id not in stations:
            stations.append(station_id)
    default_series = completed[0].station.series1_name if completed else "series1"
    series_names = list(dict.fromkeys(summary.get("series_name", pd.Series([default_series])).astype(str)))
    if default_series in series_names:
        series_names.remove(default_series)
        series_names.insert(0, default_series)
    if not series_names:
        series_names = [default_series]

    dataset = xr.Dataset(coords={
        "station": ("station", np.asarray(stations, dtype=str)),
        "series": ("series", np.asarray(series_names, dtype=str)),
    })
    dataset.station.attrs.update(long_name="Station identifier", cf_role="timeseries_id")
    dataset.series.attrs.update(long_name="Input series name")

    full_index = pd.MultiIndex.from_product([stations, series_names], names=["station_id", "series_name"])
    indexed = summary.drop_duplicates(["station_id", "series_name"], keep="last").set_index(
        ["station_id", "series_name"]
    ).reindex(full_index)
    known = set(map(tuple, summary[["station_id", "series_name"]].astype(str).to_numpy()))
    available = np.asarray([tuple(index) in known for index in full_index], dtype=np.int8).reshape(
        len(stations), len(series_names)
    )
    dataset["series_available"] = (("station", "series"), available)
    dataset["series_available"].attrs.update(
        long_name="Series availability", flag_values=np.asarray([0, 1], dtype=np.int8),
        flag_meanings="not_available available",
    )
    for name in indexed.columns:
        array, kind = _normalise_column(indexed[name].reset_index(drop=True), name)
        dataset[name] = (("station", "series"), array.reshape(len(stations), len(series_names)))
        dataset[name].attrs.update(_variable_attributes(name))
        if kind == "datetime":
            # CF encoders own the datetime ``units`` attribute (for example,
            # "days since ..."). Let xarray write it during serialization.
            dataset[name].attrs.pop("units", None)
        if kind == "boolean":
            dataset[name].attrs.update(
                flag_values=np.asarray([-1, 0, 1], dtype=np.int8),
                flag_meanings="not_assessed false true",
            )

    manifest_indexed = manifest.drop_duplicates("station_id", keep="last").set_index("station_id")
    for source_name, target_name in (
        ("status", "processing_status"), ("error_type", "error_type"),
        ("error_message", "processing_message"), ("elapsed_seconds", "elapsed_seconds"),
        ("review_required", "review_required"),
    ):
        values = manifest_indexed[source_name].reindex(stations) if source_name in manifest_indexed else pd.Series(index=stations, dtype=object)
        array, kind = _normalise_column(values.reset_index(drop=True), target_name)
        dataset[target_name] = (("station",), array)
        dataset[target_name].attrs.update(_variable_attributes(target_name))
        if kind == "boolean":
            dataset[target_name].attrs.update(
                flag_values=np.asarray([-1, 0, 1], dtype=np.int8),
                flag_meanings="not_assessed false true",
            )

    result_by_station = {result.station_id: result for result in completed}
    flag_rows: list[dict[str, Any]] = []
    counts = np.zeros((len(stations), len(series_names)), dtype=np.int16)
    statuses = np.full((len(stations), len(series_names)), "Not assessed", dtype=object)
    for station_index, station_id in enumerate(stations):
        result = result_by_station.get(station_id)
        for series_index, series_name in enumerate(series_names):
            components = pd.DataFrame() if result is None else _layer1_components(result, series_name)
            rows = [] if components.empty else components.loc[
                components["contribution"] > 0
            ].to_dict("records")
            counts[station_index, series_index] = len(rows)
            if available[station_index, series_index]:
                statuses[station_index, series_index] = "Clean" if not rows else "Concerns found"
            for row in rows:
                flag_rows.append({
                    "flagged_station_id": station_id,
                    "flagged_series_name": series_name,
                    "flagged_diagnostic_name": _text(row.get("check")),
                    "flagged_diagnostic_tier": _text(row.get("tier")),
                    "flagged_diagnostic_value": _text(row.get("raw_value")),
                    "flagged_diagnostic_tier_points": row.get("tier_points"),
                    "flagged_diagnostic_weight": row.get("weight"),
                    "flagged_diagnostic_contribution": row.get("contribution"),
                    "flagged_diagnostic_summary": _text(row.get("reason")),
                })
    dataset["flagged_diagnostic_count"] = (("station", "series"), counts)
    dataset["diagnostic_status"] = (("station", "series"), statuses.astype(str))
    dataset = dataset.assign_coords(
        flag_record=("flag_record", np.arange(1, len(flag_rows) + 1, dtype=np.int32))
    )
    for name in (
        "flagged_station_id", "flagged_series_name", "flagged_diagnostic_name",
        "flagged_diagnostic_tier", "flagged_diagnostic_value", "flagged_diagnostic_summary",
    ):
        dataset[name] = (("flag_record",), np.asarray([row[name] for row in flag_rows], dtype=str))
        dataset[name].attrs.update(_variable_attributes(name))
    for name in (
        "flagged_diagnostic_tier_points", "flagged_diagnostic_weight",
        "flagged_diagnostic_contribution",
    ):
        dataset[name] = (("flag_record",), pd.to_numeric(
            pd.Series([row[name] for row in flag_rows]), errors="coerce"
        ).to_numpy(dtype=float))
        dataset[name].attrs.update(_variable_attributes(name))
    dataset["flagged_diagnostic_count"].attrs.update(
        long_name="Number of flagged Layer 1 diagnostics", units="count", layer="Layer 1",
        description="Only diagnostics with a positive composite contribution are counted.",
    )
    dataset["diagnostic_status"].attrs.update(
        long_name="Sparse diagnostic status", layer="Layer 1",
        description="Clean when no Layer 1 diagnostic contributes points; otherwise Concerns found.",
    )

    statuses_series = manifest.get("status", pd.Series(dtype=str)).astype(str)
    review = manifest.loc[statuses_series.eq("completed"), "review_required"] if "review_required" in manifest else pd.Series(dtype=bool)
    classes = summary.get("layer1_class", pd.Series(dtype=str)).astype(str)
    dataset.attrs.update(
        title="TriHydrA streamflow plausibility assessment", software="TriHydrA",
        software_version=SOFTWARE_VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        Conventions="CF-1.10",
        assessment_layers="Layer 1: intrinsic diagnostics; Layer 2: hydrological signatures; Layer 3: network context",
        review_definition="A review classification requests inspection; it does not prove erroneous data.",
        threshold_information="Static rules are variable attributes; station-specific thresholds are variables prefixed threshold_.",
        requested_station_count=len(stations), completed_station_count=int(statuses_series.eq("completed").sum()),
        skipped_station_count=int(statuses_series.eq("skipped").sum()), failed_station_count=int(statuses_series.eq("failed").sum()),
        needs_review_station_count=int(review.fillna(False).astype(bool).sum()),
        minor_concerns_series_count=int(classes.eq("Minor concerns").sum()),
        no_review_series_count=int(classes.eq("No review needed").sum()),
        configuration_used=_plain_json(configuration or {}),
        input_datasets="; ".join(sorted(filter(None, summary.get("input_dataset", pd.Series(dtype=str)).map(_text).unique()))),
        history="Created from completed in-memory results; the writer did not recalculate diagnostics.",
    )
    return dataset


_ROOT_VARIABLES = (
    "series_available", "series_role", "units", "processing_status",
    "processing_message", "elapsed_seconds", "review_required",
    "layer1_class", "layer1_score_percent", "layer1_assessment_scope",
    "comparison_class", "comparison_score_percent", "layer3_status",
    "layer3_context_agreement_class", "flagged_diagnostic_count",
    "diagnostic_status",
)

_METADATA_VARIABLES = (
    "series_name", "series_role", "units", "input_dataset", "input_path",
    "input_format", "input_variable", "requested_timespan_mode",
    "requested_start", "requested_end", "source_calendar_start",
    "source_calendar_end", "selected_calendar_start", "selected_calendar_end",
    "first_valid_date", "last_valid_date", "selected_row_count",
    "valid_observation_count", "record_start", "record_end",
)


def _encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Return compact lossless compression settings for numeric variables."""
    return {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name, variable in dataset.data_vars.items()
        if variable.dtype.kind in "fiu" and variable.size
    }


def _present(value: Any) -> bool:
    """Return whether one scalar carries usable information."""
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    return str(value).strip() != ""


def _first_present(variable: xr.DataArray | None) -> Any:
    """Return the first populated scalar from a station-level variable."""
    if variable is None:
        return None
    for value in np.asarray(variable.values).reshape(-1):
        if _present(value):
            return value.item() if hasattr(value, "item") else value
    return None


def _collapse_sparse_series(dataset: xr.Dataset) -> xr.Dataset:
    """Collapse pair-level values that occur on only one series row."""
    if "series_slot" not in dataset.dims:
        return dataset
    collapsed = xr.Dataset(attrs=dataset.attrs)
    for name, variable in dataset.data_vars.items():
        if "series_slot" not in variable.dims:
            collapsed[name] = variable
            continue
        values = np.asarray(variable.values).reshape(-1)
        populated = [index for index, value in enumerate(values) if _present(value)]
        if len(populated) <= 1:
            if populated:
                collapsed[name] = variable.isel(series_slot=populated[0], drop=True)
            else:
                collapsed[name] = variable.isel(series_slot=0, drop=True)
        elif all(str(values[index]) == str(values[populated[0]]) for index in populated):
            collapsed[name] = variable.isel(series_slot=populated[0], drop=True)
        else:
            collapsed[name] = variable
    if any("series_slot" in variable.dims for variable in collapsed.data_vars.values()):
        collapsed = collapsed.assign_coords(series_slot=dataset.series_slot)
    return collapsed


def _station_view(flat: xr.Dataset, station_id: str) -> xr.Dataset:
    """Select one station and expose neutral series1/series2 coordinate slots."""
    selected = flat.sel(station=station_id, drop=True)
    available = np.asarray(selected["series_available"].values).astype(bool)
    indices = np.flatnonzero(available)
    if not len(indices):
        indices = np.arange(selected.sizes.get("series", 0))
    selected = selected.isel(series=indices)
    configured_names = np.asarray(selected.series.values, dtype=str)
    selected = selected.rename({"series": "series_slot"})
    slots = np.asarray([f"series{index + 1}" for index in range(len(indices))], dtype=str)
    selected = selected.assign_coords(series_slot=("series_slot", slots))
    selected["series_name"] = ("series_slot", configured_names)
    selected["series_name"].attrs.update(
        long_name="Configured series name",
        description="User-defined name from the TOML series1 or series2 section.",
    )
    selected.series_slot.attrs.update(
        long_name="Stable input-series slot",
        description="series1 is the primary/reference input; series2 is the optional comparison input.",
    )
    return selected


def _variables(dataset: xr.Dataset, names: Iterable[str]) -> xr.Dataset:
    """Select existing variables while preserving required coordinates."""
    selected = [name for name in names if name in dataset]
    result = dataset[selected] if selected else xr.Dataset()
    result.attrs = {}
    return result


def _prefixed(dataset: xr.Dataset, prefix: str, *, exclude_thresholds: bool = True) -> xr.Dataset:
    names = [
        name for name in dataset.data_vars
        if name.startswith(prefix)
        and not (exclude_thresholds and name.startswith("threshold_"))
    ]
    return _variables(dataset, names)


def _station_groups(flat: xr.Dataset, station_id: str, configuration: Mapping[str, Any] | None) -> dict[str, xr.Dataset]:
    """Arrange one station into a compact root and purpose-specific groups."""
    station = _station_view(flat, station_id)
    latitude = _first_present(station.get("latitude"))
    longitude = _first_present(station.get("longitude"))
    root_names = ["series_name", *_ROOT_VARIABLES]
    root = _variables(station, root_names)
    if latitude is not None:
        root = root.assign_coords(latitude=float(latitude))
        root.latitude.attrs.update(standard_name="latitude", units="degrees_north")
    if longitude is not None:
        root = root.assign_coords(longitude=float(longitude))
        root.longitude.attrs.update(standard_name="longitude", units="degrees_east")

    station_attrs = {
        "title": f"TriHydrA station assessment: {station_id}",
        "station_id": station_id,
        "software": "TriHydrA",
        "software_version": SOFTWARE_VERSION,
        "created_utc": flat.attrs.get("created_utc", ""),
        "Conventions": "CF-1.10",
        "review_definition": flat.attrs.get("review_definition", ""),
        "group_information": (
            "Open metadata, layer1, layer2, comparison, layer3, flags, "
            "thresholds, evidence, or configuration for detailed results."
        ),
    }
    for variable_name, attribute_name in (
        ("river_name", "river_name"),
        ("catchment_name", "catchment_name"),
        ("catchment_area_km2", "catchment_area_km2"),
        ("climate_code", "climate_code"),
        ("climate_description", "climate_description"),
    ):
        value = _first_present(station.get(variable_name))
        if value is not None:
            station_attrs[attribute_name] = value
    root.attrs.update(station_attrs)

    metadata = _variables(station, _METADATA_VARIABLES)
    metadata.attrs.update(
        title="Input-series identity, provenance, units, and selected timespans",
        station_id=station_id,
    )
    layer1 = _prefixed(station, "layer1_")
    layer1.attrs.update(title="Layer 1 intrinsic time-series diagnostics", station_id=station_id)
    layer2 = _prefixed(station, "layer2_")
    layer2.attrs.update(title="Layer 2 hydrological signatures", station_id=station_id)
    comparison = _collapse_sparse_series(_prefixed(station, "comparison_"))
    comparison.attrs.update(
        title="Pairwise comparison between configured series1 and series2",
        station_id=station_id,
        comparison_status="assessed" if comparison.data_vars else "not_assessed",
    )
    layer3 = _collapse_sparse_series(_prefixed(station, "layer3_"))
    layer3.attrs.update(title="Layer 3 network context", station_id=station_id)
    thresholds = _variables(
        station, [name for name in station.data_vars if name.startswith("threshold_")]
    )
    thresholds.attrs.update(
        title="Configured and station-derived thresholds actually used",
        station_id=station_id,
    )

    flag_mask = np.asarray(flat["flagged_station_id"].values, dtype=str) == station_id
    flag_indices = np.flatnonzero(flag_mask)
    flag_names = [
        name for name, variable in flat.data_vars.items()
        if name.startswith("flagged_") and "flag_record" in variable.dims
    ]
    flags = flat[flag_names].isel(flag_record=flag_indices).rename(
        {"flag_record": "diagnostic_record"}
    )
    if "flagged_station_id" in flags:
        flags = flags.drop_vars("flagged_station_id")
    flags = flags.assign_coords(
        diagnostic_record=(
            "diagnostic_record", np.arange(1, len(flag_indices) + 1, dtype=np.int32)
        )
    )
    flags.attrs = {
        "title": "Sparse table containing only diagnostics with positive contribution",
        "station_id": station_id,
    }

    assigned = set(root_names) | set(_METADATA_VARIABLES) | {
        name for name in station.data_vars
        if name.startswith(("layer1_", "layer2_", "comparison_", "layer3_", "threshold_"))
    }
    evidence = _variables(
        station,
        [
            name for name in station.data_vars
            if name not in assigned
            and name not in {
                "error_type", "latitude", "longitude", "river_name",
                "catchment_name", "catchment_area_km2", "climate_code",
                "climate_description",
            }
            and not name.startswith("flagged_")
            and "flag_record" not in station[name].dims
        ],
    )
    evidence.attrs.update(
        title="Additional retained processing and diagnostic evidence",
        station_id=station_id,
    )
    configuration_group = xr.Dataset(attrs={
        "title": "Complete normalized configuration used for this run",
        "station_id": station_id,
        "configuration_json": _plain_json(configuration or {}),
    })
    return {
        "root": root,
        "metadata": metadata,
        "layer1": layer1,
        "layer2": layer2,
        "comparison": comparison,
        "layer3": layer3,
        "flags": flags,
        "thresholds": thresholds,
        "evidence": evidence,
        "configuration": configuration_group,
    }


def _build_network_dataset(
    flat: xr.Dataset,
    completed: list[TriHydrAResult],
) -> xr.Dataset:
    """Build the compact network index from the prepared result table."""
    stations = np.asarray(flat.station.values, dtype=str)
    network = xr.Dataset(coords={"station": ("station", stations)})
    network.station.attrs.update(long_name="Station identifier", cf_role="timeseries_id")
    for name in ("processing_status", "processing_message", "elapsed_seconds", "review_required"):
        if name in flat:
            network[name] = flat[name]
    primary_names: list[str] = []
    primary_classes: list[str] = []
    primary_scores: list[float] = []
    comparison_classes: list[str] = []
    layer3_classes: list[str] = []
    flag_counts: list[int] = []
    for station_id in stations:
        station = _station_view(flat, station_id)
        primary_names.append(str(station["series_name"].values[0]))
        primary_classes.append(str(station["layer1_class"].values[0]) if "layer1_class" in station else "")
        primary_scores.append(float(station["layer1_score_percent"].values[0]) if "layer1_score_percent" in station else np.nan)
        comparison_classes.append(str(_first_present(station.get("comparison_class")) or "Not assessed"))
        layer3_classes.append(str(_first_present(station.get("layer3_context_agreement_class")) or "Not assessed"))
        flag_counts.append(int(np.nansum(station["flagged_diagnostic_count"].values)) if "flagged_diagnostic_count" in station else 0)
    network["primary_series_name"] = ("station", np.asarray(primary_names, dtype=str))
    network["primary_layer1_class"] = ("station", np.asarray(primary_classes, dtype=str))
    network["primary_layer1_score_percent"] = ("station", np.asarray(primary_scores, dtype=float))
    network["comparison_class"] = ("station", np.asarray(comparison_classes, dtype=str))
    network["layer3_agreement_class"] = ("station", np.asarray(layer3_classes, dtype=str))
    network["flagged_diagnostic_count"] = ("station", np.asarray(flag_counts, dtype=np.int16))
    network["station_result_file"] = (
        "station", np.asarray([f"stations/{station_id}.nc" for station_id in stations], dtype=str)
    )
    for name in network.data_vars:
        network[name].attrs.update(_variable_attributes(name))
    triggers = diagnostic_trigger_summary(completed)
    if not triggers.empty:
        network = network.assign_coords(
            diagnostic=("diagnostic", triggers["diagnostic"].to_numpy(dtype=str))
        )
        network.diagnostic.attrs.update(
            long_name="Layer 1 diagnostic name",
            description="Enabled Layer 1 check aggregated across station-series.",
        )
        for name in (
            "enabled_series_count", "assessable_series_count",
            "unassessable_series_count", "tier1_count", "tier2_count",
            "concern_series_count", "trigger_rate_percent", "total_contribution",
        ):
            network[f"diagnostic_{name}"] = (
                "diagnostic", triggers[name].to_numpy()
            )
        network["diagnostic_trigger_rate_percent"].attrs.update(
            long_name="Layer 1 diagnostic trigger rate",
            units="%",
            description=(
                "Station-series with positive contribution divided by enabled, "
                "assessable station-series for this diagnostic."
            ),
        )
        diagnostic_descriptions = {
            "enabled_series_count": "Station-series for which this check was enabled.",
            "assessable_series_count": "Enabled station-series with sufficient evidence for this check.",
            "unassessable_series_count": "Enabled station-series lacking sufficient evidence for this check.",
            "tier1_count": "Assessable station-series classified Tier 1 for this check.",
            "tier2_count": "Assessable station-series classified Tier 2 for this check.",
            "concern_series_count": "Assessable station-series with a positive contribution from this check.",
            "total_contribution": "Sum of weighted Layer 1 contributions from this check.",
        }
        for name, description in diagnostic_descriptions.items():
            network[f"diagnostic_{name}"].attrs.update(
                long_name=name.replace("_", " ").capitalize(),
                units="count" if name != "total_contribution" else "points",
                description=description,
            )
    summary_rows = _summary_rows(completed)
    ranking = station_attention_ranking(summary_rows)
    if not ranking.empty:
        network = network.assign_coords(
            attention_record=("attention_record", np.arange(len(ranking), dtype=np.int32))
        )
        network.attention_record.attrs.update(
            long_name="Station-series attention record",
            description=(
                "One record per independently assessed series; positive ranks are "
                "ordered from greatest to least Layer 1 concern, while rank 0 means "
                "no Layer 1 attention is required."
            ),
        )
        attention_variables = {
            "attention_rank": ranking["attention_rank"].to_numpy(dtype=np.int32),
            "attention_station_id": ranking["station_id"].to_numpy(dtype=str),
            "attention_series_name": ranking["series_name"].to_numpy(dtype=str),
            "attention_series_role": ranking["series_role"].to_numpy(dtype=str),
            "attention_layer1_class": ranking["layer1_class"].to_numpy(dtype=str),
            "attention_layer1_score_percent": ranking["layer1_score_percent"].to_numpy(dtype=float),
        }
        for name, values in attention_variables.items():
            network[name] = ("attention_record", values)
        network["attention_rank"].attrs.update(
            long_name="Layer 1 attention rank",
            units="1",
            description=(
                "Needs-review series rank before minor-concern series; higher normalized "
                "scores rank first within each class. Rank 0 means no attention required."
            ),
        )
        network["attention_layer1_score_percent"].attrs.update(
            long_name="Normalized Layer 1 score",
            units="%",
            description="Existing Layer 1 score normalized over enabled, assessable checks.",
        )
        for name, label in (
            ("attention_station_id", "Station identifier"),
            ("attention_series_name", "Series name"),
            ("attention_series_role", "Series role"),
            ("attention_layer1_class", "Layer 1 classification"),
        ):
            network[name].attrs.update(long_name=label)
    counts = network_assessment_counts(summary_rows)
    network.attrs.update(
        title="TriHydrA network processing summary",
        software="TriHydrA",
        software_version=SOFTWARE_VERSION,
        created_utc=flat.attrs.get("created_utc", ""),
        Conventions="CF-1.10",
        review_definition=flat.attrs.get("review_definition", ""),
        requested_station_count=flat.attrs.get("requested_station_count", len(stations)),
        completed_station_count=flat.attrs.get("completed_station_count", 0),
        skipped_station_count=flat.attrs.get("skipped_station_count", 0),
        failed_station_count=flat.attrs.get("failed_station_count", 0),
        needs_review_station_count=flat.attrs.get("needs_review_station_count", 0),
        station_series_assessment_count=counts["station_series_assessment_count"],
        layer1_needs_review_station_count=counts["needs_review_station_count"],
        layer1_minor_concerns_station_count=counts["minor_concerns_station_count"],
        layer1_no_concerns_station_count=counts["no_concerns_station_count"],
        layer1_not_assessed_station_count=counts["not_assessed_station_count"],
        layer1_needs_review_series_count=counts["needs_review_series_count"],
        layer1_minor_concerns_series_count=counts["minor_concerns_series_count"],
        layer1_no_review_series_count=counts["no_review_series_count"],
        layer1_not_assessed_series_count=counts["not_assessed_series_count"],
        file_structure="Detailed results are stored in one NetCDF file per station under stations/.",
    )
    return network


def build_netcdf_dataset(
    results: Iterable[TriHydrAResult],
    manifest: pd.DataFrame,
    *,
    configuration: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Build the compact network index shown by default to batch users."""
    completed = list(results)
    flat = _build_flat_dataset(completed, manifest, configuration=configuration)
    return _build_network_dataset(flat, completed)


def write_netcdf_results(
    results: Iterable[TriHydrAResult], manifest: pd.DataFrame, output_path: str | Path,
    *, configuration: Mapping[str, Any] | None = None,
) -> Path:
    """Write one network index plus one grouped NetCDF file per station."""
    completed = list(results)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    flat = _build_flat_dataset(completed, manifest, configuration=configuration)
    network = _build_network_dataset(flat, completed)
    network.to_netcdf(destination, engine="netcdf4", encoding=_encoding(network))

    station_directory = destination.parent / "stations"
    station_directory.mkdir(parents=True, exist_ok=True)
    for station_id in np.asarray(flat.station.values, dtype=str):
        groups = _station_groups(flat, station_id, configuration)
        station_path = station_directory / f"{station_id}.nc"
        groups["root"].to_netcdf(
            station_path, engine="netcdf4", mode="w", encoding=_encoding(groups["root"])
        )
        for group_name, dataset in groups.items():
            if group_name == "root":
                continue
            dataset.to_netcdf(
                station_path, engine="netcdf4", group=group_name, mode="a",
                encoding=_encoding(dataset),
            )
    return destination


__all__ = ["build_netcdf_dataset", "write_netcdf_results"]
