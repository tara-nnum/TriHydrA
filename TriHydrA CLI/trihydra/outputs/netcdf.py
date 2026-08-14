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
    "layer1_class": {
        "description": "Final intrinsic time-series screening class derived strictly from the Layer 1 score.",
        "interpretation": "A review class requests inspection and does not prove that data are erroneous.",
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
            is_observation = row.get("series_role") == "observation"
            provenance = result.station.obs_provenance if is_observation else result.station.ml_provenance
            for name, value in _provenance_values(provenance).items():
                frame.at[index, name] = value
            series = result.station.obs if is_observation else result.station.ml
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
        "comparison_layer2_comparison_class": "comparison_class",
    })
    summary.columns = [
        name.replace("threshold_comparison_layer2_comparison_", "threshold_comparison_")
        for name in summary.columns
    ]
    return summary


def _layer1_components(result: TriHydrAResult, series_name: str) -> pd.DataFrame:
    if series_name == "observation":
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
            "input_path", "input_format", "input_variable", "processing_status",
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


def build_netcdf_dataset(
    results: Iterable[TriHydrAResult],
    manifest: pd.DataFrame,
    *,
    configuration: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Build one flat dataset without recalculating scientific diagnostics."""
    completed = list(results)
    summary = _summary_rows(completed)
    manifest = manifest.copy()
    stations = list(dict.fromkeys(manifest.get("station_id", pd.Series(dtype=str)).astype(str)))
    for station_id in summary.get("station_id", pd.Series(dtype=str)).astype(str):
        if station_id not in stations:
            stations.append(station_id)
    series_names = list(dict.fromkeys(summary.get("series_name", pd.Series(["observation"])).astype(str)))
    if "observation" in series_names:
        series_names.remove("observation")
        series_names.insert(0, "observation")
    if not series_names:
        series_names = ["observation"]

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


def write_netcdf_results(
    results: Iterable[TriHydrAResult], manifest: pd.DataFrame, output_path: str | Path,
    *, configuration: Mapping[str, Any] | None = None,
) -> Path:
    """Write one consolidated NetCDF file and return its resolved path."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_netcdf_dataset(results, manifest, configuration=configuration)
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name, variable in dataset.data_vars.items() if variable.dtype.kind in "fiu" and variable.size
    }
    dataset.to_netcdf(destination, engine="netcdf4", encoding=encoding)
    return destination


__all__ = ["build_netcdf_dataset", "write_netcdf_results"]
