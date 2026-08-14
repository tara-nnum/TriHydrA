# TriHydrA NetCDF output schema

## Design goal

`trihydra_results.nc` is the compact, tabulated output for analysis across many
stations. It is intentionally a **single root-level NetCDF dataset**. It does
not use NetCDF groups, so the useful values are visible immediately with:

```python
import xarray as xr

results = xr.open_dataset("trihydra_results.nc")
print(results)
print(results[["overall_review_class", "layer1_score"]].to_dataframe())
```

Detailed event and anomaly records remain in the optional human-readable TXT
evidence reports. HTML diagnostics also remain separate files. Neither belongs
inside a tabulated NetCDF summary.

## Dimensions and coordinates

| Name | Meaning |
|---|---|
| `station` | One row per requested gauge/station. |
| `series` | Input series role, initially `observation` and optionally the configured model name (for example `AIFL`). |

Coordinates:

- `station_id(station)` is the stable station identifier.
- `series_name(series)` is the readable input-series name.

Variables that describe one assessment use `(station, series)`. Variables that
describe a comparison or Layer 3 context use `(station)` because they summarize
the relationship for the target station.

## Metadata and provenance

The following variables make every station/series row traceable:

- `series_available(station, series)`
- `series_role(station, series)`
- `units(station, series)`
- `input_dataset(station, series)`
- `input_format(station, series)`
- `input_variable(station, series)`
- `record_start(station, series)`
- `record_end(station, series)`
- `latitude(station)` and `longitude(station)`, when supplied
- `river_name(station)`, `catchment_name(station)`, and
  `catchment_area_km2(station)`, when supplied

Paths are recorded as provenance but are not required to be portable. Dataset
and model names remain the primary human-readable identifiers.

## Processing and final summary

The first variables a user should inspect are:

- `processing_status(station)`: `completed`, `skipped`, or `failed`
- `processing_message(station)`
- `overall_review_class(station, series)`
- `overall_review_code(station, series)`
- `primary_concerns(station, series)`
- `comparison_class(station)`
- `layer3_agreement_class(station)`

Readable classifications are stored as strings. Companion integer codes are
included only where filtering is useful, and define `flag_values` and
`flag_meanings` attributes.

Global attributes provide the requested network summary:

- `requested_station_count`
- `completed_station_count`
- `skipped_station_count`
- `failed_station_count`
- `needs_review_count`
- `minor_concerns_count`
- `no_review_count`

## Layer 1 variables

All use `(station, series)`:

- missing observation count and percentage
- long-gap count and longest gap
- duplicate timestamp and extra-row counts
- irregular timestep count and out-of-order status
- negative-discharge count and largest negative magnitude
- zero-flow percentage, spell count, and longest spell
- non-zero plateau count and longest plateau
- spike, dip, and unresolved candidate counts
- step-shift candidate count, score, and tier
- epoch regime, stable fraction, and tier
- Layer 1 score, class, and primary concerns

### Flagged-only diagnostics table

Layer 1 checks that contribute concern points are also exposed as a sparse
table on the `flag_record` dimension:

- `flagged_station_id(flag_record)`
- `flagged_series_name(flag_record)`
- `flagged_diagnostic_name(flag_record)`
- `flagged_diagnostic_tier(flag_record)`
- `flagged_diagnostic_value(flag_record)`
- `flagged_diagnostic_tier_points(flag_record)`
- `flagged_diagnostic_weight(flag_record)`
- `flagged_diagnostic_contribution(flag_record)`
- `flagged_diagnostic_summary(flag_record)`

Passing checks do not create rows. `flagged_diagnostic_count(station, series)`
and `diagnostic_status(station, series)` report either `Clean`,
`Concerns found`, or `Not assessed` without filling the table with redundant
"not flagged" entries.

## Layer 2 variables

All use `(station, series)`:

- valid observation count
- mean, median, minimum, maximum, Q05 low flow, and Q95 high flow
- number of annual-signature years and high-flow events
- median annual flashiness, baseflow, seasonality, and lag-1 autocorrelation
- typical wettest and driest months
- representative-event start, peak, and end dates
- representative-event peak flow, time to peak, recession time, duration,
  rising slope, and recession slope
- spike/peak overlap count and number of excluded flagged extrema

## Comparison variables

All use `(station)` and are missing when no second series was supplied:

- reference and candidate series names
- common start and end dates
- common timestamp count and pairwise-valid count
- Layer 2 comparison score and class
- assessable component count
- incomplete-assessment status and unavailable components

Supplied AIFL model metrics are stored with a stable `provided_metric`
dimension only when at least one input contains them. They are explicitly
labelled as supplied by the input dataset, not recalculated by TriHydrA.

## Layer 3 variables

All use `(station)` and are missing when Layer 3 was not run:

- overall contextual-agreement class
- nearby-gauge agreement class
- comparable-catchment agreement class
- numbers of nearby and comparable gauges assessed
- evidence coverage

The exact Layer 3 metric names will be taken from the existing Layer 3
`summary_metrics` contract during integration; the writer will not recompute
scientific results.

## Threshold documentation

Threshold information follows two rules:

1. A constant rule is documented on the affected variable using attributes:
   `long_name`, `description`, `units`, `layer`, `method`,
   `threshold_definition`, and `interpretation`.
2. A threshold calculated separately for each station/series is stored as a
   real `(station, series)` data variable ending in `_threshold_used`. It is
   never hidden in a global attribute.

This distinction keeps the file self-documenting without incorrectly treating
station-specific thresholds as constants.

## Missing and unavailable values

- Numeric unavailable values use NetCDF fill values and appear as `NaN` in
  xarray.
- Integer counts use nullable/fill-value encoding.
- Text uses an empty string only for absent optional metadata.
- Every layer has an explicit `*_assessed` Boolean variable so `0` is never
  confused with `not assessed`.

## File-level attributes

The root dataset contains:

- `title`: TriHydrA streamflow plausibility assessment
- `software`: TriHydrA
- `software_version`
- `created_utc`
- `Conventions`: CF-1.10
- `input_datasets`
- `assessment_layers`
- `review_definition`
- `threshold_information`
- `history`

The configuration used for the run is recorded in normalized JSON as a global
attribute for reproducibility. Large raw data and HTML are never embedded.
