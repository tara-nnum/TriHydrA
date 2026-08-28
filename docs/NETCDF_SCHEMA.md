# TriHydrA NetCDF output schema

## Output layout

TriHydrA writes two complementary products:

```text
<output>/
├── trihydra_network_summary.nc
└── stations/
    ├── <station_id>.nc
    └── ...
```

The network file answers **which stations ran and where are their results?**
Each station file answers **what did TriHydrA find for this station?**

## Series identity

TriHydrA never guesses whether a series is an observation or model output.
The TOML configuration supplies both pieces of identity:

```toml
[series1]
name = "CARAVAN"
role = "observation"

[series2]
name = "AIFL"
role = "simulation"
```

Inside a station file, `series_slot` is the stable coordinate (`series1`,
optionally `series2`). `series_name` and `series_role` are variables. This
supports observation-observation, observation-model, model-model and
historical-period comparisons without changing the schema.

## Network summary file

`trihydra_network_summary.nc` has one dimension, `station`. It intentionally
contains only the fields needed to scan or filter a batch:

- processing status and message;
- elapsed processing time;
- review-required status;
- primary series name, Layer 1 class and normalized score;
- comparison class;
- Layer 3 agreement class;
- number of flagged diagnostics;
- relative path to the detailed station file.

It also includes a `diagnostic` dimension containing the network-wide Layer 1
trigger summary. For every enabled check, the file records:

- enabled, assessable and unassessable station-series counts;
- Tier 1 and Tier 2 counts;
- the number and percentage of assessable series for which the check made a
  positive composite contribution;
- total weighted contribution across the batch.

These fields answer questions such as "which check caused most stations to be
flagged?" The denominator is the number of **assessable station-series** for
that check, not the number of physical stations. This matters when the same
station contains both an observation and a simulation.

Example:

```python
import xarray as xr

network = xr.open_dataset("trihydra_network_summary.nc")
print(network)
display(network.to_dataframe())

# Checks contributing most often, highest first
trigger_table = network[
    [
        "diagnostic_assessable_series_count",
        "diagnostic_tier1_count",
        "diagnostic_tier2_count",
        "diagnostic_concern_series_count",
        "diagnostic_trigger_rate_percent",
    ]
].to_dataframe().sort_values("diagnostic_concern_series_count", ascending=False)
display(trigger_table)
```

The file-level attributes separately report physical-station counts and
station-series counts. A station is counted once in the physical-station
totals even when both series are assessed; each series retains its own review
classification.

### Stations requiring most attention

The `attention_record` dimension provides one ranking record for each assessed
series. Its variables contain the station ID, series name and role, Layer 1
classification, normalized Layer 1 score and attention rank. Series classified
as `Needs review` are ranked before those with `Minor concerns`; within each
class, the higher normalized score appears first. Rank zero means that Layer 1
attention is not required. Observation and simulation results remain separate
records even when they belong to the same physical station.

## Station file root

One station file represents exactly one station, so it has no redundant
`station` dimension. Its root contains only the immediate summary:

- `series_slot` coordinate;
- `series_name`, `series_role`, `units`, and availability;
- processing status;
- review-required status;
- Layer 1 class, normalized score and assessment scope;
- comparison and Layer 3 classifications when assessed;
- flagged-diagnostic count and status.

Station ID, river, catchment, area, climate and software information are
file-level attributes when available. Latitude and longitude are scalar
coordinates because they locate the station; this is normal CF-style NetCDF
representation, not an extra station dimension.

## Station file groups

| Group | Purpose |
|---|---|
| `metadata` | Series identity, provenance, units and requested/selected timespans. |
| `layer1` | Intrinsic quality-control metrics and composite results. |
| `layer2` | Hydrological signatures and representative-event metrics. |
| `comparison` | Pairwise results for series1 versus series2. |
| `layer3` | Nearby-gauge and comparable-catchment context. |
| `flags` | Sparse table containing only diagnostics with positive contribution. |
| `thresholds` | Configured and station-derived thresholds actually used. |
| `evidence` | Additional retained evidence not represented by the summary groups. |
| `configuration` | Complete normalized run configuration as JSON metadata. |

Open a group explicitly with xarray:

```python
station_file = "stations/GRDC_3634220.nc"

root = xr.open_dataset(station_file)
layer1 = xr.open_dataset(station_file, group="layer1")
thresholds = xr.open_dataset(station_file, group="thresholds")
flags = xr.open_dataset(station_file, group="flags")
```

To list every available group:

```python
import netCDF4

with netCDF4.Dataset(station_file) as nc:
    print(list(nc.groups))
```

## Coordinates, variables and attributes

- **Coordinates** identify array axes or physical location: `station` in the
  network file, `series_slot` in a station file, and scalar latitude/longitude
  when available.
- **Variables** store results that vary by station, series or diagnostic
  record. Examples are missing percentage, mean discharge and flag name.
- **Attributes** explain variables or describe the complete file. Every major
  result includes readable metadata such as `long_name`, `description`,
  `units`, `method`, `threshold_definition` and `interpretation` where relevant.

## Thresholds

Thresholds are deliberately separated from diagnostic results. Open the
`thresholds` group to see both the values actually used and their definitions.
Station-derived thresholds remain variables because their values can differ
between series. Constant decision rules are also described through variable
attributes.

## Sparse flags

The `flags` group contains only checks with positive composite contribution.
Passing checks do not create rows. Its `diagnostic_record` dimension indexes:

- series name;
- diagnostic name;
- tier and raw value;
- readable summary;
- tier points, weight and weighted contribution.

This avoids filling the file with repetitive `flagged = false` rows while all
complete metrics remain available in the Layer 1 group.

## Missing and unavailable values

- Numeric unavailable values appear as `NaN` in xarray.
- Empty optional text means metadata was not supplied.
- A completed value of zero is distinct from a group or metric marked as not
  assessed.

## Why groups do not appear in `print(root)`

`xr.open_dataset(station_file)` opens only the root group. This is normal
NetCDF behavior. Open each named group explicitly as shown above. The root is
kept small precisely so a first-time user is not confronted by every detailed
metric at once.
