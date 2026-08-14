# TriHydrA CLI and configuration guide

## 1. What the CLI does

The command-line workflow reads and validates one TOML file, loads the selected
CARAVAN observations and available AIFL simulations, runs the enabled TriHydrA
layers, and writes the requested outputs.

The supplied `trihydra.toml` is the single canonical template. Its commented
examples demonstrate alternative modes but are not additional configurations.
Only uncommented values are read.

```text
trihydra run --config trihydra.toml
```

No scientific settings are entered interactively. The optional HTML question
is asked only after calculations finish and only when `html_mode = "ask"` in an
interactive terminal.

## 2. Installation

From Anaconda Prompt or a shell with Conda available:

```text
cd /d "path\to\TriHydrA CLI"
conda env create -f environment.yml
conda activate trihydra-cli
python -m pip install -e .
trihydra --help
```

`pip install -e .` registers the `trihydra` command while retaining this folder
as the editable source. Repeat it only if the environment is recreated.

## 3. Paths

Relative paths in TOML are resolved relative to the TOML file—not relative to
the terminal's current directory. Absolute paths are also accepted.

```toml
[observation]
path = "data/caravan_observations.nc"

[output]
directory = "outputs"
```

## 4. Station selection

Choose exactly one mode. Pydantic rejects ambiguous configurations.

### Inline station IDs

```toml
[run]
station_ids = ["GRDC_3634220", "GRDC_3637390"]
all_stations = false
continue_on_station_error = true
```

### Text file

```toml
[run]
station_file = "stations.example.txt"
all_stations = false
continue_on_station_error = true
```

The file contains one station ID per line. Blank lines and lines beginning with
`#` are ignored.

### Every NetCDF station

```toml
[run]
all_stations = true
continue_on_station_error = true
```

Remove or comment out `station_ids` and `station_file` when using this mode.

With `continue_on_station_error = true`, one station failure is logged and the
remaining stations continue. The CLI still returns a failure exit code at the
end. Set it to `false` to stop immediately.

## 5. Layer selection

```toml
[layers]
layer1 = true
layer2 = true
layer3 = false
comparison = true
```

- Layer 1: intrinsic data-quality screening.
- Layer 2: hydrological signatures and representative high-flow events.
- Comparison: observation versus available AIFL simulation.
- Layer 3: observation-network context; requires Layers 1 and 2.

Layer 3 compares only stations selected for the current run. Include the target
and possible context gauges in `station_ids` or the station file. Coordinates
and catchment metadata must be present in the configured context CSV.

## 6. Observation NetCDF

```toml
[observation]
name = "observation"
path = "data/caravan_observations.nc"
format = "netcdf"
variable = "streamflow"
station_coordinate = "basin"
time_coordinate = "date"
units = "mm/day"
```

Expected logical structure:

- a station coordinate containing station IDs;
- a time coordinate containing dates;
- one discharge variable indexed by station and time.

Each station is converted to an untouched `pandas.Series` before the scientific
pipeline. Change names to match the supplied NetCDF. Units are metadata and must
describe the actual values; TriHydrA does not silently convert units.

## 7. AIFL simulation pickles

```toml
[simulation]
enabled = true
format = "aifl_pickle"
path = "data"
model_name = "AIFL"
units = "mm/day"
trusted = true
time_step = 0
observation_variable = "streamflow_obs"
simulation_variable = "streamflow_sim"
```

When `path` is a directory, the reader looks for
`<station_id>_results.p`. The supported schema is the mentor-provided AIFL
long-term result structure. Only contemporaneous `time_step = 0` is used.

Observation and simulation units must match. A selected station without its own
pickle is logged as `simulation unavailable` and continues observation-only.
Malformed or untrusted files are not silently ignored.

To run observations only:

```toml
[simulation]
enabled = false
format = "aifl_pickle"
model_name = "AIFL"
units = "mm/day"
trusted = false
time_step = 0
observation_variable = "streamflow_obs"
simulation_variable = "streamflow_sim"
```

Also set `layers.comparison = false`.

## 8. Output controls

```toml
[output]
directory = "outputs"
html_mode = "ask"
non_interactive_html_mode = "review_only"
show_figures = false
write_text = true
write_netcdf = true
write_log = true
display_decimals = 3
```

HTML modes:

- `ask`: ask after calculations in an interactive terminal;
- `all`: write plots for every completed station;
- `review_only`: write plots only when a mandatory review classification exists;
- `none`: write no HTML.

On HPC, stdin may not be attached. `non_interactive_html_mode` is then used as
the safe fallback. For deterministic batch jobs, set `html_mode` directly to
`all`, `review_only`, or `none`.

`write_text`, `write_netcdf`, and `write_log` independently control their output
families. The output directory is created automatically.

## 9. Comparison settings

```toml
[comparison]
calculate_daily_metrics = false
include_provided_metrics = true
```

The default preserves and reports metrics supplied in the AIFL result. Daily
metrics are not recalculated unless explicitly enabled. Layer 1 and Layer 2 are
calculated on the full native observation and simulation records; pairwise
comparison alone uses the common span and pairwise-valid values.

## 10. Layer 3 metadata and peers

```toml
[layers]
layer3 = true

[layer3.metadata]
context_path = "data/context.csv"

[layer3.plotting]
mode = "recommended"
```

`layers.layer3` is the sole switch that enables or disables Layer 3. The
`[layer3.*]` tables configure its metadata, comparison, and plotting behaviour.

Required context columns:

```text
station_id, longitude, latitude, river_name, catchment_name,
catchment_area_km2, series_type
```

Nearby gauges are searched within the configured 50 km default. Comparable
catchments use climate and catchment-area context within the configured maximum
radius. A nearby observation gauge does not need an AIFL pickle.

Layer 3 plotting mode is `recommended`, `all`, or `none`. This controls Layer 3
dashboard eligibility after the general HTML mode has selected a station.

## 11. Threshold configuration

The supplied `trihydra.toml` contains the complete supported Layer 1, Layer 2
and Layer 3 configuration with comments. Important rules include:

- missingness and long-gap tiers;
- negative-flow tolerance;
- plateau and spike/dip definitions;
- step-shift and epoch-drift tiers and weights;
- Q95 event trigger and Q90 event boundaries;
- comparison similarity and timing tiers;
- Layer 3 search radii, agreement rules and context weights.

Unknown TOML keys are rejected. This intentionally catches misspelled threshold
names instead of silently falling back to defaults. Pydantic also rejects
inconsistent ranges, mismatched units, comparison without simulation, and
Layer 3 without Layers 1 and 2.

## 12. Outputs

### Station folders

Depending on enabled layers and available simulation data:

```text
<output>/<station_id>/summary.txt
<output>/<station_id>/layer1_evidence.txt
<output>/<station_id>/layer2_evidence.txt
<output>/<station_id>/comparison_evidence.txt
<output>/<station_id>/layer3_evidence.txt
<output>/<station_id>/*.html
<output>/network_summary.txt
<output>/trihydra_results.nc
<output>/trihydra_run.log
```

`network_summary.txt` is written once per multi-station batch when text output
is enabled. It provides a compact station-series index before users open the
individual station reports.

### Consolidated NetCDF

`trihydra_results.nc` is one flat, self-documenting xarray-compatible dataset.
It contains station and series coordinates, provenance, units, Layer 1/2/3 and
comparison summaries, thresholds used, processing status and sparse records for
flagged Layer 1 diagnostics. Clean checks do not create redundant flag records.
See `NETCDF_SCHEMA.md`.

### Log and terminal progress

The CLI prints when each station starts, completes, is skipped, continues
observation-only, or fails. A station with no valid dated streamflow is retained
in the manifest as `skipped` with the reason `NoValidObservations`. It prints
Layer 3 status, output locations, and total elapsed time.
The same messages are written to `trihydra_run.log` when enabled.

The process exits with code `0` only when every manifest row completed. An
empty run, skipped station, or failed station produces a nonzero exit code so
an HPC scheduler cannot mistake an incomplete assessment for success.

## 13. HPC example

Use a non-interactive HTML mode:

```toml
[output]
directory = "outputs/hpc_run"
html_mode = "review_only"
non_interactive_html_mode = "review_only"
show_figures = false
write_text = false
write_netcdf = true
write_log = true
display_decimals = 3
```

Then run from the job script:

```text
conda run -n trihydra-cli trihydra run --config /absolute/path/trihydra.toml
```

Check the process exit code and `trihydra_run.log`.

## 14. Troubleshooting

- **Unknown field:** correct the TOML spelling; unsupported keys are rejected.
- **Observation path does not exist:** paths are relative to the TOML file.
- **Pickle trust error:** set `trusted=true` only for verified supplied files.
- **Simulation unavailable:** the station continues observation-only; add the
  matching `<station_id>_results.p` if a comparison was expected.
- **Layer 3 not assessed:** select at least two stations and confirm matching
  metadata and coordinates in `context.csv`.
- **No HTML on HPC:** check both `html_mode` and
  `non_interactive_html_mode`.
- **Binary-library errors:** recreate the Conda environment from
  `environment.yml`; do not mix incompatible NumPy/NetCDF installations.
