# TriHydrA configuration manual

This manual explains the root `trihydra.toml` file used by TriHydrA. It is
written for both first-time users and advanced users who need to change the
scientific assessment policy.

TriHydrA uses the same TOML file from the command line, Python, and Jupyter:

```text
trihydra run --config trihydra.toml
```

```python
from trihydra import run_batch

batch = run_batch("trihydra.toml")
```

Both commands use the same readers, calculations, station selection, and
output writers. The only difference is that Python keeps the completed result
objects available for further exploration.

## 1. What is a TOML file?

TOML is a human-readable configuration format. A heading in square brackets
starts a table of related settings:

```toml
[output]
write_text = true
write_netcdf = true
```

The basic syntax is:

| Value | TOML example |
|---|---|
| On/off switch | `enabled = true` |
| Text | `name = "observation"` |
| Whole number | `maximum_peers = 5` |
| Decimal number | `tolerance = 0.001` |
| List | `station_ids = ["A", "B"]` |
| Date | `start_date = 1980-01-01` |
| Comment | `# this line is ignored` |

Use lowercase `true` and `false`. Keep quotation marks around text and paths.
Unknown or misspelled fields are rejected instead of being silently ignored.

Relative paths are resolved from the folder containing the TOML file—not from
the terminal's current folder. For example, if `trihydra.toml` is in the
project root, `data/input.nc` means the `data` folder beside that TOML file.

## 2. What most users need to edit

The TOML is ordered so the everyday controls come first. For a normal run,
check these five sections:

1. `[run]`: which stations should be processed?
2. `[layers]`: which pipeline components should run?
3. `[output]`: what should be saved, and where?
4. `[series1]`: what is the required primary input?
5. `[series2]`: is there an optional second input?

The remaining sections define scientific thresholds and comparison policies.
Their supplied defaults already make a complete working configuration.

## 3. Section 1 — station selection

TriHydrA must know which station IDs to extract. Under `[run]`, choose exactly
one of the following modes.

### 3.1 Inline station IDs

Use this for one station or a short list:

```toml
[run]
station_ids = ["GRDC_3634220", "GRDC_3637390"]
all_stations = false
continue_on_station_error = true
```

The spelling must match the input source exactly.

### 3.2 Station-list file

Use this for a reusable longer list. Comment out `station_ids`, then add:

```toml
[run]
station_file = "stations.txt"
all_stations = false
continue_on_station_error = true
```

The text file contains one ID per line:

```text
# German test gauges
GRDC_3634220
GRDC_3637390
GRDC_3638120
```

Blank lines and lines beginning with `#` are ignored.

### 3.3 Every available Series 1 station

Comment out both `station_ids` and `station_file`, then use:

```toml
[run]
all_stations = true
continue_on_station_error = true
```

“All stations” means every station discoverable from `series1`. A station that
does not exist in Series 2 can still continue as a Series 1-only assessment.

### 3.4 Error handling

`continue_on_station_error = true` records the failed station and continues
the batch. This is recommended for large jobs. When it is `false`, the first
station error stops the run.

TriHydrA rejects configurations that activate zero or multiple selection
modes because the intended station set would be ambiguous.

## 4. Section 2 — pipeline components

```toml
[layers]
layer1 = true
layer2 = true
layer3 = false
comparison = false
```

| Field | Meaning | Dependency |
|---|---|---|
| `layer1` | Intrinsic time-series checks | None |
| `layer2` | Hydrological signatures and high-flow events | None for descriptive signatures |
| `layer3` | Network/context assessment | Requires Layers 1 and 2, multiple loaded observations, and context metadata |
| `comparison` | Series 1 versus Series 2 assessment | Requires `series2.enabled = true` |

Layer 1 and Layer 2 assess each available series independently. `comparison`
compares the two configured series. Layer 3 currently uses observations loaded
in the same run; it does not silently load nearby stations omitted from the
station selection.

## 5. Section 3 — outputs

```toml
[output]
directory = "outputs/example_run"
write_text = true
write_netcdf = true
write_log = true
html_mode = "needs_review"
non_interactive_html_mode = "needs_review"
show_figures = false
```

### `directory`

The folder created for this run. Use a new descriptive folder for each manual
experiment, for example:

```toml
directory = "outputs/manual_01_single_baseline"
```

Avoid reusing a folder when you need a clean comparison between runs.

### `write_text`

When `true`, TriHydrA writes readable summaries and evidence. TXT is intended
for quick human inspection.

`station_assessment_status.txt` is always written, even when
`write_text = false`. This compact, tab-delimited run index contains one row
per station, with separate Series 1 and Series 2 assessment columns when both
are available. It also reports processing and station-series assessment totals
at the top and can be opened in a text editor or imported into a spreadsheet.

The `FILES AVAILABLE` section of each station summary is generated from the
files actually written for that run. Disabled or inapplicable outputs are not
listed.

### `write_netcdf`

When `true`, TriHydrA writes:

```text
<output>/trihydra_network_summary.nc
<output>/stations/<station_id>.nc
```

The network file is a compact batch index. Each station file contains grouped
metadata, Layer 1, Layer 2, comparison, Layer 3, flags, thresholds, evidence,
and configuration. Disabled or unavailable assessments are identified rather
than fabricated. See `NETCDF_SCHEMA.md` for the complete architecture.

### `write_log`

When `true`, the run log records progress, warnings, skipped/failed stations,
output paths, and elapsed time. Logs are especially useful for batch and HPC
runs.

### `html_mode`

Controls saved interactive Plotly reports:

| Value | Behaviour |
|---|---|
| `"all"` | Write every applicable report |
| `"concerns_and_review"` | Write reports for Minor concerns and Needs review |
| `"needs_review"` | Write reports only for Needs review |
| `"none"` | Do not write HTML |
| `"ask"` | Ask the terminal user after calculations |

Use `"all"` during visual validation. Use `"concerns_and_review"` when minor
concerns also need inspection, or `"needs_review"` for the smallest routine
plot set. Use `"none"` for NetCDF/log-only HPC jobs.

### `non_interactive_html_mode`

Used only when `html_mode = "ask"` but no interactive terminal is attached.
For unattended execution, set this explicitly to `"all"`,
`"concerns_and_review"`, `"needs_review"`, or `"none"`.

### `show_figures`

Controls whether figures open during execution. It does **not** control whether
HTML is saved. Keep it `false` for normal or batch runs.

## 6. Section 4 — required Series 1 input

Series 1 is the required primary/reference record. “Primary” does not mean it
must be an observation. Its meaning is declared with `name` and `role`.

```toml
[series1]
name = "observation"
role = "observation"
units = "mm/day"
format = "netcdf"
path = "data/caravan_observations.nc"
variable = "streamflow"
station_coordinate = "basin"
time_coordinate = "date"
```

### Common Series 1 fields

| Field | Meaning |
|---|---|
| `name` | Human-readable dataset/model name saved in outputs |
| `role` | Scientific role: `observation`, `simulation`, `historical_observation`, or `other` |
| `units` | Discharge units already used by the source |
| `format` | `netcdf`, `csv`, or `aifl_pickle` |
| `path` | File or supported directory location |

TriHydrA does not silently convert discharge units. When comparison is enabled,
Series 1 and Series 2 must already use compatible units.

### 6.1 NetCDF input

NetCDF requires:

```toml
format = "netcdf"
path = "data/input.nc"
variable = "streamflow"
station_coordinate = "basin"
time_coordinate = "date"
```

The logical dataset must contain:

- a station coordinate;
- a time coordinate;
- a discharge variable indexed by station and time.

The field names are configurable, so they do not have to be `basin`, `date`,
and `streamflow`. Files readable through xarray/netCDF4 or h5netcdf are
supported, including common `.nc`, `.nc4`, and `.cdf` filenames.

Optional engine selection:

```toml
engine = "auto"
```

Allowed values are `auto`, `netcdf4`, and `h5netcdf`. `auto` is recommended
unless a particular backend is required.

### 6.2 Wide CSV input

```toml
format = "csv"
path = "data/discharge.csv"
date_column = "date"
```

Expected shape:

```text
date,station_A,station_B
2000-01-01,1.20,0.44
2000-01-02,1.35,
```

The date column is parsed as time. Every other selected column is treated as
one station. Blank cells become missing observations. Remove the NetCDF-only
fields (`variable`, `station_coordinate`, and `time_coordinate`) when using
CSV so the active configuration remains clear.

### 6.3 Trusted AIFL pickle input

```toml
format = "aifl_pickle"
path = "data"
trusted = true
time_step = 0
observation_variable = "streamflow_obs"
simulation_variable = "streamflow_sim"
```

The path may be one result file or a directory containing files named:

```text
<station_id>_results.p
```

This adapter supports the supplied AIFL long-term-result schema, not arbitrary
pickle objects. Pickles can execute code while loading. Set `trusted = true`
only for files from a verified source. The supported contemporaneous long-term
series uses `time_step = 0`; negative lead/history coordinates are not treated
as separate forecast models.

## 7. Timespan selection

Each series has its own timespan block.

Full available record:

```toml
[series1.timespan]
mode = "full"
```

Inclusive date range:

```toml
[series1.timespan]
mode = "range"
start_date = 1980-01-01
end_date = 2009-12-31
```

Range mode requires both dates, and the start cannot be after the end. Layer 1
and Layer 2 assess only the selected timespan. The raw source file is never
rewritten.

## 8. Section 5 — optional Series 2 input

```toml
[series2]
enabled = false
name = "series2"
role = "simulation"
units = "mm/day"
format = "netcdf"
```

When `enabled = false`, Series 2 is not loaded. To compare two records:

1. set `series2.enabled = true`;
2. provide all fields required by its format;
3. give it a name different from Series 1;
4. set `[layers].comparison = true`;
5. confirm compatible units.

Series 2 supports the same NetCDF, wide CSV, and trusted AIFL formats as
Series 1. If one selected station is missing from Series 2, it can continue as
a Series 1-only assessment when station-error handling permits it.

The neutral two-slot design supports:

- observation versus simulation;
- observation versus observation;
- simulation versus simulation;
- recent observations versus historical observations.

TriHydrA knows what each slot represents from `name` and `role`; it does not
guess from the filename.

## 9. Section 6 — comparison mode

```toml
[comparison]
mode = "paired_overlap"
calculate_daily_metrics = false
include_provided_metrics = true
```

### `mode = "paired_overlap"`

Use when the records share calendar dates. Hydrological signatures are
calculated on the same outer start/end support, while optional daily paired
metrics use dates where both values are valid. Internal missing observations
in one series are not deleted from the other series' independent Layer 1/2
assessment.

### `mode = "independent_timespans"`

Use for historical-period comparison or records without shared dates. TriHydrA
compares distributions, signatures, seasonality, and event behaviour without
daily pairing. Daily paired metrics must remain disabled.

### Daily and supplied metrics

`calculate_daily_metrics = true` calculates supported paired daily performance
metrics. The default is `false` because supplied model-result metrics are
preferred when available.

`include_provided_metrics = true` includes performance metrics already stored
in a supported model result. If none are supplied, TriHydrA simply reports
them as unavailable unless calculation was requested.

## 10. Section 7 — Layer 1 switches

Each Layer 1 check has its own table and an `enabled` switch. Change only that
switch when deciding whether the check should run.

Disabled checks:

- are not calculated;
- are not labelled clean;
- do not contribute to the Layer 1 composite denominator.

Therefore, a deliberate run containing only step-shift detection uses step
shift as its sole assessable composite component and reports reduced evidence
coverage transparently.

### `layer1.missing_values`

Reports internal missingness using expected timestamps between the first and
last record. Calculations use valid observations; the raw record remains
unchanged.

### `layer1.long_gaps`

`minimum_reported_gap_days` controls which gaps appear in detailed evidence.
It does not define every composite tier; those rules are in
`layer1.composite.long_gaps`.

### `layer1.negative_discharge`

`tolerance` ignores tiny negative numerical noise near zero. Material negative
flows are judged relative to the station's low-flow reference.

### `layer1.duplicate_timestamps`

Reports repeated dates/rows. Duplicate timestamps remain evidence about the
input and are not silently erased from the raw record.

### `layer1.timestep_consistency`

Checks ordering and irregular unique-date intervals. Long missing spans and
irregular timestamps are distinct concepts and are reported separately.

### `layer1.zero_flow_regime`

Describes zero-flow observations and spells using valid observations only.
Zero flow is a hydrological descriptor, not automatically an error flag.
`decimals` defines the rounding used to identify zero.

### `layer1.low_variability`

Screens persistent non-zero plateaus/flatlines. `minimum_plateau_days` is the
minimum candidate duration; composite severity begins at separately configured
durations.

### `layer1.spike_dip`

Uses robust local change and recovery evidence. The robust MAD and quantile
parameters describe what counts as unusual relative to normal station change.
Candidates are propagated to Layer 2 so possible peaks and artefacts can be
cross-checked rather than treated independently.

### `layer1.step_shift`

Screens persistent changes in regime median using adaptive blocks. Candidate
magnitudes are interpreted against station low-flow percentiles. The tier-point
average across retained boundaries determines the dominant station-level
step-shift tier.

The supplied block, coverage, refinement, quantile, and tier fields form one
scientific method. Change them only with a documented reason.

### `layer1.epoch_drift`

Summarizes long-term stable, rising, and falling behaviour using valid annual
levels grouped into configured epochs. `epoch_years = 5` means five-year
assessment blocks. The final tier depends on the fraction of assessed years
classified as stable.

## 11. Layer 1 composite policy

Layer 1 assigns each assessable check a severity tier:

| Tier | Default points | Interpretation |
|---|---:|---|
| Tier 3 | 0 | no material concern |
| Tier 2 | 1 | minor concern |
| Tier 1 | 2 | strongest concern |

Each check's tier points are multiplied by its configured weight. The final
classification uses the percentage of the maximum possible score from only
the checks that are both enabled and assessable:

```toml
[layer1.composite.classification]
minor_concerns_minimum_percent = 7.5
needs_review_minimum_percent = 20.0
```

This normalization is why disabling checks does not automatically make the
station look safer. Raw additive score, normalized percentage, contributing
checks, and evidence coverage are retained in outputs.

The missingness, long-gap, negative-flow, plateau, and spike/dip subsections
define their tier boundaries. The weights subsection defines relative
influence. These settings describe an organization's review policy and should
be changed consistently, not one number at random.

## 12. Section 8 — Layer 2 settings

Layer 2 is descriptive for one series and comparative when two series are
available.

### Annual signatures

`minimum_valid_days_per_year` and `minimum_valid_days_per_month` define minimum
support. Baseflow uses the Lyne-Hollick filter parameters:

- `baseflow_alpha`;
- `baseflow_passes`;
- `minimum_baseflow_segment_days`.

Missing observations are temporarily excluded from a calculation; values are
not imputed into the raw record.

### High-flow events

```toml
trigger_percentile = 0.95
boundary_percentile = 0.90
```

Events cross the record's Q95 percentile and retain shoulders above Q90.
`spike_crosscheck_minimum_event_duration_days` prevents a one-day exceedance
from automatically explaining away the spike that created it. Layer 1
spike/dip candidates are excluded where intended from trusted representative
event and annual-extreme calculations, with overlap evidence retained.

### Comparison thresholds

Inverse Jensen-Shannon divergence and cosine similarity are oriented so larger
values mean greater similarity. The shared defaults are:

```toml
similarity_tier3_minimum = 0.80
similarity_tier2_minimum = 0.50
```

Seasonal timing uses circular month distance. Event time-to-peak and duration
use differences in days. `minimum_assessable_components` prevents a final
classification based on too little evidence.

The final normalized dissimilarity classification uses:

```toml
similar_maximum_percent = 12.5
review_maximum_percent = 43.75
```

Lower percentages mean closer agreement.

### Comparison component switches

Under `[layer2.comparison.components]`, set any unwanted comparison to `false`.
Disabled components do not contribute to the score or its denominator.

Weights are configured separately. A weight controls relative influence among
enabled, assessable components; it does not enable the component by itself.

## 13. Section 9 — Layer 3 network context

Layer 3 asks whether the primary observation is hydrologically consistent with
other observations loaded in the same run. It is optional.

### Metadata file

```toml
[layer3.metadata]
context_path = "data/context.csv"
```

Expected columns are:

```text
station_id, longitude, latitude, river_name, catchment_name,
catchment_area_km2, series_type
```

Rows may be missing for some stations; those stations can still complete
Layers 1 and 2 but cannot receive complete context. Coordinates are required
for peer search. The bundled Köppen-Geiger resource supplies climate context.

### Nearby gauges

Local peers are searched within `maximum_search_radius_km` (default 50 km),
with preferences for the same catchment and river. `minimum_peers` defines
whether local context can be assessed, and `maximum_peers` limits workload.

### Comparable catchments

These peers may be farther away but must satisfy the configured climate and
catchment-area rules. `maximum_search_radius_km = 1000.0` is a safety ceiling,
not a claim that a gauge 1000 km away is locally representative.

### Context comparison

Local and comparable-catchment evidence have configurable weights. Timing
checks use tolerances in days. Shape/signature checks use similarity rules.
Peer consensus and final agreement boundaries determine low, partial, or
strong contextual agreement.

`report_minimum_similarity_percent` controls whether the detailed context
report is recommended. `[layer3.plotting].mode` may be `recommended`, `all`,
or `none`.

Layer 3 agreement is corroborative evidence, not proof that either series is
correct.

## 14. Common configuration recipes

### Observation only

```toml
[layers]
layer1 = true
layer2 = true
layer3 = false
comparison = false

[series2]
enabled = false
```

### Simulation only

Configure Series 1 with `role = "simulation"`; keep Series 2 and comparison
disabled. Layer 1 and Layer 2 then assess the simulation independently.

### Observation versus model

```toml
[layers]
layer1 = true
layer2 = true
layer3 = false
comparison = true

[series1]
name = "CARAVAN"
role = "observation"

[series2]
enabled = true
name = "AIFL"
role = "simulation"
```

Complete the appropriate format-specific fields and use compatible units.

### Recent period versus historical period

Point both slots to the same source and station, give them different names,
configure non-overlapping date ranges, and use:

```toml
[comparison]
mode = "independent_timespans"
calculate_daily_metrics = false
```

### Layer 3 context run

Select multiple observations, enable Layers 1-3, ensure every usable station
has a context row, and choose whether reports are `recommended` or `all`.

### One Layer 1 check only

Keep Layer 1 enabled, disable every individual Layer 1 check except the one of
interest, and optionally disable Layer 2. Outputs will state that the composite
is based only on the enabled, assessable evidence.

### NetCDF-only HPC run

```toml
[output]
write_text = false
write_netcdf = true
write_log = true
html_mode = "none"
non_interactive_html_mode = "none"
show_figures = false
```

## 15. Validation before a long run

Validate the TOML without processing stations:

```text
python -c "from trihydra.settings.loader import load_toml_config; load_toml_config('trihydra.toml'); print('Configuration valid')"
```

TriHydrA rejects, among other things:

- ambiguous station selection;
- missing input paths;
- incomplete NetCDF field definitions;
- untrusted pickle loading;
- comparison without Series 2;
- mismatched comparison units;
- incomplete or reversed date ranges;
- invalid threshold ordering;
- Layer 3 without its required layers or metadata;
- unknown/misspelled fields.

Read the final lines of a validation error first. Pydantic normally reports the
exact table and field that caused it.

## 16. Running after validation

From the project directory:

```text
conda activate trihydra
trihydra run --config trihydra.toml
```

Equivalent module form:

```text
python -m trihydra run --config trihydra.toml
```

For Python/Jupyter:

```python
from trihydra import run_batch

batch = run_batch("trihydra.toml")
batch.manifest
batch.summary
batch.station_results
```

For an HPC scheduler, use a deterministic HTML mode and, when convenient, an
absolute TOML path:

```text
conda run -n trihydra trihydra run --config /path/to/trihydra.toml
```

## 17. Troubleshooting checklist

### “Choose exactly one station mode”

More than one of `station_ids`, `station_file`, and `all_stations=true` is
active, or none is active.

### “Path does not exist”

Remember that a relative path starts from the TOML file's folder. Check the
spelling and whether the expected input is a file or directory.

### Comparison is not assessed

Confirm:

- `[layers].comparison = true`;
- `[series2].enabled = true`;
- the station exists in both sources;
- the units match;
- enough components are assessable;
- the selected comparison mode suits the timespans.

### Layer 3 is not assessed

Confirm:

- Layers 1, 2, and 3 are enabled;
- multiple observations are selected and loaded;
- matching context metadata exists;
- peer-selection rules find sufficient candidates.

### HTML is missing

Check both `[output].html_mode` and `[layer3.plotting].mode`. With
`html_mode="needs_review"`, a clean station may intentionally have no detailed
report. Use `html_mode="all"` while manually inspecting visuals.

### Binary NumPy/NetCDF import errors

Recreate the supplied Conda environment rather than mixing binary packages
from unrelated environments.

## 18. Reproducibility record

For an important run, preserve:

- the exact `trihydra.toml` used;
- the run log;
- the TriHydrA version;
- input dataset names and units;
- station-selection file, if used;
- generated network and station NetCDF files.

The grouped station NetCDF also stores the effective configuration and
threshold information, allowing a later reader to understand how the result
was produced.

---

For output-group details, see `NETCDF_SCHEMA.md`. Python and Jupyter users can
call `run_batch("trihydra.toml")` for the same configured workflow used by the
CLI, or call the prepared-series API documented in the root `README.txt`.
