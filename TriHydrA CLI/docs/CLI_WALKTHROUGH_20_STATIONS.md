# TriHydrA CLI walkthrough: first 20-station run

This walkthrough assumes you are using the supplied TriHydrA CLI folder
and have no previous command-line experience. The scientific settings are read
from `trihydra.toml`; nothing scientific is entered into the terminal.

## What this run will use

- Observations: `data/caravan_observations.nc`
- AIFL simulations: matching `data/<station_id>_results.p` files
- Station selection: `stations_20.txt`
- Configuration: `trihydra.toml`
- Results: `outputs/stations_20`

The station file contains 20 station IDs. Blank lines and lines beginning with
`#` are comments and are ignored.

## Part A — first-time installation

You do this section only once on a computer.

### 1. Open Anaconda Prompt

Open the Windows Start menu, type **Anaconda Prompt**, and open it.

### 2. Move into the TriHydrA Advanced folder

Find the folder where you extracted or cloned TriHydrA CLI. Replace the
example path below with that folder's actual location, keeping the quotation
marks:

```bat
cd /d "C:\path\to\TriHydrA-CLI"
```

Press **Enter**. The prompt should now end with `TriHydrA CLI>`.

### 3. Create the TriHydrA environment

```bat
conda env create -f environment.yml
```

Press **Enter** and wait for it to finish. This can take several minutes.

If Conda says the environment already exists, do not recreate it. Continue to
the next step.

### 4. Activate the environment

```bat
conda activate trihydra-cli
```

The beginning of the prompt should now show `(trihydra-cli)`.

### 5. Install the local TriHydrA command

```bat
python -m pip install -e .
```

The final dot is required: it means “install the package in this folder.”

### 6. Confirm that the CLI is available

```bat
trihydra --help
```

You should see a usage message containing the `run` command.

## Part B — check the 20-station configuration

### 7. Open `trihydra.toml`

Open `trihydra.toml` in a text editor. Under `[run]`, confirm that it contains:

```toml
[run]
station_file = "stations_20.txt"
all_stations = false
continue_on_station_error = true
```

Do not also enable `station_ids` or `all_stations = true`. Exactly one station
selection method may be active.

### 8. Confirm the enabled analyses

For the planned observation–AIFL run, use:

```toml
[layers]
layer1 = true
layer2 = true
layer3 = false
comparison = true
```

This runs Layer 1 and Layer 2 on every observation and AIFL series and compares
each available observation–AIFL pair. Layer 3 is off for this first run. To test
network context later, change only `layer3 = true`; the selected stations must
include suitable nearby/context gauges.

### 9. Confirm the input locations

The supplied data layout uses:

```toml
[observation]
path = "data/caravan_observations.nc"

[simulation]
enabled = true
path = "data"
trusted = true
```

Keep `trusted = true` only for the trusted mentor-provided pickle files.

### 10. Confirm the output settings

For an interactive first run:

```toml
[output]
directory = "outputs/stations_20"
html_mode = "ask"
non_interactive_html_mode = "review_only"
show_figures = false
write_text = true
write_netcdf = true
write_log = true
```

`html_mode = "ask"` means TriHydrA performs the calculations first and then
asks which HTML diagnostics you want.

Save and close `trihydra.toml`.

## Part C — run the 20 stations

### 11. Open Anaconda Prompt and enter the project folder

If you closed the prompt, reopen it and run:

```bat
cd /d "C:\path\to\TriHydrA-CLI"
conda activate trihydra-cli
```

### 12. Start TriHydrA

```bat
trihydra run --config trihydra.toml
```

Press **Enter** once. Do not close the terminal while it is processing.

### 13. Watch the progress messages

TriHydrA prints progress for each station. A station may be reported as:

- completed with observation and AIFL comparison;
- completed observation-only when no matching simulation is available;
- skipped because no valid observations were available; or
- failed, followed by an error reason.

With `continue_on_station_error = true`, one failed station does not prevent the
remaining stations from being processed.

### 14. Answer the HTML question

After calculations finish, select one of the offered HTML choices:

- **all** — create HTML diagnostics for every completed station;
- **review only** — create HTML only for stations requiring review;
- **none** — do not create HTML.

For this diagnostic experiment, choose **all** if disk space permits. Choose
**review only** for the smaller, practical output.

### 15. Wait for the completion message

The terminal prints the total elapsed time and output locations. A successful
run returns to the normal command prompt after all selected stations finish.

## Part D — inspect the results

Open `outputs/stations_20`. Important files are:

```text
outputs/stations_20/
├── trihydra_run.log
├── trihydra_results.nc
├── network_summary.txt
└── <station_id>/
    ├── summary.txt
    ├── layer1_evidence.txt
    ├── layer2_evidence.txt
    ├── comparison_evidence.txt
    └── HTML diagnostics when requested
```

Start with `network_summary.txt`, then open the `summary.txt` inside a station
folder. Use the evidence files and HTML diagnostics when investigating why a
station was classified a particular way.

`trihydra_results.nc` is the consolidated tabulated output for analysis across
all stations. `trihydra_run.log` records processing, warnings, failures, and
timings.

## Part E — rerunning safely

TriHydrA writes to the configured output directory. Before a separate
experiment, give it a different directory so results are not mixed:

```toml
[output]
directory = "outputs/another_run"
```

Then run the same CLI command again.

## Troubleshooting

### `trihydra` is not recognized

Confirm that `(trihydra-cli)` appears at the beginning of the prompt, then run:

```bat
python -m pip install -e .
```

As a fallback, use:

```bat
python -m trihydra.cli run --config trihydra.toml
```

### Configuration validation error

Read the reported field name. Common causes are a misspelled TOML key, more
than one active station-selection mode, or a path that does not exist.

### Simulation unavailable

TriHydrA could not find `data/<station_id>_results.p`. The observation can still
run, but there will be no observation–AIFL comparison for that station.

### Layer 3 not assessed

Layer 3 needs multiple selected observations plus matching rows in
`data/context.csv`. It does not automatically load stations omitted from the
station list.

### The command finishes with a non-zero exit status

At least one selected station was skipped or failed. Open
`outputs/stations_20/trihydra_run.log` to find the station and reason. Completed stations'
results remain available.
