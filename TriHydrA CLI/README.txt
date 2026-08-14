================================================================================
                    TRIHYDRA ADVANCED / ECMWF WORKFLOW
================================================================================

This package runs TriHydrA as a validated, TOML-configured command-line
workflow. It is intended for repeatable multi-station and HPC processing.

Supported inputs in this release
--------------------------------

- CARAVAN-style station-by-time NetCDF observations.
- Trusted AIFL long-term result pickles named <station_id>_results.p.

As of currently, Zarr, GRIB and arbitrary pickle schemas are deliberately 
outside the supported interface. TriHydrA analyses long time series only. 

FIRST-TIME INSTALLATION
-----------------------

Open Anaconda Prompt, change into the working folder, then run:

    conda env create -f environment.yml
    conda activate trihydra-cli
    python -m pip install -e .

The final command installs the `trihydra` command from this folder. Confirm it:

    trihydra --help
    trihydra run --help

QUICK START
-----------

1. Put the CARAVAN NetCDF and trusted AIFL pickle files in `data/`, or set
   their actual locations in `trihydra.toml`.
2. Edit `trihydra.toml`.
3. Choose exactly one station-selection mode under `[run]`.
4. For Layer 3, include the target and its possible context gauges in the run;
   Layer 3 does not automatically load unselected stations.
5. Run:

       trihydra run --config trihydra.toml

Development equivalent, if the command has not been installed:

       python -m trihydra.cli run --config trihydra.toml

The terminal reports each completed, observation-only, or failed station and
the total elapsed time. The process exits with a non-zero status if any selected
station fails.

CONFIGURATION
-------------

`trihydra.toml` is the single canonical editable template. It contains compact
commented examples beside station selection, simulation, HTML output and
Layer 3 settings. The examples are comments only; the uncommented values are
the configuration that will actually run.

The complete reference is:

    docs/CLI_CONFIGURATION_GUIDE.md

IMPORTANT BEHAVIOUR
-------------------

- Layer 1 and Layer 2 use each series' full native record.
- Observation-model comparison uses their common date span and pairwise-valid
  values only.
- A selected gauge without a matching AIFL pickle continues observation-only.
  This allows observation-only gauges to support Layer 3 context.
- Pickles can execute code. Set `simulation.trusted=true` only for trusted
  mentor-provided result files.
- `html_mode="ask"` prompts after calculation in an interactive terminal.
  Unattended jobs use `non_interactive_html_mode` instead.

OUTPUTS
-------

Depending on `[output]` toggles, TriHydrA writes:

- one station folder containing TXT evidence and separate Layer 1, Layer 2,
  comparison, and Layer 3 HTML diagnostics;
- one `trihydra_results.nc` containing station/series summaries, provenance,
  thresholds and sparse flagged-diagnostic records;
- one `trihydra_run.log` containing progress, warnings, failures and timings.

The NetCDF design is documented in `NETCDF_SCHEMA.md`.

TESTS
-----

Run from this folder:

    pytest

================================================================================
