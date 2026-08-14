# TriHydrA

TriHydrA is a layered plausibility-assessment framework for long-term
streamflow time series.

This repository contains two editions:

## TriHydrA Basic

Designed for individual users working in Python or Jupyter Notebook.

- Simple CSV input
- Minimal configuration
- Interactive HTML diagnostics
- Human-readable text reports

See [TriHydrA Basic/README.txt](TriHydrA%20Basic/README.txt).

## TriHydrA CLI

Designed for configurable batch processing and HPC-oriented workflows.

- NetCDF observation input
- AIFL pickle simulation input
- TOML and Pydantic configuration
- Multi-station processing
- TXT, NetCDF, and HTML outputs
- Automated test suite

See [TriHydrA CLI/README.txt](TriHydrA%20CLI/README.txt).

## Assessment structure

- Layer 1: intrinsic time-series quality checks
- Layer 2: hydrological signatures and event behaviour
- Layer 3: network and catchment context
- Comparison: observation–simulation or series–series assessment

> A TriHydrA review classification indicates that evidence should be
> inspected. It does not by itself establish that the data are erroneous.
