# Data resources

This directory contains the example station data and metadata used by the
root `trihydra.toml` configuration. The public NetCDF and CSV resources may be
used to reproduce the observation and Layer 3 examples. Model-result pickle
files are local validation data and are not distributed.

## Public example files

### `caravan_observations.nc`

Daily streamflow observations prepared from the CARAVAN and associated
hydrological observation collections used during TriHydrA development.

- Dimensions: `date` (27,760) and `basin` (883)
- Station coordinate: `basin`
- Time coordinate: `date`
- Data variable: `streamflow`
- Intended TriHydrA format: `netcdf`

The active root configuration demonstrates how these names are supplied to
the NetCDF adapter. Users may point TriHydrA to another compatible NetCDF file
and change the variable and coordinate names in `trihydra.toml`.

### `context.csv`

Compact station metadata used to select and describe Layer 3 peers. It has one
row per available station and the following fields:

- `station_id`
- `longitude`
- `latitude`
- `river_name`
- `catchment_name`
- `catchment_area_km2`
- `series_type`

Station identifiers must match those in the input time-series source. Missing
context rows do not stop Layer 1 or Layer 2, but the affected station cannot
participate fully in Layer 3.

### `static_attributes_filtered.csv`

Catchment attributes associated with the example stations, including gauge
location, catchment area, climate, topographic, land-cover and hydrological
descriptors. This is a supporting data table retained for reproducibility and
future context extensions. The current Layer 3 workflow primarily uses
`context.csv`; users do not normally need to edit this larger table.

## Private files not distributed

Files named `*_results.p`, along with other `.p` and `.pickle` files, contain
trusted AIFL model-result examples supplied for development and validation.
They are intentionally excluded by `.gitignore` because the source dataset is
not publicly available. TriHydrA's trusted AIFL adapter remains available for
authorised users who provide these files locally.

Never load an untrusted pickle file: Python pickle content can execute code.

## Attribution and redistribution

The observation and catchment information originates from public hydrological
data products, including CARAVAN and their upstream sources. Users should cite
the relevant source datasets when publishing results. Before redistributing a
derived copy, verify and follow the licences and attribution requirements of
the original data providers; inclusion here does not replace those terms.

These files are example inputs, not Python package resources. TriHydrA's
scientific layers consume canonical pandas time series and do not depend on
these filenames.
