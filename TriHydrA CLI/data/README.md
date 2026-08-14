# Development data

This directory contains local files used to validate TriHydrA. They are not
part of the Python package and large formats are excluded by `.gitignore`.

- `caravan_observations.nc`: station-by-time observation dataset.
- `*_results.p`: trusted model-result pickles used for comparison testing.
- `static_attributes_filtered.csv`: station metadata for future Layer 3 work.

TriHydrA calculations consume canonical pandas time series produced by the
generic ingestion boundary; the scientific layers do not depend on these
specific dataset names.
