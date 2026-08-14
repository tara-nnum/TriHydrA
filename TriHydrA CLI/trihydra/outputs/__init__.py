"""Plain-text result writer used by the basic API."""

from trihydra.outputs.api import save_results
from trihydra.outputs.netcdf import build_netcdf_dataset, write_netcdf_results

__all__ = ["build_netcdf_dataset", "save_results", "write_netcdf_results"]
