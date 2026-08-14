"""TriHydrA: layered plausibility assessment for streamflow time series."""

from trihydra.io.models import SourceProvenance, StationData
from trihydra.pipeline import run_station, run_trihydra, station_from_series
from trihydra.network import run_trihydra_network
from trihydra.plotting.api import plot_results
from trihydra.outputs.api import save_results
from trihydra.result import TriHydrANetworkResult, TriHydrAResult

__version__ = "0.4.0"

__all__ = [
    "SourceProvenance",
    "StationData",
    "TriHydrAResult",
    "TriHydrANetworkResult",
    "run_station",
    "run_trihydra",
    "run_trihydra_network",
    "plot_results",
    "save_results",
    "station_from_series",
    "__version__",
]
