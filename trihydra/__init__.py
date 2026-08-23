"""TriHydrA: layered plausibility assessment for streamflow time series."""

from trihydra.io.models import SourceProvenance, StationData
from trihydra.io.api import load_stations
from trihydra.pipeline import run_trihydra
from trihydra.network import run_trihydra_batch, run_trihydra_network
from trihydra.plotting.api import plot_results
from trihydra.outputs.api import save_results
from trihydra.result import TriHydrABatchResult, TriHydrANetworkResult, TriHydrAResult
from trihydra.batch import run_batch

__version__ = "0.5.0"

__all__ = [
    "SourceProvenance",
    "StationData",
    "load_stations",
    "TriHydrAResult",
    "TriHydrABatchResult",
    "TriHydrANetworkResult",
    "run_batch",
    "run_trihydra",
    "run_trihydra_batch",
    "run_trihydra_network",
    "plot_results",
    "save_results",
    "__version__",
]
