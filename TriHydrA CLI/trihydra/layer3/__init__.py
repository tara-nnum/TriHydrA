"""Hydrological network-context assessment (Layer 3)."""

from trihydra.layer3.climate import ClimateLookup, ClimateResult
from trihydra.layer3.metadata import (
    ContextValidationResult,
    attach_climate_context,
    read_context_metadata,
)
from trihydra.layer3.evidence import (
    StationContextEvidence,
    build_station_context_evidence,
)
from trihydra.layer3.peers import (
    ContextPeerGroups,
    PeerSelectionResult,
    select_analogue_peers,
    select_local_peers,
    select_peer_groups,
)
from trihydra.layer3.local_comparison import (
    ContextCheckResult,
    LocalContextResult,
    compare_local_context,
)
from trihydra.layer3.analogue_comparison import (
    AnalogueContextResult,
    compare_analogue_context,
)
from trihydra.layer3.summary import (
    ContextSummary,
    Layer3Summary,
    summarise_layer3,
)
from trihydra.layer3.orchestrator import (
    Layer3RunResult,
    Layer3StationResult,
    run_layer3_context,
)
from trihydra.layer3.visualisation import (
    build_layer3_overview,
    write_layer3_overview,
)

__all__ = [
    "ClimateLookup",
    "ClimateResult",
    "ContextValidationResult",
    "attach_climate_context",
    "read_context_metadata",
    "PeerSelectionResult",
    "ContextPeerGroups",
    "select_analogue_peers",
    "select_local_peers",
    "select_peer_groups",
    "StationContextEvidence",
    "build_station_context_evidence",
    "ContextCheckResult",
    "LocalContextResult",
    "compare_local_context",
    "AnalogueContextResult",
    "compare_analogue_context",
    "ContextSummary",
    "Layer3Summary",
    "summarise_layer3",
    "Layer3RunResult",
    "Layer3StationResult",
    "run_layer3_context",
    "build_layer3_overview",
    "write_layer3_overview",
]
