"""Shared result container for every Layer 2 hydrological signature.

It keeps scalar metrics, supporting tables, coverage metadata, warnings, and
execution status together so orchestration and comparison use one contract.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

@dataclass
class SignatureResult:
    """
    Standard output container for one signature calculation.

    Attributes
    ----------
    status
        "ok", "warning", or "insufficient_data".

    metrics
        Dictionary containing calculated scalar metrics.

    tables
        Dictionary containing supporting DataFrames or Series.

    metadata
        Information about record length, dates, timestep, and data coverage.

    warnings
        List of warning messages generated during calculation.
    """

    status: str
    metrics: dict[str, Any]
    tables: dict[str, Any]
    metadata: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary."""
        return {
            "status": self.status,
            "metrics": self.metrics,
            "tables": self.tables,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }
