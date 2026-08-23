"""Format-neutral station selection shared by every input adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal


StationSelection = str | Sequence[str] | Literal["all"] | None


def select_station_ids(
    requested: StationSelection,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve one, several, or all station IDs in deterministic order."""
    if requested is None or (
        isinstance(requested, str) and requested.casefold() == "all"
    ):
        return available
    if isinstance(requested, str):
        selected = (requested,)
    elif isinstance(requested, Iterable):
        selected = tuple(map(str, requested))
    else:
        raise TypeError("stations must be one ID, a sequence of IDs, or 'all'.")
    if not selected:
        raise ValueError("The station selection is empty.")
    selected = tuple(dict.fromkeys(selected))
    missing = [station for station in selected if station not in available]
    if missing:
        preview = ", ".join(available[:5])
        raise KeyError(
            f"Station(s) not found: {missing}. First available IDs: {preview}"
        )
    return selected


__all__ = ["StationSelection", "select_station_ids"]
