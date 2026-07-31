"""Human-facing numeric formatting for Layer 2 outputs.

Calculations retain full floating-point precision. Only figure labels, hover
text, diagnostic messages, and exported presentation tables use this module.
The precision constant is owned by the central Layer 1-2 configuration.
"""

from __future__ import annotations

import math
from typing import Any


from src.trihydra.layer1.config import DISPLAY_DECIMALS


def format_display_number(value: Any) -> str:
    """Return at most three decimals, using scientific notation when tiny."""
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    if number != 0 and abs(number) < 10 ** (-DISPLAY_DECIMALS):
        return f"{number:.{DISPLAY_DECIMALS}e}"
    return f"{number:.{DISPLAY_DECIMALS}f}".rstrip("0").rstrip(".")


__all__ = ["DISPLAY_DECIMALS", "format_display_number"]
