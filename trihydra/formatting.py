"""Small formatting helpers shared by the domain-owned TXT summaries."""

from __future__ import annotations

import calendar
from typing import Any

import numpy as np
import pandas as pd


WIDTH = 80


def value(value: Any, *, decimals: int | None = None, suffix: str = "") -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "not provided"
    if isinstance(value, (bool, np.bool_)):
        text = "Yes" if value else "No"
    elif decimals is not None:
        text = f"{float(value):.{decimals}f}"
    else:
        text = str(value)
    return f"{text}{suffix}"


def date(value_: Any) -> str:
    if value_ is None or pd.isna(value_):
        return "not provided"
    return pd.Timestamp(value_).strftime("%d %b %Y").lstrip("0")


def month(value_: Any) -> str:
    if value_ is None or pd.isna(value_):
        return "not provided"
    number = int(value_)
    return calendar.month_name[number] if 1 <= number <= 12 else str(value_)


def field(label: str, value_: Any, indent: int = 4) -> str:
    left = " " * indent + label + " "
    dots = "." * max(1, 48 - len(left))
    return f"{left}{dots} {value_}"


def line(character: str = "-") -> str:
    return character * WIDTH


def section(name: str) -> list[str]:
    return ["", line(), name, line()]


def metric(row: pd.Series, name: str, **kwargs) -> str:
    return value(row.get(name), **kwargs)


__all__ = ["WIDTH", "date", "field", "line", "metric", "month", "section", "value"]
