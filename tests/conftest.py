"""Small, readable time series shared by the TriHydrA tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_daily_flow():
    """Two years of positive daily flow with a clear seasonal cycle."""
    dates = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    day = np.arange(len(dates))
    flow = 2.0 + np.sin(2 * np.pi * day / 365.25)
    return pd.Series(flow, index=dates, name="discharge")


@pytest.fixture
def simple_flood_event():
    """A short hydrograph whose start, peak and end are easy to verify."""
    dates = pd.date_range("2000-01-01", periods=9, freq="D")
    flow = [1, 1, 2, 4, 8, 5, 3, 1, 1]
    return pd.Series(flow, index=dates, dtype=float, name="discharge")
