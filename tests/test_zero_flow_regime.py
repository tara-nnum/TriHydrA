"""Regression tests for zero-flow summaries with missing calendar months."""

import numpy as np
import pandas as pd
import pytest

from trihydra.layer1.zero_flow_regime import check_zero_flow_regime


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_fully_missing_internal_month_is_not_treated_as_zero_flow(dtype):
    """A month without observations is unavailable, not a zero-flow month."""
    dates = pd.date_range("1980-01-01", "1980-03-31", freq="D")
    flow = pd.Series(1.0, index=dates, dtype=dtype)
    flow.loc["1980-02-01":"1980-02-29"] = pd.NA
    flow.loc["1980-01-10"] = 0.0

    result = check_zero_flow_regime(flow)

    assert result["status"] == "descriptor"
    assert result["valid_observation_count"] == 62
    assert result["zero_count"] == 1
    assert result["zero_ratio"] == pytest.approx(1 / 62)
    assert result["monthly_zero_ratio"][1] == pytest.approx(1 / 31)
    assert 2 not in result["monthly_zero_ratio"]
    assert result["monthly_zero_ratio"][3] == 0.0
    assert result["zero_flow_months"] == [1]


def test_multiple_missing_months_and_trailing_missing_rows_do_not_crash():
    """Internal missing months are skipped and trailing gaps remain excluded."""
    dates = pd.date_range("1980-01-01", "1981-04-30", freq="D")
    flow = pd.Series(2.0, index=dates, dtype="Float64")
    flow.loc[flow.index.month.isin([2, 3])] = pd.NA
    flow.loc["1981-04-01":] = pd.NA

    result = check_zero_flow_regime(flow)

    assert result["status"] == "descriptor"
    assert result["zero_ratio"] == 0.0
    assert set(result["monthly_zero_ratio"]) == {
        1, 4, 5, 6, 7, 8, 9, 10, 11, 12
    }
    assert all(np.isfinite(value) for value in result["monthly_zero_ratio"].values())


def test_completely_missing_series_is_skipped():
    """An all-missing record keeps the established skipped result."""
    dates = pd.date_range("1980-01-01", "1980-03-31", freq="D")
    flow = pd.Series(pd.NA, index=dates, dtype="Float64")

    result = check_zero_flow_regime(flow)

    assert result["status"] == "skipped"
    assert result["value"] is None
