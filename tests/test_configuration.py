"""Plain-language checks for TriHydrA's runtime and plotting choices."""

import pytest

from trihydra.batch import _selected_for_plotting
from trihydra.io.models import SourceProvenance, StationData
from trihydra.result import TriHydrAResult
from trihydra.settings.defaults import get_default_config
import pandas as pd


def _result_with_class(label: str) -> TriHydrAResult:
    series = pd.Series([1.0], index=pd.to_datetime(["2000-01-01"]))
    station = StationData(
        station_id="gauge_1",
        obs=series,
        unit="mm/day",
        obs_provenance=SourceProvenance.in_memory(unit="mm/day"),
    )
    return TriHydrAResult(
        station=station,
        summary=pd.DataFrame({"layer1_class": [label]}),
    )


def test_internal_runtime_defaults_only_hold_pipeline_layer_switches():
    run = get_default_config()["run"]
    assert set(run) == {"layers"}
    assert run["layers"] == {
        "run_layer1": True,
        "run_layer2": True,
        "run_comparison": True,
    }


@pytest.mark.parametrize("label", ["Needs review", "Review"])
def test_review_labels_trigger_review_only_plots(label):
    result = _result_with_class(label)
    assert _selected_for_plotting([result], "review_only") == [result]


def test_non_review_labels_do_not_trigger_review_only_plots():
    result = _result_with_class("Minor concerns")
    assert _selected_for_plotting([result], "review_only") == []


def test_concerns_and_review_mode_includes_minor_concerns():
    result = _result_with_class("Minor concerns")
    assert _selected_for_plotting([result], "concerns_and_review") == [result]


def test_needs_review_mode_excludes_minor_concerns():
    result = _result_with_class("Minor concerns")
    assert _selected_for_plotting([result], "needs_review") == []


def test_each_call_gets_an_independent_configuration_copy():
    first = get_default_config()
    second = get_default_config()
    first["layer2"]["events"]["trigger_percentile"] = 0.50
    assert second["layer2"]["events"]["trigger_percentile"] == 0.95


def test_layer2_comparison_defaults_are_available_in_central_config():
    comparison = get_default_config()["layer2"]["comparison"]
    assert comparison["similarity_tier3_minimum"] == 0.80
    assert comparison["minimum_assessable_components"] == 4
    assert comparison["time_to_peak_tier3_max_days"] == 3.0
    assert comparison["weights"]["representative_event_shape"] == 1
