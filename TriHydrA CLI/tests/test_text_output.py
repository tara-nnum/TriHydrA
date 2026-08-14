"""Checks for readable station-network text output."""

import pandas as pd

from trihydra.io.models import SourceProvenance, StationData
from trihydra.outputs import save_results
from trihydra.result import TriHydrANetworkResult, TriHydrAResult


def _result(station_id: str) -> TriHydrAResult:
    station = StationData(
        station_id=station_id,
        obs=pd.Series([1.0], index=pd.to_datetime(["2000-01-01"])),
        unit="mm/day",
        obs_provenance=SourceProvenance.in_memory(unit="mm/day"),
    )
    summary = pd.DataFrame([{
        "station_id": station_id,
        "series_name": "observation",
        "series_role": "observation",
        "unit": "mm/day",
        "layer1_score": 0,
        "layer1_class": "No review needed",
    }])
    return TriHydrAResult(station=station, summary=summary)


def test_network_save_writes_one_index_and_each_station_summary(tmp_path):
    results = {station_id: _result(station_id) for station_id in ("A", "B")}
    network = TriHydrANetworkResult(
        station_results=results,
        layer3_run=None,
        summary=pd.concat([result.summary for result in results.values()], ignore_index=True),
        series_by_station={key: value.station.obs for key, value in results.items()},
    )

    written = save_results(network, tmp_path)

    assert (tmp_path / "network_summary.txt").is_file()
    assert (tmp_path / "A" / "summary.txt").is_file()
    assert (tmp_path / "B" / "summary.txt").is_file()
    assert "A" in (tmp_path / "network_summary.txt").read_text(encoding="utf-8")
    assert "B" in (tmp_path / "network_summary.txt").read_text(encoding="utf-8")
    assert "network_summary" in written
