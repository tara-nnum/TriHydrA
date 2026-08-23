"""Checks for meaningful command-line process status codes."""

import pandas as pd
from types import SimpleNamespace

from trihydra import cli


def test_cli_fails_when_no_station_was_processed(monkeypatch):
    monkeypatch.setattr(
        cli, "run_batch", lambda _path: SimpleNamespace(manifest=pd.DataFrame())
    )
    monkeypatch.setattr("sys.argv", ["trihydra", "run", "--config", "example.toml"])

    assert cli.main() == 1


def test_cli_succeeds_only_when_every_manifest_row_completed(monkeypatch):
    monkeypatch.setattr(
        cli,
        "run_batch",
        lambda _path: SimpleNamespace(
            manifest=pd.DataFrame({"status": ["completed", "completed"]})
        ),
    )
    monkeypatch.setattr("sys.argv", ["trihydra", "run", "--config", "example.toml"])

    assert cli.main() == 0
