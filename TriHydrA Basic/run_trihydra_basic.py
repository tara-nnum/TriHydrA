"""Minimal TriHydrA example for plain Python users."""

from pathlib import Path

import pandas as pd

from trihydra import plot_results, run_trihydra_network, save_results


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "example_three_stations.csv"
CONTEXT_FILE = HERE / "context.csv"
RESULTS_DIRECTORY = HERE / "results"
UNIT = "mm/d"


def main() -> None:
    """Read the example CSV, run TriHydrA, and save TXT/HTML results."""
    data = pd.read_csv(DATA_FILE, parse_dates=["date"]).set_index("date")
    stations = {station_id: data[station_id] for station_id in data.columns}

    result = run_trihydra_network(
        stations,
        context_path=CONTEXT_FILE,
        target_station_ids=list(stations),
        unit=UNIT,
    )

    save_results(result, RESULTS_DIRECTORY)
    plot_results(result, RESULTS_DIRECTORY)
    print(f"Finished. Open the files in: {RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()
