"""Guided two-series TriHydrA comparison for non-programming users."""

from pathlib import Path

import pandas as pd

from trihydra import plot_results, run_trihydra, save_results


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "example_three_stations.csv"
RESULTS_DIRECTORY = HERE / "comparison_results"
UNIT = "mm/d"


def _choose(columns: list[str], prompt: str, *, excluded: str | None = None) -> str:
    """Ask for one numbered series and reject invalid selections politely."""
    while True:
        print("\nAvailable series:")
        for number, name in enumerate(columns, start=1):
            note = " (already selected)" if name == excluded else ""
            print(f"  {number}. {name}{note}")
        answer = input(f"{prompt} [1-{len(columns)}]: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(columns):
            selected = columns[int(answer) - 1]
            if selected != excluded:
                return selected
        print("Please enter the number of an available series that has not already been selected.")


def main() -> None:
    """Guide the user through selecting and comparing two packaged series."""
    data = pd.read_csv(DATA_FILE, parse_dates=["date"]).set_index("date")
    columns = [str(column) for column in data.columns]
    if len(columns) < 2:
        raise ValueError("The CSV must contain at least two series columns.")

    print("=" * 72)
    print("TRIHYDRA GUIDED TWO-SERIES COMPARISON")
    print("=" * 72)
    print(f"Input file: {DATA_FILE.name}")
    print("Choose the observed/reference series first, then the series to compare.")
    reference_name = _choose(columns, "Select the observation/reference series")
    candidate_name = _choose(
        columns, "Select the model or other comparison series", excluded=reference_name
    )

    result = run_trihydra(
        data[reference_name],
        station_id=reference_name,
        unit=UNIT,
        model_series=data[candidate_name],
        model_name=candidate_name,
    )
    save_results(result, RESULTS_DIRECTORY)
    plot_results(result, RESULTS_DIRECTORY)

    station_directory = RESULTS_DIRECTORY / reference_name
    print("\nFinished successfully.")
    print(f"Reference series: {reference_name}")
    print(f"Comparison series: {candidate_name}")
    print(f"Open the results in: {station_directory}")


if __name__ == "__main__":
    main()
