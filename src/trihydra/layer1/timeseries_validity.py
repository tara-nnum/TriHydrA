import pandas as pd


def to_datetime_sorted(series: pd.Series) -> pd.Series:
    """Return a copy with datetime index sorted."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def get_valid_record(series: pd.Series) -> pd.Series:
    """
    Keep only the period between first and last valid value.
    Leading/trailing NaNs are ignored.
    """
    s = to_datetime_sorted(series)

    first_valid = s.first_valid_index()
    last_valid = s.last_valid_index()

    if first_valid is None or last_valid is None:
        return s.iloc[0:0]

    return s.loc[first_valid:last_valid]


def timestamps_to_strings(index) -> list:
    """Convert timestamps to readable strings."""
    return [str(x) for x in index]