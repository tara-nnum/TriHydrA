"""Assign the Köppen–Geiger class at a gauge coordinate."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from PIL import Image


def bundled_climate_paths() -> tuple[Path, Path]:
    """Return the climate raster and legend installed with TriHydrA."""
    resources = files("trihydra.resources.climate")
    return (
        Path(str(resources.joinpath("koppen_geiger_1991_2020_0p1.tif"))),
        Path(str(resources.joinpath("koppen_geiger_legend.txt"))),
    )


@dataclass(frozen=True)
class ClimateResult:
    climate_id: int
    climate_code: str | None
    climate_description: str | None
    lookup_status: str


def read_climate_legend(path: str | Path) -> dict[int, tuple[str, str]]:
    """Read the 30 class codes and descriptions supplied with the raster."""
    classes: dict[int, tuple[str, str]] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line or not line.split(":", 1)[0].isdigit():
            continue
        number, remainder = line.split(":", 1)
        words = remainder.split("[", 1)[0].split()
        classes[int(number)] = (words[0], " ".join(words[1:]))
    if set(classes) != set(range(1, 31)):
        raise ValueError("Climate legend must define classes 1 through 30.")
    return classes


class ClimateLookup:
    """Sample a global categorical raster without interpolating class codes."""

    def __init__(self, raster_path: str | Path, legend_path: str | Path):
        self.image = Image.open(Path(raster_path))
        self.classes = read_climate_legend(legend_path)
        self.width, self.height = self.image.size
        if self.width != 2 * self.height:
            raise ValueError("Expected a global longitude/latitude raster with a 2:1 shape.")

    def close(self) -> None:
        self.image.close()

    def __enter__(self) -> "ClimateLookup":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def lookup(self, latitude: float, longitude: float) -> ClimateResult:
        """Return the climate class in the raster cell containing the gauge."""
        latitude, longitude = float(latitude), float(longitude)
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Station coordinates are outside valid latitude/longitude ranges.")
        column = min(int((longitude + 180.0) / (360.0 / self.width)), self.width - 1)
        row = min(int((90.0 - latitude) / (180.0 / self.height)), self.height - 1)
        climate_id = int(self.image.getpixel((column, row)))
        climate = self.classes.get(climate_id)
        if climate is None:
            return ClimateResult(0, None, None, "no_land_class_at_station")
        return ClimateResult(climate_id, climate[0], climate[1], "matched_station_cell")
