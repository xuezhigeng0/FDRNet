"""Generate a small synthetic WeatherBench-style precipitation dataset."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def generate_precipitation(
    hours: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    year: int,
) -> np.ndarray:
    """Create deterministic moving precipitation patterns."""
    rng = np.random.default_rng(seed=year)

    lat_grid, lon_grid = np.meshgrid(
        latitudes,
        longitudes,
        indexing="ij",
    )

    data = np.empty(
        (hours, len(latitudes), len(longitudes)),
        dtype=np.float32,
    )

    for hour in range(hours):
        center_lat = 25.0 * np.sin(2.0 * np.pi * hour / hours)
        center_lon = (hour * 360.0 / hours) % 360.0

        lon_distance = np.minimum(
            np.abs(lon_grid - center_lon),
            360.0 - np.abs(lon_grid - center_lon),
        )

        rain_band = np.exp(
            -(
                ((lat_grid - center_lat) / 18.0) ** 2
                + (lon_distance / 28.0) ** 2
            )
        )

        noise = rng.normal(
            loc=0.0,
            scale=0.01,
            size=rain_band.shape,
        )

        data[hour] = np.maximum(
            0.0,
            0.2 * rain_band + noise,
        )

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic WeatherBench total-precipitation NetCDF files."
        )
    )
    parser.add_argument(
        "--output_root",
        default="./data",
        help="Parent directory in which the weather directory is created.",
    )
    parser.add_argument(
        "--hours_per_year",
        type=int,
        default=48,
        help="Number of hourly frames generated for each year.",
    )
    args = parser.parse_args()

    if args.hours_per_year < 25:
        raise ValueError(
            "hours_per_year must be at least 25 for 12-to-12 prediction."
        )

    output_dir = (
        Path(args.output_root)
        / "weather"
        / "total_precipitation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    latitudes = np.linspace(
        87.1875,
        -87.1875,
        32,
        dtype=np.float32,
    )
    longitudes = np.linspace(
        0.0,
        354.375,
        64,
        dtype=np.float32,
    )

    for year in range(2010, 2019):
        times = pd.date_range(
            start=f"{year}-01-01",
            periods=args.hours_per_year,
            freq="h",
        )

        precipitation = generate_precipitation(
            hours=args.hours_per_year,
            latitudes=latitudes,
            longitudes=longitudes,
            year=year,
        )

        dataset = xr.Dataset(
            data_vars={
                "tp": (
                    ("time", "lat", "lon"),
                    precipitation,
                    {
                        "long_name": "synthetic total precipitation",
                        "units": "m",
                    },
                )
            },
            coords={
                "time": times,
                "lat": latitudes,
                "lon": longitudes,
            },
            attrs={
                "description": (
                    "Synthetic data for SFDRNet installation and smoke tests."
                )
            },
        )

        output_path = output_dir / (
            f"total_precipitation_{year}_5.625deg.nc"
        )
        dataset.to_netcdf(output_path)
        dataset.close()

        print(f"Created: {output_path}")

    print(f"Dummy dataset is ready under: {args.output_root}")


if __name__ == "__main__":
    main()
