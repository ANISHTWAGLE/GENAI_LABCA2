"""Generate a synthetic, IMD-like daily maximum-temperature dataset.

Mimics the IMD Pune 1-degree gridded Tmax product after extraction from GRD
files: one row per (date, grid point) with lat/lon, the IMD homogeneous
temperature region, terrain class and Tmax in degC (1 decimal).

Realism features (these are what make the generated code fail in interesting ways):
  * region-specific annual cycles (pre-monsoon peak, monsoon dip)
  * shared regional + local AR(1) weather noise, small warming trend
  * spatially coherent heatwave episodes in Mar-Jun, more frequent in NW / NC
  * ~1% NaN and ~0.3% IMD missing-value sentinel 99.9
  * a guaranteed heatwave during 2024-05-20..26 (used by the advisory task)

Usage:  python data/generate_data.py   (writes data/imd_tmax.csv)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2015-01-01", "2024-12-31"
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "imd_tmax.csv"
META_PATH = OUT_DIR / "imd_tmax_meta.json"

# IMD's seven homogeneous temperature regions: terrain, grid points, monthly normal Tmax (Jan..Dec)
REGIONS = {
    "Western Himalaya": ("hilly", [(32, 76), (33, 75), (31, 77), (30, 79), (34, 77)],
                         [10, 12, 17, 22, 26, 28, 27, 26, 25, 21, 16, 12]),
    "North West": ("plains", [(28, 77), (27, 75), (26, 73), (29, 76), (30, 75)],
                   [20, 24, 30, 37, 41, 40, 36, 34, 34, 33, 28, 22]),
    "North Central": ("plains", [(25, 80), (24, 78), (26, 82), (23, 79), (25, 77)],
                      [24, 28, 34, 39, 42, 39, 33, 31, 32, 32, 29, 25]),
    "North East": ("plains", [(26, 92), (25, 91), (27, 94), (24, 93), (26, 90)],
                   [23, 25, 28, 30, 31, 31, 31, 31, 31, 30, 27, 24]),
    "East Coast": ("coastal", [(17, 82), (16, 81), (13, 80), (19, 84), (20, 86)],
                   [29, 31, 34, 36, 38, 36, 33, 32, 32, 31, 29, 28]),
    "West Coast": ("coastal", [(19, 73), (18, 73), (15, 74), (12, 75), (10, 76)],
                   [30, 31, 32, 33, 33, 30, 28, 28, 29, 31, 32, 31]),
    "Interior Peninsula": ("plains", [(17, 78), (15, 77), (18, 79), (16, 76), (14, 78)],
                           [30, 33, 36, 39, 39, 34, 30, 29, 30, 30, 29, 28]),
}
# expected heatwave episodes per pre-monsoon season
EPISODE_RATE = {"Western Himalaya": 1.0, "North West": 2.5, "North Central": 2.5, "North East": 0.5,
                "East Coast": 1.5, "West Coast": 0.4, "Interior Peninsula": 1.5}
# forced event for the advisory window: (region, amplitude degC)
FORCED_EVENT = {"start": "2024-05-20", "days": 5,
                "amplitude": {"North West": 7.0, "North Central": 6.0, "Interior Peninsula": 5.0}}


def _daily_normal(monthly: list[float], doy: np.ndarray) -> np.ndarray:
    """Smoothly interpolate monthly normals (placed mid-month) to day-of-year, with wrap-around."""
    mid = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
    x = np.concatenate([mid - 365, mid, mid + 365])
    y = np.tile(np.asarray(monthly, dtype=float), 3)
    return np.interp(doy, x, y)


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    eps = rng.normal(0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(1 - phi**2)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _episode_signal(rng: np.random.Generator, dates: pd.DatetimeIndex, rate: float) -> np.ndarray:
    """Bell-shaped warm anomalies in Mar-Jun, shared by all grid points of a region."""
    sig = np.zeros(len(dates))
    for year in sorted(set(dates.year)):
        for _ in range(rng.poisson(rate)):
            start = pd.Timestamp(year=year, month=3, day=1) + pd.Timedelta(days=int(rng.integers(0, 110)))
            length = int(rng.integers(3, 10))
            amp = rng.uniform(4.5, 8.5)
            idx = np.where((dates >= start) & (dates < start + pd.Timedelta(days=length)))[0]
            if len(idx):
                sig[idx] += amp * np.sin(np.linspace(0.35, np.pi - 0.35, len(idx)))
    return sig


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")
    doy = dates.dayofyear.to_numpy()
    years = dates.year.to_numpy() - 2015
    forced_idx = np.where((dates >= FORCED_EVENT["start"])
                          & (dates < pd.Timestamp(FORCED_EVENT["start"]) + pd.Timedelta(days=FORCED_EVENT["days"])))[0]

    frames = []
    for region, (terrain, points, monthly) in REGIONS.items():
        regional_noise = _ar1(rng, len(dates), 0.75, 1.1)
        episodes = _episode_signal(rng, dates, EPISODE_RATE[region])
        if region in FORCED_EVENT["amplitude"]:
            amp = FORCED_EVENT["amplitude"][region]
            episodes[forced_idx] = np.maximum(episodes[forced_idx],
                                              amp * np.sin(np.linspace(0.6, np.pi - 0.6, len(forced_idx))) + 0.8)
        # suppress random episodes in the advisory window for the "quiet" regions
        else:
            episodes[forced_idx] = 0.0
            regional_noise[forced_idx] = np.clip(regional_noise[forced_idx], -1.5, 1.0)
        for lat, lon in points:
            offset = rng.normal(0, 0.8)
            local_noise = _ar1(rng, len(dates), 0.5, 0.7)
            jitter = rng.uniform(0.85, 1.15)
            tmax = (_daily_normal(monthly, doy) + offset + 0.025 * years
                    + regional_noise + local_noise + jitter * episodes)
            frames.append(pd.DataFrame({"date": dates, "lat": float(lat), "lon": float(lon),
                                        "region": region, "terrain": terrain, "tmax": np.round(tmax, 1)}))

    df = pd.concat(frames, ignore_index=True)
    # data-quality problems, as in real extracted GRD data
    n = len(df)
    df.loc[rng.random(n) < 0.010, "tmax"] = np.nan
    df.loc[rng.random(n) < 0.003, "tmax"] = 99.9
    # same readable location columns as the real dataset
    from prepare_real_data import CITIES, _haversine_km
    names, cll = list(CITIES), np.array(list(CITIES.values()))
    near = {}
    for lat, lon in df[["lat", "lon"]].drop_duplicates().to_numpy():
        d = _haversine_km(lat, lon, cll[:, 0], cll[:, 1])
        near[(lat, lon)] = (names[int(np.argmin(d))], int(round(d.min())))
    keys = list(zip(df["lat"], df["lon"]))
    df["nearest_city"] = [near[k][0] for k in keys]
    df["city_km"] = [near[k][1] for k in keys]
    df = df[["date", "lat", "lon", "region", "terrain", "nearest_city", "city_km", "tmax"]]
    return df.sort_values(["date", "lat", "lon"]).reset_index(drop=True)


def main() -> None:
    df = generate()
    df.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d")
    meta = {
        "source": "Synthetic IMD-like 1-degree gridded daily Tmax (stand-in for IMD Pune GRD extraction)",
        "rows": len(df), "grid_points": int(df.groupby(["lat", "lon"]).ngroups),
        "date_range": [START, END], "seed": SEED,
        "columns": {"date": "datetime64", "lat": "degrees N (1-degree grid)", "lon": "degrees E",
                    "region": "IMD homogeneous temperature region (7)", "terrain": "plains | coastal | hilly",
                    "tmax": "daily maximum temperature, degC, 1 decimal",
                    "nearest_city": "closest major city", "city_km": "distance to that city"},
        "missing_values": "NaN, or the IMD sentinel 99.9",
        "regions": list(REGIONS),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_PATH.name}: {len(df):,} rows, {meta['grid_points']} grid points")


if __name__ == "__main__":
    main()
