"""Convert real IMD Pune 1-degree gridded daily Tmax (.GRD) files into the project's table format.

Input : data/imd_raw/tmax/<year>.GRD   (downloaded with:  python data/prepare_real_data.py --download)
Output: data/imd_tmax_real.csv          columns: date, lat, lon, region, terrain, nearest_city, city_km, tmax

IMD files hold only (time, lat, lon, tmax). Region, terrain and nearest city are ADDED here:
  * region  - IMD's 7 homogeneous temperature regions, approximated with lat/lon boundaries
  * terrain - hilly (Himalaya / NE hills), coastal (land cell touching the sea), else plains
  * nearest_city - closest of ~70 major Indian cities (great-circle distance), for readable answers
These are documented approximations (no shapefiles are used), good enough at 1-degree (~110 km) resolution.

Usage:
  python data/prepare_real_data.py --download   # fetch 2015-2024 from imdpune.gov.in (retries)
  python data/prepare_real_data.py              # convert downloaded files
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "imd_raw"
OUT = HERE / "imd_tmax_real.csv"
META = HERE / "imd_tmax_real_meta.json"
YEARS = range(2015, 2025)

CITIES = {  # name: (lat, lon) - approximate
    "Delhi": (28.6, 77.2), "Jaipur": (26.9, 75.8), "Jodhpur": (26.3, 73.0), "Bikaner": (28.0, 73.3),
    "Jaisalmer": (26.9, 70.9), "Barmer": (25.7, 71.4), "Churu": (28.3, 74.9), "Kota": (25.2, 75.8),
    "Udaipur": (24.6, 73.7), "Ahmedabad": (23.0, 72.6), "Bhuj": (23.3, 69.7), "Rajkot": (22.3, 70.8),
    "Surat": (21.2, 72.8), "Chandigarh": (30.7, 76.8), "Amritsar": (31.6, 74.9), "Ludhiana": (30.9, 75.9),
    "Hisar": (29.2, 75.7), "Srinagar": (34.1, 74.8), "Jammu": (32.7, 74.9), "Leh": (34.2, 77.6),
    "Shimla": (31.1, 77.2), "Dehradun": (30.3, 78.0), "Lucknow": (26.8, 80.9), "Kanpur": (26.4, 80.3),
    "Agra": (27.2, 78.0), "Prayagraj": (25.4, 81.8), "Varanasi": (25.3, 83.0), "Gorakhpur": (26.8, 83.4),
    "Jhansi": (25.4, 78.6), "Bhopal": (23.3, 77.4), "Indore": (22.7, 75.9), "Gwalior": (26.2, 78.2),
    "Jabalpur": (23.2, 79.9), "Raipur": (21.3, 81.6), "Bilaspur": (22.1, 82.1), "Patna": (25.6, 85.1),
    "Gaya": (24.8, 85.0), "Ranchi": (23.3, 85.3), "Jamshedpur": (22.8, 86.2), "Kolkata": (22.6, 88.4),
    "Siliguri": (26.7, 88.4), "Bhubaneswar": (20.3, 85.8), "Sambalpur": (21.5, 84.0), "Guwahati": (26.1, 91.7),
    "Dibrugarh": (27.5, 94.9), "Shillong": (25.6, 91.9), "Imphal": (24.8, 93.9), "Agartala": (23.8, 91.3),
    "Aizawl": (23.7, 92.7), "Itanagar": (27.1, 93.6), "Mumbai": (19.1, 72.9), "Pune": (18.5, 73.9),
    "Nagpur": (21.1, 79.1), "Akola": (20.7, 77.0), "Aurangabad": (19.9, 75.3), "Solapur": (17.7, 75.9),
    "Nanded": (19.2, 77.3), "Goa": (15.5, 73.8), "Hyderabad": (17.4, 78.5), "Ramagundam": (18.8, 79.5),
    "Vijayawada": (16.5, 80.6), "Visakhapatnam": (17.7, 83.2), "Kurnool": (15.8, 78.0),
    "Anantapur": (14.7, 77.6), "Nellore": (14.4, 79.9), "Bengaluru": (13.0, 77.6), "Mangaluru": (12.9, 74.9),
    "Belagavi": (15.9, 74.5), "Kalaburagi": (17.3, 76.8), "Chennai": (13.1, 80.3), "Madurai": (9.9, 78.1),
    "Tiruchirappalli": (10.8, 78.7), "Coimbatore": (11.0, 77.0), "Kochi": (10.0, 76.3),
    "Thiruvananthapuram": (8.5, 76.9), "Kozhikode": (11.3, 75.8), "Port Blair": (11.6, 92.7),
}


def download(years=YEARS, tries: int = 4) -> None:
    import imdlib as imd
    for y in years:
        f = RAW_DIR / "tmax" / f"{y}.GRD"
        for attempt in range(1, tries + 1):
            if f.exists() and f.stat().st_size > 1_000_000:
                break
            try:
                imd.get_data("tmax", y, y, fn_format="yearwise", file_dir=str(RAW_DIR))
            except Exception as e:  # IMD's server times out often
                print(f"{y}: attempt {attempt} failed ({type(e).__name__}); retrying")
                time.sleep(10)
        print(y, "ok" if f.exists() else "MISSING")


def region_of(lat: float, lon: float) -> str:
    """IMD homogeneous temperature regions, approximated by lat/lon boxes."""
    if lon >= 89.5 and lat >= 21.5:
        return "North East"
    if lat >= 32.0 or (lat >= 29.5 and 76.5 <= lon <= 81.5 and not (lat < 30.5 and lon < 77.5)):
        return "Western Himalaya"
    if (lat >= 23.5 and lon < 78.0) or (lat >= 20.5 and lon < 74.5):    # incl. Delhi / Haryana
        return "North West"
    if lat >= 21.5:
        return "North Central"
    # peninsula: the coastline bends east going south, so the coastal strips follow it
    west_edge = 74.0 if lat >= 15 else 75.5 if lat >= 12 else 77.0          # Konkan / Karnataka / Kerala
    east_edge = 80.0 if lat >= 13 else 79.0 if lat >= 10 else 78.0          # AP-Odisha / TN coast
    if lon <= west_edge:
        return "West Coast"
    if lon >= east_edge:
        return "East Coast"
    return "Interior Peninsula"


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def convert() -> pd.DataFrame:
    import imdlib as imd
    years = [y for y in YEARS if (RAW_DIR / "tmax" / f"{y}.GRD").exists()]
    if not years:
        raise SystemExit("no GRD files found - run with --download first")
    missing = sorted(set(YEARS) - set(years))
    if missing:
        print(f"warning: missing years {missing}; converting {years[0]}-{years[-1]} with gaps")
    frames = []
    for y in years:
        ds = imd.open_data("tmax", y, y, "yearwise", str(RAW_DIR)).get_xarray()
        t = ds["tmax"].where(ds["tmax"] < 90)          # IMD missing code 99.9 -> NaN
        frames.append(t.to_dataframe().reset_index())
    df = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})

    land = df.groupby(["lat", "lon"])["tmax"].transform("count") > 0   # drop sea / outside-India cells
    df = df[land].reset_index(drop=True)

    cells = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    valid = set(map(tuple, cells.to_numpy()))
    names, cll = list(CITIES), np.array(list(CITIES.values()))
    rows = []
    for lat, lon in cells.to_numpy():
        region = region_of(lat, lon)
        touches_sea = any((lat + dy, lon + dx) not in valid for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        ne_hills = region == "North East" and (lat <= 25.5 or (lat >= 27.5 and lon >= 92.5))  # not Assam valley
        if region == "Western Himalaya" or ne_hills:
            terrain = "hilly"
        elif touches_sea and lat < 23.5 and not (lon >= 88.0 and lat > 21.5):
            terrain = "coastal"
        else:
            terrain = "plains"
        d = _haversine_km(lat, lon, cll[:, 0], cll[:, 1])
        i = int(np.argmin(d))
        rows.append((lat, lon, region, terrain, names[i], int(round(d[i]))))
    meta = pd.DataFrame(rows, columns=["lat", "lon", "region", "terrain", "nearest_city", "city_km"])
    df = df.merge(meta, on=["lat", "lon"], how="left")
    df["tmax"] = df["tmax"].astype("float64").round(1)
    df = df[["date", "lat", "lon", "region", "terrain", "nearest_city", "city_km", "tmax"]]
    return df.sort_values(["date", "lat", "lon"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()
    df = convert()
    df.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    cells = df.drop_duplicates(["lat", "lon"])
    meta = {"source": "IMD Pune 1-degree gridded daily maximum temperature (imdpune.gov.in), via imdlib",
            "rows": len(df), "grid_points": len(cells),
            "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
            "missing_tmax_fraction": round(float(df["tmax"].isna().mean()), 4),
            "cells_per_region": cells["region"].value_counts().to_dict(),
            "cells_per_terrain": cells["terrain"].value_counts().to_dict(),
            "note": "region, terrain and nearest_city are approximations added by prepare_real_data.py"}
    META.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
