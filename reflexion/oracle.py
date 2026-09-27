"""Trusted reference computations used by the SEMANTIC evaluation layer.

The generated code runs in the sandbox; this module runs in the (trusted) parent
process and recomputes the ground truth from the same data with the same rules
that the task prompt states. Comparing the two is what lets the evaluator reject
code that runs without crashing but produces wrong numbers.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import dataset

SEASONS = {1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
           6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "Post-monsoon", 11: "Post-monsoon", 12: "Post-monsoon"}
SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
TERRAIN_THRESHOLD = {"plains": 40.0, "coastal": 37.0, "hilly": 30.0}
SEVERITY_BANDS = [(10.0, "Extreme"), (5.0, "High"), (2.0, "Moderate"), (0.0, "Low")]
ADVISORY_WINDOW = ("2024-05-20", "2024-05-26")
FORECAST_REGION = "North West"
FORECAST_SPLIT = "2023-01-01"


def load_clean(path: str | None = None) -> pd.DataFrame:
    return _load_clean(path or str(dataset.active().path))


@lru_cache(maxsize=4)
def _load_clean(path: str) -> pd.DataFrame:
    df = dataset.read(path)
    df = df[df["tmax"].notna() & (df["tmax"] < 90)].copy()
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["season"] = df["month"].map(SEASONS)
    return df


def classified(path: str | None = None) -> pd.DataFrame:
    """Every valid grid-point-day with its departure and IMD heatwave class."""
    return _classified(path or str(dataset.active().path))


@lru_cache(maxsize=4)
def _classified(path: str) -> pd.DataFrame:
    df = _load_clean(path).copy()
    df["normal"] = df.groupby(["lat", "lon", "month", "day"])["tmax"].transform("mean")
    df["departure"] = df["tmax"] - df["normal"]
    eligible = df["tmax"] >= df["terrain"].map(TERRAIN_THRESHOLD)
    df["severe"] = eligible & ((df["departure"] > 6.4) | (df["tmax"] >= 47))
    df["hw"] = eligible & ((df["departure"] >= 4.5) | (df["tmax"] >= 45))
    return df


def heatwave_events(df: pd.DataFrame) -> pd.DataFrame:
    """Runs of >=2 consecutive heatwave days at one grid point; season = season of run start."""
    hw = df[df["hw"]].sort_values(["lat", "lon", "date"])
    new_run = ((hw["date"].diff() != pd.Timedelta(days=1))
               | (hw["lat"] != hw["lat"].shift()) | (hw["lon"] != hw["lon"].shift()))
    hw = hw.assign(run=new_run.cumsum())
    runs = hw.groupby("run").agg(region=("region", "first"), start=("date", "min"), length=("date", "size"))
    runs = runs[runs["length"] >= 2]
    runs["season"] = runs["start"].dt.month.map(SEASONS)
    return runs


def heatwave_detection_truth() -> dict:
    df = classified()
    runs = heatwave_events(df)
    out = {"regions": {}}
    for region in sorted(df["region"].unique()):
        r = df[df["region"] == region]
        out["regions"][region] = {}
        for season in SEASON_ORDER:
            s = r[r["season"] == season]
            out["regions"][region][season] = {
                "hw_days": int(s["hw"].sum()), "severe_days": int(s["severe"].sum()),
                "events": int(((runs["region"] == region) & (runs["season"] == season)).sum())}
    out["total_hw_days"] = int(df["hw"].sum())
    out["total_severe_days"] = int(df["severe"].sum())
    out["total_events"] = int(len(runs))
    return out


def severity_class(hw_days_per_year: float) -> str:
    for lower, name in SEVERITY_BANDS:
        if hw_days_per_year >= lower:
            return name
    return "Low"


def hotspot_truth() -> dict:
    df = classified()
    n_years = df["date"].dt.year.nunique()
    g = (df.groupby(["lat", "lon", "region"])["hw"].sum() / n_years).rename("hw_days_per_year").reset_index()
    g = g.sort_values("hw_days_per_year", ascending=False).reset_index(drop=True)
    g["severity"] = g["hw_days_per_year"].map(severity_class)
    counts = {name: int((g["severity"] == name).sum()) for _, name in SEVERITY_BANDS}
    return {"n_gridpoints": len(g), "table": g, "severity_counts": counts}


def regional_series(region: str = FORECAST_REGION) -> pd.Series:
    df = load_clean()
    s = df[df["region"] == region].groupby("date")["tmax"].mean()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full).interpolate(limit_direction="both")


def forecast_truth() -> dict:
    s = regional_series()
    frame = pd.DataFrame({"y": s, "lag1": s.shift(1)}).dropna()
    frame = frame.iloc[6:]  # lag-7 feature removes the first 7 days
    test = frame[frame.index >= FORECAST_SPLIT]
    err = test["y"] - test["lag1"]
    return {"n_test": len(test), "n_train": int((frame.index < FORECAST_SPLIT).sum()),
            "baseline_mae": float(err.abs().mean()), "baseline_rmse": float(np.sqrt((err**2).mean()))}


def advisory_truth() -> dict:
    df = classified()
    start, end = ADVISORY_WINDOW
    w = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    w["level"] = np.where(w["severe"], 2, np.where(w["hw"], 1, 0))
    out = {}
    for region, r in w.groupby("region"):
        daily = r.groupby("date")["level"].max()
        worst = int(daily.max())
        hw_days = int((daily >= 1).sum())
        severity = ["None", "Heatwave", "Severe Heatwave"][worst]
        if worst == 2:
            color = "Red"
        elif worst == 1:
            color = "Orange" if hw_days >= 2 else "Yellow"
        else:
            color = "Green"
        out[region] = {"max_severity": severity, "hw_day_count": hw_days,
                       "peak_tmax": round(float(r["tmax"].max()), 1), "color_code": color}
    return out


if __name__ == "__main__":
    t = heatwave_detection_truth()
    print("totals:", t["total_hw_days"], t["total_severe_days"], t["total_events"])
    for reg, seasons in t["regions"].items():
        print(f"  {reg:20s}", {k: v["hw_days"] for k, v in seasons.items()})
    h = hotspot_truth()
    print(h["table"].head(10).to_string(), h["severity_counts"])
    print(forecast_truth())
    for k, v in advisory_truth().items():
        print(" ", k, v)
