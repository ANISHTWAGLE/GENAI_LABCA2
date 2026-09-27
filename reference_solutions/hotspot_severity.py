import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

THRESHOLD = {"plains": 40.0, "coastal": 37.0, "hilly": 30.0}


def severity(v):
    if v >= 10:
        return "Extreme"
    if v >= 5:
        return "High"
    if v >= 2:
        return "Moderate"
    return "Low"


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/imd_tmax.csv", parse_dates=["date"])
    # <<FAULT>>
    df = df[df["tmax"].notna() & (df["tmax"] < 90)].copy()  # <<CLEAN>>
    df["month"], df["day"] = df["date"].dt.month, df["date"].dt.day
    df["normal"] = df.groupby(["lat", "lon", "month", "day"])["tmax"].transform("mean")
    dep = df["tmax"] - df["normal"]
    eligible = df["tmax"] >= df["terrain"].map(THRESHOLD)
    df["hw"] = eligible & ((dep >= 4.5) | (df["tmax"] >= 45))

    n_years = df["date"].dt.year.nunique()
    g = (df.groupby(["lat", "lon", "region"])["hw"].sum() / n_years).round(2)
    g = g.rename("hw_days_per_year").reset_index().sort_values("hw_days_per_year", ascending=False)
    g["severity"] = g["hw_days_per_year"].map(severity)

    top = g.head(10).reset_index(drop=True)
    result = {
        "n_gridpoints": int(len(g)),
        "hotspots": [{"rank": i + 1, "lat": float(r.lat), "lon": float(r.lon), "region": r.region,
                      "hw_days_per_year": float(r.hw_days_per_year), "severity": r.severity}
                     for i, r in top.iterrows()],
        "severity_counts": {k: int((g["severity"] == k).sum()) for k in ["Low", "Moderate", "High", "Extreme"]},  # <<SCHEMA>>
    }
    with open("outputs/result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(g["lon"], g["lat"], c=g["hw_days_per_year"], cmap="YlOrRd", s=160, edgecolor="grey")
    fig.colorbar(sc, ax=ax, label="heatwave days per year")
    for _, r in top.head(3).iterrows():
        ax.annotate(f"{r.hw_days_per_year:.1f}", (r.lon, r.lat), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("longitude (E)")
    ax.set_ylabel("latitude (N)")
    ax.set_title("Heatwave hotspots, 2015-2024")
    fig.tight_layout()
    fig.savefig("outputs/hotspot_map.png", dpi=110)
    plt.close(fig)
    print(top[["lat", "lon", "region", "hw_days_per_year", "severity"]].to_string(index=False))


if __name__ == "__main__":
    main()
