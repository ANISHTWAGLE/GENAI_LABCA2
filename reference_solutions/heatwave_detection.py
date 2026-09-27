import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEASONS = {1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
           6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "Post-monsoon", 11: "Post-monsoon", 12: "Post-monsoon"}
ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
THRESHOLD = {"plains": 40.0, "coastal": 37.0, "hilly": 30.0}


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/imd_tmax.csv", parse_dates=["date"])
    # <<FAULT>>
    df = df[df["tmax"].notna() & (df["tmax"] < 90)].copy()  # <<CLEAN>>
    df["month"], df["day"] = df["date"].dt.month, df["date"].dt.day
    df["season"] = df["month"].map(SEASONS)

    df["normal"] = df.groupby(["lat", "lon", "month", "day"])["tmax"].transform("mean")
    dep = df["tmax"] - df["normal"]
    eligible = df["tmax"] >= df["terrain"].map(THRESHOLD)
    df["severe"] = eligible & ((dep > 6.4) | (df["tmax"] >= 47))
    df["hw"] = eligible & ((dep >= 4.5) | (df["tmax"] >= 45))

    hw = df[df["hw"]].sort_values(["lat", "lon", "date"])
    new_run = ((hw["date"].diff() != pd.Timedelta(days=1))
               | (hw["lat"] != hw["lat"].shift()) | (hw["lon"] != hw["lon"].shift()))
    hw = hw.assign(run=new_run.cumsum())
    runs = hw.groupby("run").agg(region=("region", "first"), season=("season", "first"), n=("date", "size"))
    runs = runs[runs["n"] >= 2]

    result = {"regions": {}}
    for region in sorted(df["region"].unique()):
        r = df[df["region"] == region]
        result["regions"][region] = {}
        for s in ORDER:
            rs = r[r["season"] == s]
            result["regions"][region][s] = {
                "hw_days": int(rs["hw"].sum()),
                "severe_days": int(rs["severe"].sum()),
                "events": int(((runs["region"] == region) & (runs["season"] == s)).sum()),
            }
    result["total_hw_days"] = int(df["hw"].sum())
    result["total_severe_days"] = int(df["severe"].sum())  # <<SCHEMA>>
    result["total_events"] = int(len(runs))
    with open("outputs/result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    regions = list(result["regions"])
    x = np.arange(len(regions))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, s in enumerate(ORDER):
        ax.bar(x + (i - 1.5) * 0.2, [result["regions"][r][s]["hw_days"] for r in regions], 0.2, label=s)
    ax.set_xticks(x, regions, rotation=20, ha="right")
    ax.set_ylabel("heatwave grid-point-days")
    ax.set_title("Heatwave days by IMD region and season (2015-2024)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/heatwave_days_by_region.png", dpi=110)
    plt.close(fig)

    print(f"heatwave days={result['total_hw_days']} severe={result['total_severe_days']} "
          f"events={result['total_events']}")


if __name__ == "__main__":
    main()
