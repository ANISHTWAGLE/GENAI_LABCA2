import json
import os

import numpy as np
import pandas as pd

THRESHOLD = {"plains": 40.0, "coastal": 37.0, "hilly": 30.0}
START, END = "2024-05-20", "2024-05-26"
STATUS = "DRAFT - PENDING METEOROLOGIST APPROVAL"

TEMPLATES = {
    "citizens": {
        "Red": "{region}: SEVERE HEATWAVE - peak {peak} degC. Stay indoors 12-4 pm, drink water often, check on elderly neighbours.",
        "Orange": "{region}: Heatwave for {days} days - peak {peak} degC. Avoid outdoor work at midday and stay hydrated.",
        "Yellow": "{region}: Heatwave on 1 day - peak {peak} degC. Limit afternoon sun exposure.",
        "Green": "{region}: No heatwave expected; peak {peak} degC. Normal precautions for summer heat.",
    },
    "farmers": {
        "Red": "{region}: Severe heat (peak {peak} degC). Irrigate in the evening, shade livestock, postpone spraying.",
        "Orange": "{region}: Heatwave (peak {peak} degC). Give light, frequent irrigation and water livestock often.",
        "Yellow": "{region}: Brief heatwave (peak {peak} degC). Schedule field work for early morning.",
        "Green": "{region}: No heatwave (peak {peak} degC). Continue routine operations; monitor soil moisture.",
    },
    "health": {
        "Red": "{region}: Peak {peak} degC - activate heat-action plan, prepare ORS stocks and cooling wards.",
        "Orange": "{region}: Peak {peak} degC over {days} days - alert PHCs for heat-stroke cases.",
        "Yellow": "{region}: Peak {peak} degC on 1 day - keep heat-illness protocols ready.",
        "Green": "{region}: Peak {peak} degC - no heatwave; routine surveillance of heat illness.",
    },
}


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/imd_tmax.csv", parse_dates=["date"])
    # <<FAULT>>
    df = df[df["tmax"].notna() & (df["tmax"] < 90)].copy()  # <<CLEAN>>
    df["month"], df["day"] = df["date"].dt.month, df["date"].dt.day
    df["normal"] = df.groupby(["lat", "lon", "month", "day"])["tmax"].transform("mean")
    dep = df["tmax"] - df["normal"]
    eligible = df["tmax"] >= df["terrain"].map(THRESHOLD)
    severe = eligible & ((dep > 6.4) | (df["tmax"] >= 47))
    hw = eligible & ((dep >= 4.5) | (df["tmax"] >= 45))
    df["level"] = np.where(severe, 2, np.where(hw, 1, 0))

    w = df[(df["date"] >= START) & (df["date"] <= END)]
    regions = []
    for region, r in w.groupby("region"):
        daily = r.groupby("date")["level"].max()
        worst, days = int(daily.max()), int((daily >= 1).sum())
        sev = ["None", "Heatwave", "Severe Heatwave"][worst]
        color = "Red" if worst == 2 else ("Orange" if days >= 2 else "Yellow") if worst == 1 else "Green"
        peak = round(float(r["tmax"].max()), 1)
        texts = {k: TEMPLATES[k][color].format(region=region, peak=f"{peak:.1f}", days=days) for k in TEMPLATES}
        regions.append({"region": region, "max_severity": sev, "hw_day_count": days, "peak_tmax": peak,
                        "color_code": color, "advisories": texts})

    result = {"window": {"start": START, "end": END}, "status": STATUS,
              "requires_human_review": True, "regions": regions}  # <<SCHEMA>>
    with open("outputs/result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    lines = [STATUS, "", f"# Heatwave advisory bulletin {START} to {END}", ""]
    for x in regions:
        lines += [f"## {x['region']} - {x['color_code']} ({x['max_severity']}, peak {x['peak_tmax']:.1f} degC)"]
        lines += [f"- **{k}**: {v}" for k, v in x["advisories"].items()] + [""]
    with open("outputs/advisories.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    for x in regions:
        print(f"{x['region']:20s} {x['color_code']:7s} {x['max_severity']:16s} peak {x['peak_tmax']}")


if __name__ == "__main__":
    main()
