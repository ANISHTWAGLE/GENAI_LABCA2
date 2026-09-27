import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REGION = "North West"
SPLIT = "2023-01-01"


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/imd_tmax.csv", parse_dates=["date"])
    # <<FAULT>>
    df = df[df["tmax"].notna() & (df["tmax"] < 90)]  # <<CLEAN>>
    s = df[df["region"] == REGION].groupby("date")["tmax"].mean()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).interpolate(limit_direction="both")

    X = pd.DataFrame({f"lag{k}": s.shift(k) for k in (1, 2, 3, 7)})
    doy = X.index.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X["y"] = s
    X = X.dropna()
    train, test = X[X.index < SPLIT], X[X.index >= SPLIT]
    feats = [c for c in X.columns if c != "y"]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=2)
    model.fit(train[feats], train["y"])
    pred = model.predict(test[feats])
    err, base_err = test["y"] - pred, test["y"] - test["lag1"]
    mae, rmse = float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))
    bmae, brmse = float(np.abs(base_err).mean()), float(np.sqrt((base_err ** 2).mean()))

    result = {"region": REGION, "model": "RandomForestRegressor", "n_train": int(len(train)),
              "n_test": int(len(test)), "mae": mae, "rmse": rmse, "baseline_mae": bmae,
              "baseline_rmse": brmse, "skill_vs_baseline": 1 - mae / bmae}  # <<SCHEMA>>
    with open("outputs/result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    window = (test.index >= "2024-03-01") & (test.index <= "2024-06-30")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(test.index[window], test["y"][window], label="observed", color="black")
    ax.plot(test.index[window], pred[window], label="random forest")
    ax.plot(test.index[window], test["lag1"][window], label="persistence", alpha=0.6)
    ax.set_ylabel("Tmax (degC)")
    ax.set_title(f"{REGION}: 1-day-ahead Tmax forecast, pre-monsoon 2024")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/forecast.png", dpi=110)
    plt.close(fig)
    print(f"MAE {mae:.3f} vs persistence {bmae:.3f} (skill {1 - mae / bmae:+.1%})")


if __name__ == "__main__":
    main()
