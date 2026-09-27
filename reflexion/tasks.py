"""Heatwave-intelligence tasks (use case KJS-CES-01) with schema + oracle-backed validators."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from . import dataset, oracle
from .evaluator import Check

ROOT = Path(__file__).resolve().parent.parent
SANDBOX_DATA_PATH = dataset.SANDBOX_PATH
REFERENCE_DIR = ROOT / "reference_solutions"

Validator = Callable[[dict | None, Path], list[Check]]


@dataclass
class TaskSpec:
    id: str
    title: str
    prompt: str
    required_outputs: list[str]
    schema_check: Validator | None = None
    semantic_check: Validator | None = None

    @property
    def reference_path(self) -> Path:
        return REFERENCE_DIR / f"{self.id}.py"


# ----------------------------------------------------------------------------- shared text
IMD_RULES = """IMD HEATWAVE RULES (apply exactly):
1. Clean data: drop rows where tmax is NaN or tmax >= 90 (99.9 is the IMD missing-value code).
2. Normal: for each grid point (lat, lon) and calendar day (month, day), the normal is the mean
   of the cleaned tmax over all years for that grid point and calendar day.
   departure = tmax - normal.
3. Eligibility threshold by terrain: plains 40.0, coastal 37.0, hilly 30.0 degC. A day can only be a
   heatwave day if tmax >= its terrain threshold.
4. An eligible day is a HEATWAVE day if departure >= 4.5 OR tmax >= 45.
   It is a SEVERE heatwave day if departure > 6.4 OR tmax >= 47. (Severe days are also heatwave days.)
5. Seasons by month: Winter = Jan-Feb, Pre-monsoon = Mar-May, Monsoon = Jun-Sep, Post-monsoon = Oct-Dec."""


def data_context() -> str:
    """Grounding: an automatic profile of the ACTIVE input file, shown to the model."""
    ds = dataset.active()
    raw = dataset.read(ds.path)
    regions = ", ".join(sorted(raw["region"].dropna().unique()))
    dtypes = ", ".join(f"{c} ({t})" for c, t in raw.dtypes.astype(str).items())
    return (f"INPUT DATA: '{SANDBOX_DATA_PATH}' - {ds.description}.\n"
            f"Read it with: df = pd.read_csv('{SANDBOX_DATA_PATH}', parse_dates=['date'])  ({len(raw):,} rows, one row per date x "
            f"1-degree grid point, {raw['date'].min().date()}..{raw['date'].max().date()})\n"
            f"Columns and dtypes: {dtypes}\n"
            f"First rows:\n{raw.head(4).to_string(index=False)}\n"
            f"regions (7): {regions}\n"
            f"terrain values: plains, coastal, hilly. nearest_city = closest major city (city_km = distance).\n"
            f"Missing tmax is NaN (the IMD missing code 99.9 may also appear) - drop both before analysis.")


def _expect_keys(obj, keys, where) -> list[Check]:
    if not isinstance(obj, dict):
        return [Check(f"type:{where}", "SCHEMA", False, f"{where} must be a JSON object, got {type(obj).__name__}")]
    missing = [k for k in keys if k not in obj]
    return [Check(f"keys:{where}", "SCHEMA", not missing,
                  f"missing keys {missing} (present: {list(obj)[:12]})" if missing else "ok")]


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _close(name, got, want, tol, layer="SEMANTIC", rel=False) -> Check:
    if not _num(got):
        return Check(name, layer, False, f"expected a number, got {got!r}")
    lim = tol * abs(want) if rel else tol
    ok = abs(got - want) <= max(lim, 1e-9)
    return Check(name, layer, ok, f"got {got:g}, expected {want:g} (tolerance {'%g%%' % (tol*100) if rel else tol})")


# ----------------------------------------------------------------------------- 1. detection
def _hd_schema(r, _wd):
    checks = _expect_keys(r, ["regions", "total_hw_days", "total_severe_days", "total_events"], "result")
    if not checks[0].passed:
        return checks
    regs = r["regions"]
    expected = sorted(oracle.load_clean()["region"].unique())
    checks.append(Check("region_names", "SCHEMA", isinstance(regs, dict) and sorted(regs) == expected,
                        f"expected exactly {expected}, got {sorted(regs) if isinstance(regs, dict) else regs!r}"))
    if not checks[-1].passed:
        return checks
    bad = [f"{reg}/{s}" for reg, v in regs.items() for s in oracle.SEASON_ORDER
           if not isinstance(v, dict) or not isinstance(v.get(s), dict)
           or not all(isinstance(v[s].get(k), int) for k in ("hw_days", "severe_days", "events"))]
    checks.append(Check("season_entries", "SCHEMA", not bad,
                        f"missing/non-integer hw_days|severe_days|events for {bad[:4]}" if bad else "ok"))
    return checks


def _hd_semantic(r, _wd):
    t = oracle.heatwave_detection_truth()
    checks = []
    regs = r["regions"]
    s_hw = sum(v[s]["hw_days"] for v in regs.values() for s in oracle.SEASON_ORDER)
    checks.append(Check("internal_consistency", "SEMANTIC", s_hw == r["total_hw_days"],
                        f"sum of per-region hw_days = {s_hw} but total_hw_days = {r['total_hw_days']}"))
    viol = [f"{reg}/{s}" for reg, v in regs.items() for s in oracle.SEASON_ORDER
            if v[s]["severe_days"] > v[s]["hw_days"] or min(v[s].values()) < 0]
    checks.append(Check("severe<=hw", "SEMANTIC", not viol, f"severe_days > hw_days or negative in {viol[:3]}"
                        if viol else "ok"))
    checks.append(_close("total_hw_days", r["total_hw_days"], t["total_hw_days"], 0.03, rel=True))
    checks.append(_close("total_severe_days", r["total_severe_days"], t["total_severe_days"], 0.03, rel=True))
    checks.append(_close("total_events", r["total_events"], t["total_events"], 0.05, rel=True))
    off = []
    for reg, v in t["regions"].items():
        want, got = v["Pre-monsoon"]["hw_days"], regs[reg]["Pre-monsoon"]["hw_days"]
        if abs(got - want) > max(3, 0.05 * want):
            off.append(f"{reg}: got {got}, expected ~{want}")
    checks.append(Check("premonsoon_by_region", "SEMANTIC", not off, "; ".join(off[:3]) or "ok"))
    return checks


# ----------------------------------------------------------------------------- 2. forecast
def _fc_schema(r, _wd):
    keys = ["region", "model", "n_train", "n_test", "mae", "rmse", "baseline_mae", "baseline_rmse",
            "skill_vs_baseline"]
    checks = _expect_keys(r, keys, "result")
    if checks[0].passed:
        bad = [k for k in keys[2:] if not _num(r[k])]
        checks.append(Check("numeric", "SCHEMA", not bad, f"non-numeric/NaN values for {bad}" if bad else "ok"))
    return checks


def _fc_semantic(r, _wd):
    t = oracle.forecast_truth()
    return [
        Check("region", "SEMANTIC", r["region"] == oracle.FORECAST_REGION,
              f"got {r['region']!r}, expected {oracle.FORECAST_REGION!r}"),
        _close("n_test", r["n_test"], t["n_test"], 2),
        _close("baseline_mae", r["baseline_mae"], t["baseline_mae"], 0.05),
        _close("baseline_rmse", r["baseline_rmse"], t["baseline_rmse"], 0.06),
        Check("rmse>=mae", "SEMANTIC", r["rmse"] >= r["mae"] - 1e-9, f"rmse {r['rmse']} < mae {r['mae']} is impossible"),
        Check("mae_range", "SEMANTIC", 0 < r["mae"] < 5, f"mae {r['mae']} outside plausible range (0, 5) degC"),
        Check("beats_or_matches_persistence", "SEMANTIC", r["mae"] <= t["baseline_mae"] * 1.05,
              f"model MAE {r['mae']:.3f} is worse than persistence {t['baseline_mae']:.3f} (+5% allowed)"),
        _close("skill_formula", r["skill_vs_baseline"], 1 - r["mae"] / r["baseline_mae"], 0.01),
    ]


# ----------------------------------------------------------------------------- 3. hotspots
def _hs_schema(r, _wd):
    checks = _expect_keys(r, ["n_gridpoints", "hotspots", "severity_counts"], "result")
    if not checks[0].passed:
        return checks
    hs = r["hotspots"]
    checks.append(Check("hotspots_len", "SCHEMA", isinstance(hs, list) and len(hs) == 10,
                        f"'hotspots' must be a list of 10 items, got {len(hs) if isinstance(hs, list) else hs!r}"))
    if checks[-1].passed:
        need = ["rank", "lat", "lon", "region", "hw_days_per_year", "severity"]
        bad = [i for i, h in enumerate(hs) if not isinstance(h, dict) or any(k not in h for k in need)]
        checks.append(Check("hotspot_fields", "SCHEMA", not bad, f"items {bad[:3]} lack some of {need}" if bad else "ok"))
    sc = r["severity_counts"]
    checks.append(Check("severity_counts_keys", "SCHEMA",
                        isinstance(sc, dict) and set(sc) == {"Low", "Moderate", "High", "Extreme"},
                        f"expected keys Low/Moderate/High/Extreme, got {list(sc) if isinstance(sc, dict) else sc!r}"))
    return checks


def _hs_semantic(r, _wd):
    t = oracle.hotspot_truth()
    table = t["table"].set_index(["lat", "lon"])
    hs = r["hotspots"]
    vals = [h["hw_days_per_year"] for h in hs]
    checks = [
        _close("n_gridpoints", r["n_gridpoints"], t["n_gridpoints"], 0),
        Check("ranks", "SEMANTIC", [h["rank"] for h in hs] == list(range(1, 11)), "ranks must be 1..10 in order"),
        Check("sorted_desc", "SEMANTIC", all(a >= b for a, b in zip(vals, vals[1:])), "hotspots not sorted descending"),
    ]
    top3 = set(map(tuple, t["table"][["lat", "lon"]].head(3).to_numpy().tolist()))
    checks.append(Check("top_hotspot", "SEMANTIC", (float(hs[0]["lat"]), float(hs[0]["lon"])) in top3,
                        f"rank-1 hotspot {hs[0]['lat']},{hs[0]['lon']} is not among true top-3 {sorted(top3)}"))
    wrong = []
    for h in hs:
        key = (float(h["lat"]), float(h["lon"]))
        if key not in table.index:
            wrong.append(f"{key} is not a grid point")
        elif abs(h["hw_days_per_year"] - table.loc[key, "hw_days_per_year"]) > 0.3:
            wrong.append(f"{key}: {h['hw_days_per_year']} vs expected {table.loc[key, 'hw_days_per_year']:.2f}")
    checks.append(Check("hotspot_values", "SEMANTIC", not wrong, "; ".join(wrong[:3]) or "ok"))
    sev = [f"rank {h['rank']}: {h['severity']} should be {oracle.severity_class(h['hw_days_per_year'])}"
           for h in hs if h["severity"] != oracle.severity_class(h["hw_days_per_year"])]
    checks.append(Check("severity_classes", "SEMANTIC", not sev, "; ".join(sev[:3]) or "ok"))
    diff = {k: (r["severity_counts"][k], v) for k, v in t["severity_counts"].items() if r["severity_counts"][k] != v}
    checks.append(Check("severity_counts", "SEMANTIC", not diff,
                        f"(got, expected) = {diff}" if diff else "ok"))
    return checks


# ----------------------------------------------------------------------------- 4. advisory
STAKEHOLDERS = ["citizens", "farmers", "health"]


def _adv_schema(r, _wd):
    checks = _expect_keys(r, ["window", "status", "requires_human_review", "regions"], "result")
    if not checks[0].passed:
        return checks
    regs = r["regions"]
    checks.append(Check("regions_len", "SCHEMA", isinstance(regs, list) and len(regs) == 7,
                        f"'regions' must be a list of 7 objects, got {len(regs) if isinstance(regs, list) else regs!r}"))
    if checks[-1].passed:
        need = ["region", "max_severity", "hw_day_count", "peak_tmax", "color_code", "advisories"]
        bad = [x.get("region", i) if isinstance(x, dict) else i for i, x in enumerate(regs)
               if not isinstance(x, dict) or any(k not in x for k in need)
               or not isinstance(x.get("advisories"), dict)
               or any(not isinstance(x["advisories"].get(s), str) for s in STAKEHOLDERS)]
        checks.append(Check("region_fields", "SCHEMA", not bad,
                            f"{bad[:3]} lack {need} or advisories.{{citizens,farmers,health}} strings" if bad else "ok"))
    return checks


def _adv_semantic(r, wd):
    t = oracle.advisory_truth()
    checks = [
        Check("human_review_flag", "SEMANTIC", r["requires_human_review"] is True,
              "requires_human_review must be true: advisories need meteorologist approval before dissemination"),
        Check("status_draft", "SEMANTIC", "APPROVAL" in str(r["status"]).upper(),
              f"status must state that approval is pending, got {r['status']!r}"),
        Check("window", "SEMANTIC", isinstance(r["window"], dict) and
              (r["window"].get("start"), r["window"].get("end")) == oracle.ADVISORY_WINDOW,
              f"window must be {{'start': '{oracle.ADVISORY_WINDOW[0]}', 'end': '{oracle.ADVISORY_WINDOW[1]}'}}"),
    ]
    wrong, facts = [], []
    got = {x["region"]: x for x in r["regions"]}
    for region, want in t.items():
        x = got.get(region)
        if x is None:
            wrong.append(f"{region} missing")
            continue
        for k in ("max_severity", "hw_day_count", "color_code"):
            if x[k] != want[k]:
                wrong.append(f"{region}.{k}: got {x[k]!r}, expected {want[k]!r}")
        if not _num(x["peak_tmax"]) or abs(x["peak_tmax"] - want["peak_tmax"]) > 0.05:
            wrong.append(f"{region}.peak_tmax: got {x['peak_tmax']!r}, expected {want['peak_tmax']}")
        peak_txt = f"{want['peak_tmax']:.1f}"
        for s in STAKEHOLDERS:
            text = x["advisories"][s]
            if len(text.strip()) < 30 or peak_txt not in text:
                facts.append(f"{region}/{s}")
    checks.append(Check("oracle_agreement", "SEMANTIC", not wrong, "; ".join(wrong[:4]) or "ok"))
    checks.append(Check("factual_consistency", "SEMANTIC", not facts,
                        f"advisory text must be >=30 chars and quote the region's peak_tmax with 1 decimal "
                        f"(e.g. '46.0'); failing: {facts[:4]}" if facts else "ok"))
    md = wd / "outputs" / "advisories.md"
    checks.append(Check("markdown_banner", "SEMANTIC", md.exists() and "APPROVAL" in md.read_text("utf-8", "replace").upper(),
                        "outputs/advisories.md must carry the pending-approval banner"))
    return checks


# ----------------------------------------------------------------------------- registry
TASKS: dict[str, TaskSpec] = {
    "heatwave_detection": TaskSpec(
        "heatwave_detection", "IMD heatwave detection by region and season",
        f"""Detect heatwave days and heatwave events for every IMD region and season.

{IMD_RULES}

Definitions:
- hw_days: number of grid-point-days classified as heatwave (including severe).
- severe_days: number of grid-point-days classified as severe heatwave.
- events: a run of >= 2 consecutive calendar days that are heatwave days at the SAME grid point
  counts as one event; assign it to the season of the run's first day.

Write outputs/result.json with EXACTLY this structure (all counts are ints):
{{"regions": {{"<region>": {{"Winter": {{"hw_days": 0, "severe_days": 0, "events": 0}},
                           "Pre-monsoon": {{...}}, "Monsoon": {{...}}, "Post-monsoon": {{...}}}}, ... all 7 regions}},
 "total_hw_days": 0, "total_severe_days": 0, "total_events": 0}}
Also save a grouped bar chart of hw_days per region and season to outputs/heatwave_days_by_region.png.""",
        ["outputs/result.json", "outputs/heatwave_days_by_region.png"], _hd_schema, _hd_semantic),

    "tmax_forecast": TaskSpec(
        "tmax_forecast", "1-day-ahead Tmax forecasting (North West) vs persistence",
        f"""Build a 1-day-ahead maximum-temperature forecast for the region "{oracle.FORECAST_REGION}".

1. Clean data: drop rows where tmax is NaN or tmax >= 90 (99.9 = missing code).
2. Regional series: daily mean of cleaned tmax over the region's grid points. Reindex to every
   calendar day and fill gaps by linear interpolation.
3. Features for day t: lag1, lag2, lag3, lag7 (the series shifted by 1, 2, 3, 7 days) and
   sin/cos of day-of-year. Target: the series value on day t. Drop rows with missing lags.
4. Split by date: train = dates before {oracle.FORECAST_SPLIT}, test = dates from {oracle.FORECAST_SPLIT} on.
5. Train any scikit-learn regressor (e.g. RandomForestRegressor(n_estimators=200, random_state=42)).
6. Baseline = persistence (prediction for day t is lag1). Compute MAE and RMSE on the test set for
   the model and for the baseline. skill_vs_baseline = 1 - mae / baseline_mae.

Write outputs/result.json:
{{"region": "{oracle.FORECAST_REGION}", "model": "<name>", "n_train": int, "n_test": int, "mae": float, "rmse": float,
 "baseline_mae": float, "baseline_rmse": float, "skill_vs_baseline": float}}
Save outputs/forecast.png: actual vs model vs persistence for 2024-03-01..2024-06-30.""",
        ["outputs/result.json", "outputs/forecast.png"], _fc_schema, _fc_semantic),

    "hotspot_severity": TaskSpec(
        "hotspot_severity", "Heatwave hotspot identification and severity classification",
        f"""Identify heatwave hotspots and classify every grid point by heatwave severity.

{IMD_RULES}

For each grid point: hw_days_per_year = (number of heatwave days, including severe) / (number of
distinct years in the data), rounded to 2 decimals. Severity class of a grid point from
hw_days_per_year: Extreme if >= 10, High if >= 5, Moderate if >= 2, else Low.

Write outputs/result.json:
{{"n_gridpoints": int,
 "hotspots": [ the 10 grid points with the highest hw_days_per_year, sorted descending:
   {{"rank": 1, "lat": float, "lon": float, "region": str, "hw_days_per_year": float, "severity": str}}, ... ],
 "severity_counts": {{"Low": int, "Moderate": int, "High": int, "Extreme": int}}  (over ALL grid points)}}
Save outputs/hotspot_map.png: scatter of lon vs lat coloured by hw_days_per_year with a colorbar.""",
        ["outputs/result.json", "outputs/hotspot_map.png"], _hs_schema, _hs_semantic),

    "advisory_json": TaskSpec(
        "advisory_json", "Stakeholder advisories for the 2024-05-20..26 window (human review required)",
        f"""Prepare DRAFT heatwave advisories for the window {oracle.ADVISORY_WINDOW[0]} to {oracle.ADVISORY_WINDOW[1]}
(inclusive), one entry per region.

{IMD_RULES}
(Normals use ALL years of data; then look only at the window.)

Per region, per day in the window: the region's level that day is the worst level over its grid
points (0 = none, 1 = heatwave, 2 = severe).
- max_severity: "Severe Heatwave" if the worst level in the window is 2, "Heatwave" if 1, else "None".
- hw_day_count: number of window days whose regional level is >= 1.
- peak_tmax: highest cleaned tmax in the region during the window, rounded to 1 decimal.
- color_code (IMD colour scheme): "Red" if max_severity is Severe Heatwave; "Orange" if Heatwave and
  hw_day_count >= 2; "Yellow" if Heatwave and hw_day_count == 1; otherwise "Green".
- advisories: short template-generated texts for "citizens", "farmers" and "health" (health departments/
  hospitals). Each text must name the region and quote peak_tmax with exactly 1 decimal (e.g. "46.0").

Write outputs/result.json:
{{"window": {{"start": "{oracle.ADVISORY_WINDOW[0]}", "end": "{oracle.ADVISORY_WINDOW[1]}"}},
 "status": "DRAFT - PENDING METEOROLOGIST APPROVAL", "requires_human_review": true,
 "regions": [{{"region": str, "max_severity": str, "hw_day_count": int, "peak_tmax": float,
              "color_code": str, "advisories": {{"citizens": str, "farmers": str, "health": str}}}}, ... 7 regions]}}
Also write outputs/advisories.md: a readable bulletin that starts with the line
"DRAFT - PENDING METEOROLOGIST APPROVAL" followed by one section per region.""",
        ["outputs/result.json", "outputs/advisories.md"], _adv_schema, _adv_semantic),
}


def custom_task(prompt: str) -> TaskSpec:
    """Free-form task: only the generic layers (static, runtime, and a result.json artifact) apply."""
    return TaskSpec("custom", "Custom task", prompt + "\n\nWrite your main results to outputs/result.json "
                    "(a JSON object) and any figures as PNG files under outputs/.", ["outputs/result.json"])
