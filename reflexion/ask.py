"""Ask mode: a plain-English question from an official -> self-fixing code -> verified answer.

The Reflexion loop is unchanged; only the task is built from the user's question. Because a
free-form question has no oracle, correctness is checked in two ways:
  1. SANITY checks on answer.json (physically plausible values, dates inside the data range,
     the answer text agrees with the computed result, the chart is a real image);
  2. SELF-CONSISTENCY (optional): a second, independent loop solves the same question from
     scratch; if both computed results agree the answer is marked HIGH confidence, otherwise
     it is flagged LOW and both answers are shown for a human to judge.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import dataset
from .evaluator import Check
from .loop import LoopConfig, LoopResult, ReflexionLoop
from .tasks import TaskSpec

ASK_PROMPT = """Answer this question from a government official using the temperature data:

QUESTION: "{question}"

Rules:
- Clean the data first (drop NaN tmax and tmax >= 90).
- Where the question needs heatwave definitions, use the IMD rules below.
- The data only has daily MAXIMUM temperature on a 1-degree grid (no rainfall, humidity, minimum
  temperature or district boundaries). If the question asks for something the data cannot give,
  answer the closest thing it CAN give and state the limitation plainly in "answer".
- Relative dates ("last year", "recently") are relative to the END of the dataset.
- Thresholds: "above/over/more than X" means > X; "at least/X or more" means >= X; "below/under" means < X.
  Say which comparison you used in "method".
- "The area near <city>" means all grid points whose nearest_city is that city; a day counts once if ANY of
  those points meets the condition. Say this in "method".
- Refer to places by region and nearest_city so a non-technical reader understands.
- Counting days for a place or region: count CALENDAR days (a date counts once even if several grid
  points qualify). Never add up grid-point-days unless the question asks for that.

{rules}

Write outputs/answer.json with exactly these keys:
{{"answer": "1-4 plain-English sentences for a non-technical official, with the key numbers and units",
 "result": <the computed result: a single number, OR a short string, OR a list of at most 20
            objects {{"label": str, "value": number}} sorted by importance. Every label must be UNIQUE and
            name its row: for per-year results use the year ("2015"); for places use
            "<nearest_city> (<region>)", e.g. "Jaipur (North West)">,
 "unit": "<unit of the numbers, e.g. degC, days, %, or null>",
 "method": "1-2 sentences: exactly how it was computed (filters, dates, definitions)",
 "chart": "outputs/chart.png" or null}}
If a chart helps (trends, rankings, maps), save it to outputs/chart.png (matplotlib, Agg) and set "chart".
Every number in "answer" must come from the computed result."""

INDEPENDENT_NOTE = ("\n\nINDEPENDENT VERIFICATION RUN: solve this from scratch with your own approach; "
                    "another analyst's answer will be compared with yours.")


# ----------------------------------------------------------------------------- checks
def _numbers(result) -> list[float]:
    if isinstance(result, bool):
        return []
    if isinstance(result, (int, float)):
        return [float(result)]
    if isinstance(result, list):
        return [float(r["value"]) for r in result if isinstance(r, dict) and isinstance(r.get("value"), (int, float))
                and not isinstance(r.get("value"), bool)]
    return []


def _ask_schema(r, _wd) -> list[Check]:
    if not isinstance(r, dict):
        return [Check("type", "SCHEMA", False, f"answer.json must be an object, got {type(r).__name__}")]
    missing = [k for k in ("answer", "result", "unit", "method", "chart") if k not in r]
    checks = [Check("keys", "SCHEMA", not missing, f"missing keys {missing}" if missing else "ok")]
    if missing:
        return checks
    res = r["result"]
    ok_type = (isinstance(res, (int, float, str)) and not isinstance(res, bool)) or (
        isinstance(res, list) and 0 < len(res) <= 20
        and all(isinstance(x, dict) and isinstance(x.get("label"), str) for x in res))
    checks.append(Check("result_type", "SCHEMA", ok_type,
                        "result must be a number, a string, or a list of 1-20 {label, value} objects"
                        if not ok_type else "ok"))
    checks.append(Check("answer_text", "SCHEMA", isinstance(r["answer"], str) and 10 <= len(r["answer"].strip()) <= 1500,
                        "answer must be a 10-1500 character plain-English string"))
    return checks


def _ask_sanity(r, wd: Path, question: str = "") -> list[Check]:
    nums = _numbers(r["result"])
    unit = str(r.get("unit") or "").lower()
    checks = [Check("finite", "SEMANTIC", all(math.isfinite(x) for x in nums),
                    "result contains NaN or infinity - the computation lost data")]
    if nums and ("c" == unit.strip() or "degc" in unit or "°c" in unit or "celsius" in unit):
        bad = [x for x in nums if not -15 <= x <= 60]
        checks.append(Check("temperature_range", "SEMANTIC", not bad,
                            f"temperatures {bad[:3]} are physically implausible for India (expected -15..60 degC) - "
                            "probably missing values (99.9) or a unit error" if bad else "ok"))
    if nums and ("day" in unit or "count" in unit):
        checks.append(Check("counts_nonnegative", "SEMANTIC", all(x >= 0 for x in nums), "counts cannot be negative"))
        limit = max_days_in_question(question) if "day" in unit else None
        if limit:
            over = [x for x in nums if x > limit]
            checks.append(Check("days_within_period", "SEMANTIC", not over,
                                f"{over[:3]} days is impossible: the period in the question has only {limit} days. "
                                "You probably added up several grid points - count each calendar date once "
                                "per place." if over else "ok"))
    if nums and unit.strip() in {"%", "percent"}:
        checks.append(Check("percent_range", "SEMANTIC", all(-100 <= x <= 100 for x in nums),
                            "percentages outside -100..100"))

    raw = dataset.read()
    lo, hi = raw["date"].min().date().isoformat(), raw["date"].max().date().isoformat()
    text = " ".join(str(r.get(k, "")) for k in ("answer", "method")) + " " + json.dumps(r["result"])
    out_of_range = [d for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", text) if not lo <= d <= hi]
    checks.append(Check("dates_in_range", "SEMANTIC", not out_of_range,
                        f"dates {out_of_range[:3]} are outside the data ({lo}..{hi})" if out_of_range else "ok"))

    if isinstance(r["result"], list):
        empty = [x.get("label") for x in r["result"] if isinstance(x, dict)
                 and not (isinstance(x.get("value"), (int, float)) and not isinstance(x.get("value"), bool))]
        checks.append(Check("values_present", "SEMANTIC", not empty,
                            f"rows {empty[:3]} have no numeric value (null/None/text) - every row needs its computed "
                            "number" if empty else "ok"))
        labels = [_norm_label(x.get("label", "")) for x in r["result"] if isinstance(x, dict)]
        dups = sorted({l for l in labels if labels.count(l) > 1})
        checks.append(Check("unique_labels", "SEMANTIC", not dups,
                            f"labels {dups[:3]} appear more than once - each row needs its own label (e.g. the year "
                            "for per-year results), otherwise a reader cannot tell the rows apart" if dups else "ok"))

    answer = r["answer"]
    if isinstance(r["result"], (int, float)) and not isinstance(r["result"], bool):
        v = float(r["result"])
        found = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))]
        ok = any(abs(x - v) <= max(0.051, 0.01 * abs(v)) for x in found)
        checks.append(Check("answer_matches_result", "SEMANTIC", ok,
                            f"the answer text does not state the computed result {v:g}" if not ok else "ok"))
    elif isinstance(r["result"], list):
        labels = [str(x["label"]).lower() for x in r["result"]]
        ok = any(l and l in answer.lower() for l in labels[:5])
        checks.append(Check("answer_matches_result", "SEMANTIC", ok,
                            "the answer text mentions none of the top result labels" if not ok else "ok"))
    if r.get("chart"):
        p = wd / str(r["chart"])
        ok = p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        checks.append(Check("chart", "SEMANTIC", ok, f"chart '{r['chart']}' is missing or not a PNG" if not ok else "ok"))
    return checks


def ask_task(question: str, independent: bool = False) -> TaskSpec:
    from .tasks import IMD_RULES
    prompt = ASK_PROMPT.format(question=question.strip(), rules=IMD_RULES)
    return TaskSpec("ask", question.strip()[:80], prompt + (INDEPENDENT_NOTE if independent else ""),
                    ["outputs/answer.json"], _ask_schema, lambda r, wd: _ask_sanity(r, wd, question))


# ----------------------------------------------------------------------------- self-consistency
def _norm_label(x) -> str:
    """'Western Himalaya – Leh' == 'western himalaya - leh': ignore case, dashes, punctuation, spacing."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(x).lower()).split())


MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                      "august", "september", "october", "november", "december"], 1)}


def max_days_in_question(question: str) -> int | None:
    """Upper bound on a per-place day count, from 'May 2024' / 'in 2023' style periods in the question."""
    import calendar
    q = question.lower()
    m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(20\d\d)\b", q)
    if m:
        return calendar.monthrange(int(m.group(2)), MONTHS[m.group(1)])[1]
    years = re.findall(r"\b(20\d\d)\b", q)
    if len(years) == 1 and not re.search(r"since|from|between|trend|to 20", q):
        return 366
    return None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.1, 0.02 * max(abs(a), abs(b)))


def _as_number(x):
    """A single numeric answer, whatever format it was stored in: 46.3, [{"value": 46.3}], "46.3 degC"."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], dict):
        return _as_number(x[0].get("value"))
    if isinstance(x, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", x.replace(",", ""))
        return float(nums[0]) if len(nums) == 1 else None
    return None


def _match_by_label(ra: list, rb: list):
    """Non-ranked lists (e.g. per-year counts): pair rows by label - equal word sets, or one label contained in
    the other ('2015' ~ 'Delhi (North West) 2015'). Rows missing from one side are fine only if their value is 0
    (some solutions omit zero rows). Returns (agree, detail), or None if the rows cannot be paired."""
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    da, db = dict(ra), dict(rb)
    if len(da) < len(ra) or len(db) < len(rb):
        return None  # duplicate labels: rows cannot be paired reliably
    pairs, used = [], set()
    for la in da:
        m = next((lb for lb in db if lb not in used and (la == lb or la <= lb or lb <= la)), None)
        if m is not None:
            pairs.append((la, m))
            used.add(m)
    if len(pairs) < 0.8 * min(len(da), len(db)):
        return None
    diff = [la for la, lb in pairs if not (num(da[la]) and num(db[lb]))  # a missing value never "agrees"
            or not _close(float(da[la]), float(db[lb]))]
    paired_a = {p[0] for p in pairs}
    extra = [v for l, v in da.items() if l not in paired_a] + [v for l, v in db.items() if l not in used]
    extra_ok = all(num(v) and float(v) == 0 for v in extra)
    ok = not diff and extra_ok
    return ok, (f"rows paired by label: {len(pairs)} matched, {len(extra)} unmatched"
                + (" (all zero)" if extra and extra_ok else "") + f", values {'agree' if not diff else 'differ'}")


def compare(a, b) -> tuple[bool, str]:
    """Do two independently computed results agree?"""
    if type(a) is not type(b):  # same answer stored in different formats (number vs one-item list)
        na, nb = _as_number(a), _as_number(b)
        if na is not None and nb is not None:
            return _close(na, nb), f"{na:g} vs {nb:g} (stored in different formats)"
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b, "boolean results"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _close(float(a), float(b)), f"{a:g} vs {b:g}"
    if isinstance(a, str) and isinstance(b, str):
        x, y = _norm_label(a), _norm_label(b)
        return (x == y or x in y or y in x), f"'{a}' vs '{b}'"
    if isinstance(a, list) and isinstance(b, list):
        # labels compared as word SETS: "Leh (Western Himalaya)" == "Western Himalaya - Leh"
        def rows(x):
            return [(frozenset(_norm_label(r.get("label", "")).split()), r.get("value"))
                    for r in x if isinstance(r, dict)]
        ra, rb = rows(a), rows(b)
        k = min(len(ra), len(rb), 10)
        if k == 0:
            return False, "empty result"
        ra, rb = ra[:k], rb[:k]
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        # 1) the ranked values must match position by position
        values_ok = all(num(x[1]) and num(y[1]) and _close(float(x[1]), float(y[1])) for x, y in zip(ra, rb)) \
            if all(num(x[1]) for x in ra + rb) else False  # missing values never "agree"
        # 2) labels must match, except items tied with the cut-off value (either answer is valid)
        la, lb = {x[0] for x in ra}, {y[0] for y in rb}
        cutoff = ra[-1][1]
        unmatched = [x for x in ra if x[0] not in lb and not (num(x[1]) and num(cutoff) and _close(float(x[1]), float(cutoff)))]
        overlap = len(la & lb) / len(la | lb)
        ok = values_ok and not unmatched
        detail = f"top-{k} same places {overlap:.0%}" + (" (difference only in a tie at the cut-off)" if ok and overlap < 1 else "") \
            + f", values {'agree' if values_ok else 'differ'}"
        if not ok:
            by_label = _match_by_label(rows(a), rows(b))
            if by_label is not None:
                return by_label
        return ok, detail
    return False, f"different result types ({type(a).__name__} vs {type(b).__name__})"


def _unit(ans: dict) -> str:
    u = re.sub(r"[^a-z%]", "", str(ans.get("unit") or "").lower().replace("°", "deg"))
    return {"c": "degc", "celsius": "degc", "percent": "%", "day": "days"}.get(u, u)


def _facts(text: str) -> tuple[list[float], set[str]]:
    """Key facts in a written answer: its numbers and the months it names."""
    t = str(text).lower().replace(",", "")
    t = re.sub(r"\b(20\d\d)-(\d\d)-(\d\d)\b", r"\1 \2 \3", t)
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]
    return nums, {m for m in MONTHS if re.search(rf"\b{m}\b", t)}


def compare_answers(a: dict, b: dict) -> tuple[bool, str]:
    """Compare two independent answers. If both stored the same kind of value, compare the computed
    results; if they stored different kinds (e.g. month number vs temperature), compare the facts in the
    written answers instead - the stored numbers are not comparable."""
    ok, detail = compare(a.get("result"), b.get("result"))
    if ok or _unit(a) == _unit(b):  # same kind of value -> the stored results are directly comparable
        return ok, detail
    (na, ma), (nb, mb) = _facts(a.get("answer", "")), _facts(b.get("answer", ""))
    if not na and not ma:
        return ok, detail
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    matched = sum(any(abs(x - y) <= max(0.15, 0.02 * abs(x)) for y in long_) for x in short)
    nums_ok = not short or matched / len(short) >= 0.8
    months_ok = ma == mb
    agree = nums_ok and months_ok
    return agree, (f"stored different kinds of values ({_unit(a) or 'none'} vs {_unit(b) or 'none'}), so the written "
                   f"answers were compared: {matched}/{len(short)} numbers match, months "
                   f"{'match' if months_ok else 'differ'}")


@dataclass
class AskResult:
    question: str
    primary: LoopResult
    check: LoopResult | None = None
    agreement: bool | None = None
    agreement_detail: str = ""
    answer: dict | None = None
    check_answer: dict | None = None
    tiebreak: LoopResult | None = None       # third solution, run only when the first two disagree
    tiebreak_answer: dict | None = None
    swapped: bool = False                    # the second solution won the vote and is now the main answer
    events: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        if not self.primary.success:
            return "FAILED"
        if self.check is None:
            return "UNVERIFIED"
        if not self.check.success:
            return "UNVERIFIED"
        return "HIGH" if self.agreement else "LOW"

    @property
    def outputs_dir(self) -> Path | None:
        return self.primary.attempts[-1].run.workdir / "outputs" if self.primary.success else None


def _load_answer(r: LoopResult) -> dict | None:
    if not r.success:
        return None
    return r.attempts[-1].evaluation.result


def verify_answer(res: AskResult, llm, cfg: LoopConfig, *, on_event=None, pause_s: float = 0.0) -> AskResult:
    """Self-consistency check. A second independent solution must agree with the first; if they disagree,
    a third solution breaks the tie and the answer that 2 of 3 solutions agree on becomes the main answer."""
    emit = on_event or (lambda k, d: None)
    if not res.primary.success:
        return res
    q = res.question

    def solve(label: str, temperature: float) -> LoopResult:
        time.sleep(pause_s)
        emit("phase", {"name": label})
        return ReflexionLoop(ask_task(q, independent=True), llm, LoopConfig(**{**cfg.__dict__, "temperature": temperature}),
                             on_event=on_event).run()

    res.check = solve("INDEPENDENT VERIFICATION", 0.7)
    res.check_answer = _load_answer(res.check)
    if not res.check_answer:
        return res
    res.agreement, res.agreement_detail = compare_answers(res.answer, res.check_answer)
    if res.agreement:
        return res
    res.tiebreak = solve("TIE-BREAKER (third solution)", 0.9)
    res.tiebreak_answer = _load_answer(res.tiebreak)
    if not res.tiebreak_answer:
        res.agreement_detail += "; the tie-breaker did not finish"
        return res
    with_first, d1 = compare_answers(res.answer, res.tiebreak_answer)
    with_second, d2 = compare_answers(res.check_answer, res.tiebreak_answer)
    if with_first:
        res.agreement, res.agreement_detail = True, f"2 of 3 solutions agree - a third solution confirms the first ({d1})"
    elif with_second:
        res.primary, res.check = res.check, res.primary
        res.answer, res.check_answer = res.check_answer, res.answer
        res.swapped = True
        res.agreement, res.agreement_detail = True, (f"2 of 3 solutions agree - a third solution confirms the second, "
                                                     f"which is now the main answer ({d2})")
    else:
        res.agreement_detail += "; a third solution agrees with neither"
    return res


def answer_question(question: str, llm, cfg: LoopConfig, *, verify: bool = True, reviewer=None,
                    on_event=None) -> AskResult:
    emit = on_event or (lambda k, d: None)
    emit("phase", {"name": "PRIMARY SOLUTION"})
    primary = ReflexionLoop(ask_task(question), llm, cfg, reviewer=reviewer, on_event=on_event).run()
    res = AskResult(question, primary, answer=_load_answer(primary))
    if verify:
        verify_answer(res, llm, cfg, on_event=on_event)
    run_dir = primary.run_dir
    (run_dir / "ask_summary.json").write_text(json.dumps({
        "question": question, "confidence": res.confidence, "agreement_detail": res.agreement_detail,
        "answer": res.answer, "independent_answer": res.check_answer, "tiebreak_answer": res.tiebreak_answer,
        "second_solution_won_vote": res.swapped,
        "primary_run": str(primary.run_dir), "check_run": str(res.check.run_dir) if res.check else None,
    }, indent=2, default=str), encoding="utf-8")
    emit("ask_done", {"result": res})
    return res
