"""Prompt construction and response parsing for the GENERATE and REFLECT states.

Context-window strategy (what the model sees on attempt k):
  * task + data profile                          - constant
  * Reflexion memory: <= 5 one-line lessons      - distilled, cumulative, deduplicated
  * attempt history table: 1 line per attempt    - shows the trajectory, prevents cycling
  * ONLY the latest attempt in full (code + formatted error + its reflection)
Older code is never replayed, so prompt size grows by ~1 line per attempt, not by a whole script.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import feedback as fb
from .evaluator import Evaluation
from .guard import ALLOWED_MODULES
from .sandbox import ExecutionResult

ACTOR_SYSTEM = """You are a senior Python data scientist on the India Meteorological Department (IMD) heatwave \
intelligence team. You write ONE complete, self-contained Python 3.12 script that runs non-interactively in a \
locked-down sandbox.

SANDBOX RULES (violations are blocked and count as failed attempts):
- Allowed imports ONLY: {modules}.
- No network, no subprocess/shell, no eval/exec, no input(). Use only the local data file.
- Read input only from the path given. Write files ONLY under ./outputs/ (os.makedirs("outputs", exist_ok=True)).
- matplotlib: call matplotlib.use("Agg") before importing pyplot; never plt.show(); plt.savefig(...) then plt.close().
- Hard limits: {timeout:.0f} s wall-clock, {memory} MB RAM. Use vectorised pandas/numpy, never row-by-row loops.
- Convert numpy scalars to int/float before json.dump.
- Finish by printing a short summary (a few lines) to stdout.
- Mark each stage of the script with a plain-English comment a non-programmer understands, exactly in the
  form "# Step 1: Load the temperature data", "# Step 2: ...". Use 3-8 steps.

RESPONSE FORMAT: exactly one fenced block ```python ... ``` containing the whole script. No text outside it."""

REFLECTOR_SYSTEM = """You are a meticulous debugging reviewer for data-analysis scripts. You receive a task, \
a failed script and structured evaluator feedback. Find the ROOT CAUSE, not the symptom. Do NOT write code.

Reply in exactly this format:
DIAGNOSIS: <1-3 sentences naming the root cause and the failing line>
FIX_PLAN: <1-3 short, concrete steps>
LESSON: <one general rule, <= 25 words, that prevents this class of mistake in future attempts>"""


@dataclass
class Reflection:
    diagnosis: str
    fix_plan: str
    lesson: str
    raw: str = ""


@dataclass
class AttemptRecord:
    n: int
    code: str
    run: ExecutionResult | None
    evaluation: Evaluation | None
    reflection: Reflection | None = None
    injected: str | None = None

    def history_row(self) -> str:
        ev = self.evaluation
        if ev is None:
            return f"| {self.n} | - | (no evaluation) |"
        what = "PASSED" if ev.passed else ev.headline()
        return f"| {self.n} | {ev.stage} | {what[:150].replace('|', '/')} |"


@dataclass
class Memory:
    """Reflexion episodic memory: short verbal lessons carried across attempts."""
    lessons: list[str] = field(default_factory=list)
    max_items: int = 5

    def add(self, lesson: str) -> None:
        lesson = lesson.strip()
        key = fb.normalise(lesson.lower())
        if lesson and all(fb.normalise(l.lower()) != key for l in self.lessons):
            self.lessons.append(lesson)
            self.lessons = self.lessons[-self.max_items:]


def actor_system(timeout: float, memory_mb: int) -> str:
    mods = ", ".join(sorted(ALLOWED_MODULES - {"__future__"}))
    return ACTOR_SYSTEM.format(modules=mods, timeout=timeout, memory=memory_mb)


def format_failure(attempt: AttemptRecord, ctx: dict) -> str:
    """The evaluator feedback block for one failed attempt - the core 'error trace' formatting."""
    ev, run = attempt.evaluation, attempt.run
    out = [f"FAILED AT LAYER: {ev.stage}"]
    if ev.error:
        e = ev.error
        out.append(f"ERROR: {e.type}: {fb.trim_middle(e.message, 600)}")
        if e.line:
            out.append(f"WHERE: solution.py line {e.line}" + (f", in {e.func}()" if e.func and e.func != "<module>" else ""))
            out.append(fb.code_excerpt(attempt.code, e.line))
        if e.library_frame:
            out.append(f"RAISED INSIDE LIBRARY: {e.library_frame}")
        if e.cause and not e.cause.startswith(f"{e.type}: {e.message}"[:40]):
            out.append(f"UNDERLYING CAUSE: {e.cause}")
        if hint := fb.hint_for(e.type, e.message, ctx):
            out.append(f"HINT: {hint}")
    if ev.stage == "STATIC" and run and run.guard and run.guard.violations:
        out.append("FORBIDDEN CONSTRUCTS:\n" + "\n".join(f"  - {v}" for v in run.guard.violations[:6]))
        out.append(f"HINT: {fb.hint_for('BLOCKED', '', ctx)}")
    if run and run.violations:
        out.append("SANDBOX BLOCKED AT RUNTIME:\n" + "\n".join(f"  - {v['event']}: {v['detail'][:160]}"
                                                           for v in run.violations[:4]))
    failed = ev.failed_checks
    if failed and not ev.error:
        out.append("FAILED CHECKS (expected vs got):\n" + "\n".join(f"  - {c.name}: {c.detail[:300]}" for c in failed[:8]))
        passed = [c.name for c in ev.checks if c.passed]
        if passed:
            out.append(f"Checks already passing (keep them passing): {', '.join(passed[:12])}")
    if run and ev.stage in {"RUNTIME", "ARTIFACT", "SCHEMA", "SEMANTIC"}:
        if run.stdout.strip():
            out.append("STDOUT (trimmed):\n" + fb.trim_middle(run.stdout, 800))
        if ev.error is None and run.stderr.strip():
            out.append("STDERR (trimmed):\n" + fb.trim_middle(run.stderr, 800))
    if run and run.guard and run.guard.warnings:
        out.append("STATIC WARNINGS: " + "; ".join(str(w) for w in run.guard.warnings[:3]))
    return "\n".join(out)


def generate_messages(task, data_ctx: str, sys_prompt: str, history: list[AttemptRecord], memory: Memory,
                      ctx: dict) -> list[dict]:
    parts = [f"# TASK\n{task.prompt}", f"# DATA\n{data_ctx}"]
    if history:
        last = history[-1]
        if memory.lessons:
            parts.append("# REFLEXION MEMORY (lessons from your earlier failed attempts - obey them)\n"
                         + "\n".join(f"{i}. {l}" for i, l in enumerate(memory.lessons, 1)))
        parts.append("# ATTEMPT HISTORY\n| # | failed layer | error |\n|---|---|---|\n"
                     + "\n".join(a.history_row() for a in history))
        parts.append(f"# YOUR LATEST SCRIPT (attempt {last.n})\n```python\n{last.code}\n```")
        parts.append(f"# EVALUATOR FEEDBACK FOR ATTEMPT {last.n}\n{format_failure(last, ctx)}")
        if last.reflection:
            parts.append(f"# YOUR REFLECTION\nDIAGNOSIS: {last.reflection.diagnosis}\nFIX_PLAN: {last.reflection.fix_plan}")
        parts.append("Write the corrected COMPLETE script now. Fix the root cause; do not break what already works.")
    else:
        parts.append("Write the complete script now.")
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "\n\n".join(parts)}]


def reflect_messages(task, attempt: AttemptRecord, history: list[AttemptRecord], memory: Memory,
                     ctx: dict) -> list[dict]:
    parts = [f"# TASK (summary)\n{task.title}\n\n{task.prompt[:1800]}",
             f"# FAILED SCRIPT (attempt {attempt.n})\n```python\n{attempt.code}\n```",
             f"# EVALUATOR FEEDBACK\n{format_failure(attempt, ctx)}"]
    if len(history) > 1:
        parts.append("# EARLIER ATTEMPTS\n" + "\n".join(a.history_row() for a in history[:-1]))
        if memory.lessons:
            parts.append("# LESSONS ALREADY RECORDED\n" + "\n".join(f"- {l}" for l in memory.lessons))
        if len({a.evaluation.signature for a in history[-2:] if a.evaluation}) == 1:
            parts.append("NOTE: the same failure occurred twice in a row - the previous fix did not work. "
                         "Consider a different approach.")
    return [{"role": "system", "content": REFLECTOR_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


_FENCE = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)


def extract_code(text: str) -> str | None:
    """Take the longest fenced block (models sometimes add a short usage example after the script)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    open_fence = re.search(r"```(?:python|py)?[ \t]*\n(.*)", text, re.S)  # truncated response, no closing fence
    if open_fence:
        return open_fence.group(1).strip() + "\n"
    stripped = text.strip()
    if stripped.startswith(("import ", "from ", "#")):
        return stripped + "\n"
    return None


def parse_reflection(text: str) -> Reflection:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def grab(label: str) -> str:
        m = re.search(rf"{label}\s*:\s*(.*?)(?=\n\s*(?:DIAGNOSIS|FIX_PLAN|LESSON)\s*:|\Z)", text, re.S | re.I)
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    diagnosis, plan, lesson = grab("DIAGNOSIS"), grab("FIX_PLAN"), grab("LESSON")
    if not diagnosis:  # model ignored the format: keep the content, degrade gracefully
        diagnosis = re.sub(r"\s+", " ", text)[:500]
    if not lesson:
        lesson = diagnosis.split(". ")[0][:200]
    return Reflection(diagnosis[:800], plan[:600], lesson[:250], text)
