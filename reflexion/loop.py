"""The Reflexion state machine:  GENERATE -> EXECUTE -> EVALUATE -> (DONE | REFLECT -> GENERATE ...)

                +-----------+   code    +-----------+  result  +-----------+
      start --> | GENERATE  | --------> |  EXECUTE  | -------> | EVALUATE  | --pass--> DONE(SUCCESS)
                +-----------+           +-----------+          +-----------+
                      ^                                              | fail
                      |           lesson -> memory             +-----------+
                      +------------------------------------------ |  REFLECT  |
                                                                  +-----------+
Termination (checked explicitly, reason recorded):
  SUCCESS            all evaluation layers pass
  MAX_ATTEMPTS       attempt budget used (checked BEFORE reflecting, so no wasted LLM call)
  REPEATED_ERROR     same normalised error signature K times in a row -> the model is stuck
  UNSAFE_CODE_LIMIT  N attempts blocked by the sandbox -> the model keeps trying forbidden things
  TIME_BUDGET        total wall-clock budget exhausted (checked on every state transition)
  TOKEN_BUDGET       cumulative prompt+completion tokens exhausted (checked before each LLM call)
  LLM_UNAVAILABLE    all providers failed after retries + fallback
  USER_REJECTED      human reviewer rejected a script (--review mode)
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from . import feedback as fb
from . import prompts as P
from .evaluator import Check, Evaluation, evaluate
from .llm import FallbackLLM, LLMUnavailable
from .sandbox import Sandbox, SandboxLimits
from . import dataset
from .tasks import SANDBOX_DATA_PATH, TaskSpec, data_context


class State(str, Enum):
    GENERATE = "GENERATE"
    EXECUTE = "EXECUTE"
    EVALUATE = "EVALUATE"
    REFLECT = "REFLECT"
    DONE = "DONE"


class Termination(str, Enum):
    SUCCESS = "SUCCESS"
    MAX_ATTEMPTS = "MAX_ATTEMPTS"
    REPEATED_ERROR = "REPEATED_ERROR"
    UNSAFE_CODE_LIMIT = "UNSAFE_CODE_LIMIT"
    TIME_BUDGET = "TIME_BUDGET"
    TOKEN_BUDGET = "TOKEN_BUDGET"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    USER_REJECTED = "USER_REJECTED"


@dataclass
class LoopConfig:
    max_attempts: int = 5
    repeat_limit: int = 3          # identical consecutive error signatures
    unsafe_limit: int = 3          # attempts blocked by the sandbox (static or runtime)
    time_budget_s: float = 600.0
    token_budget: int = 60_000
    temperature: float = 0.2       # actor sampling temperature (verification runs use more)
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    runs_dir: Path = Path(__file__).resolve().parent.parent / "runs"
    keep_data_copies: bool = False


@dataclass
class LoopResult:
    termination: Termination
    reason: str
    attempts: list[P.AttemptRecord]
    memory: P.Memory
    tokens_used: int
    elapsed_s: float
    run_dir: Path
    trace: list[dict]
    config: LoopConfig | None = None        # limits in force (for explaining termination)
    ctx: dict = field(default_factory=dict)  # facts used for hints (for explaining feedback)
    llm_events: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.termination == Termination.SUCCESS

    @property
    def final_code(self) -> str | None:
        return self.attempts[-1].code if self.attempts else None


Reviewer = Callable[[str, int], tuple[str, str]]   # (code, attempt) -> ("run"|"reject", code)


class ReflexionLoop:
    def __init__(self, task: TaskSpec, llm: FallbackLLM, config: LoopConfig | None = None, *,
                 injected_first_code: str | None = None, injected_fault: str | None = None,
                 reviewer: Reviewer | None = None, on_event: Callable[[str, dict], None] | None = None) -> None:
        self.task, self.llm, self.cfg = task, llm, config or LoopConfig()
        self.injected_code, self.injected_fault = injected_first_code, injected_fault
        self.reviewer = reviewer
        self.emit = on_event or (lambda kind, data: None)
        self.sandbox = Sandbox(self.cfg.limits, {SANDBOX_DATA_PATH: dataset.active().path})
        self.ctx = {"columns": list(dataset.read().columns), "data_path": SANDBOX_DATA_PATH,
                    "timeout": self.cfg.limits.timeout_s, "memory": self.cfg.limits.memory_mb}
        self.data_ctx = data_context()
        self.sys_prompt = P.actor_system(self.cfg.limits.timeout_s, self.cfg.limits.memory_mb)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.cfg.runs_dir / f"{stamp}_{task.id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace: list[dict] = []
        self.tokens = 0
        self.t0 = 0.0

    # ------------------------------------------------------------------ helpers
    def _transition(self, state: State, **info) -> None:
        rec = {"t": round(time.monotonic() - self.t0, 2), "state": state.value, **info}
        self.trace.append(rec)
        self.emit("state", rec)

    def _budget_exceeded(self) -> tuple[Termination, str] | None:
        elapsed = time.monotonic() - self.t0
        if elapsed >= self.cfg.time_budget_s:
            return Termination.TIME_BUDGET, f"wall-clock budget of {self.cfg.time_budget_s:.0f}s exhausted ({elapsed:.0f}s)"
        if self.tokens >= self.cfg.token_budget:
            return Termination.TOKEN_BUDGET, f"token budget of {self.cfg.token_budget:,} exhausted ({self.tokens:,})"
        return None

    def _call_llm(self, messages: list[dict], log_name: str, attempt_dir: Path, **kw):
        resp = self.llm.chat(messages, **kw)
        self.tokens += resp.total_tokens
        (attempt_dir / f"{log_name}_prompt.md").write_text(
            "\n\n".join(f"<!-- {m['role']} -->\n{m['content']}" for m in messages), encoding="utf-8")
        (attempt_dir / f"{log_name}_response.md").write_text(resp.text, encoding="utf-8")
        self.emit("llm", {"kind": log_name, "provider": resp.provider, "model": resp.model,
                          "tokens": resp.total_tokens, "latency": round(resp.latency_s, 1), "total_tokens": self.tokens})
        return resp

    def _is_unsafe(self, a: P.AttemptRecord) -> bool:
        ev = a.evaluation
        return bool(ev and ((ev.error and ev.error.type in {"UnsafeCode", "SandboxViolation"})
                            or (a.run and a.run.violations)))

    def _repeat_streak(self, attempts: list[P.AttemptRecord]) -> int:
        sig = attempts[-1].evaluation.signature
        streak = 0
        for a in reversed(attempts):
            if a.evaluation and a.evaluation.signature == sig:
                streak += 1
            else:
                break
        return streak

    def _after_failure(self, attempts: list[P.AttemptRecord]) -> tuple[Termination, str] | None:
        """Termination checks after a failed attempt, in priority order."""
        unsafe = sum(self._is_unsafe(a) for a in attempts)
        streak = self._repeat_streak(attempts)
        if unsafe >= self.cfg.unsafe_limit:
            return Termination.UNSAFE_CODE_LIMIT, f"{unsafe} attempts were blocked by the sandbox"
        if streak >= self.cfg.repeat_limit:
            return (Termination.REPEATED_ERROR,
                    f"same failure signature {streak}x in a row: {attempts[-1].evaluation.signature}")
        if len(attempts) >= self.cfg.max_attempts:  # stop BEFORE reflecting: no wasted LLM call
            return Termination.MAX_ATTEMPTS, f"{self.cfg.max_attempts} attempts used without passing"
        return None

    # ------------------------------------------------------------------ main loop
    def run(self) -> LoopResult:
        self.t0 = time.monotonic()
        attempts: list[P.AttemptRecord] = []
        memory = P.Memory()
        state, code, current, attempt_dir = State.GENERATE, None, None, self.run_dir
        term: tuple[Termination, str] | None = None

        while term is None:
            if (b := self._budget_exceeded()) is not None:
                term = b
                break

            if state is State.GENERATE:
                if len(attempts) >= self.cfg.max_attempts:
                    term = (Termination.MAX_ATTEMPTS, f"{self.cfg.max_attempts} attempts used without passing")
                    break
                n = len(attempts) + 1
                attempt_dir = self.run_dir / f"attempt_{n}"
                attempt_dir.mkdir(exist_ok=True)
                self._transition(State.GENERATE, attempt=n)
                if n == 1 and self.injected_code is not None:
                    code = self.injected_code
                    (attempt_dir / "INJECTED_FAULT.txt").write_text(self.injected_fault or "", encoding="utf-8")
                    self.emit("inject", {"fault": self.injected_fault})
                else:
                    msgs = P.generate_messages(self.task, self.data_ctx, self.sys_prompt, attempts, memory, self.ctx)
                    try:
                        resp = self._call_llm(msgs, "generate", attempt_dir, temperature=self.cfg.temperature)
                    except LLMUnavailable as e:
                        term = (Termination.LLM_UNAVAILABLE, str(e))
                        break
                    code = P.extract_code(resp.text)
                current = P.AttemptRecord(n, code or "", None, None, injected=self.injected_fault if n == 1 else None)
                attempts.append(current)
                if code is None:  # malformed response: skip execution, feed the format error back
                    err = fb.ErrorInfo("FormatError", "the response contained no ```python fenced code block")
                    current.evaluation = Evaluation(False, "FORMAT", [Check("code_block", "FORMAT", False, err.message)],
                                                    err, fb.signature("FormatError", ""))
                    self.emit("evaluation", {"attempt": n, "evaluation": current.evaluation})
                    term = self._after_failure(attempts)
                    state = State.REFLECT
                    continue
                (attempt_dir / "solution.py").write_text(code, encoding="utf-8")
                if self.reviewer is not None:
                    decision, code = self.reviewer(code, n)
                    current.code = code
                    if decision == "reject":
                        term = (Termination.USER_REJECTED, f"reviewer rejected the script of attempt {n}")
                        break
                state = State.EXECUTE

            elif state is State.EXECUTE:
                self._transition(State.EXECUTE, attempt=current.n)
                current.run = self.sandbox.run(current.code, attempt_dir / "sandbox")
                if not self.cfg.keep_data_copies:
                    shutil.rmtree(attempt_dir / "sandbox" / "data", ignore_errors=True)
                self.emit("execution", {"attempt": current.n, "run": current.run})
                state = State.EVALUATE

            elif state is State.EVALUATE:
                self._transition(State.EVALUATE, attempt=current.n)
                ev = evaluate(self.task, current.run)
                current.evaluation = ev
                self.emit("evaluation", {"attempt": current.n, "evaluation": ev})
                if ev.passed:
                    term = (Termination.SUCCESS, f"all evaluation layers passed on attempt {current.n}")
                    break
                term = self._after_failure(attempts)
                state = State.REFLECT

            elif state is State.REFLECT:
                self._transition(State.REFLECT, attempt=current.n)
                msgs = P.reflect_messages(self.task, current, attempts, memory, self.ctx)
                try:
                    resp = self._call_llm(msgs, "reflect", attempt_dir, max_tokens=2048)
                except LLMUnavailable as e:
                    term = (Termination.LLM_UNAVAILABLE, str(e))
                    break
                current.reflection = P.parse_reflection(resp.text)
                memory.add(current.reflection.lesson)
                self.emit("reflection", {"attempt": current.n, "reflection": current.reflection, "memory": memory})
                state = State.GENERATE

        termination, reason = term
        self._transition(State.DONE, termination=termination.value, reason=reason)
        result = LoopResult(termination, reason, attempts, memory, self.tokens,
                            round(time.monotonic() - self.t0, 1), self.run_dir, self.trace,
                            self.cfg, self.ctx, list(self.llm.events))
        self._save(result)
        self.emit("done", {"result": result})
        return result

    # ------------------------------------------------------------------ persistence
    def _save(self, r: LoopResult) -> None:
        cur = self.llm.current
        summary = {
            "task": self.task.id, "termination": r.termination.value, "reason": r.reason,
            "attempts": len(r.attempts), "tokens_used": r.tokens_used, "elapsed_s": r.elapsed_s,
            "llm": f"{cur.name}:{cur.model}" if cur else None, "llm_events": self.llm.events,
            "config": {"max_attempts": self.cfg.max_attempts, "repeat_limit": self.cfg.repeat_limit,
                       "unsafe_limit": self.cfg.unsafe_limit, "time_budget_s": self.cfg.time_budget_s,
                       "token_budget": self.cfg.token_budget, "timeout_s": self.cfg.limits.timeout_s,
                       "memory_mb": self.cfg.limits.memory_mb},
            "memory": r.memory.lessons,
            "history": [{
                "attempt": a.n, "injected_fault": a.injected,
                "status": a.run.status if a.run else None,
                "duration_s": a.run.duration_s if a.run else None,
                "peak_memory_mb": a.run.peak_memory_mb if a.run else None,
                "stage": a.evaluation.stage if a.evaluation else None,
                "signature": a.evaluation.signature if a.evaluation else None,
                "headline": a.evaluation.headline() if a.evaluation else None,
                "failed_checks": [str(c) for c in a.evaluation.failed_checks] if a.evaluation else [],
                "reflection": a.reflection.__dict__ if a.reflection else None,
            } for a in r.attempts],
            "trace": r.trace,
        }
        (self.run_dir / "trace.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (self.run_dir / "report.md").write_text(_report_md(summary), encoding="utf-8")
        if r.success and r.final_code:
            (self.run_dir / "final_solution.py").write_text(r.final_code, encoding="utf-8")


def _report_md(s: dict) -> str:
    lines = [f"# Reflexion run - {s['task']}", "",
             f"**Termination:** `{s['termination']}` - {s['reason']}  ",
             f"**Attempts:** {s['attempts']} | **Tokens:** {s['tokens_used']:,} | **Time:** {s['elapsed_s']}s | "
             f"**LLM:** {s['llm']}", "", "| # | status | layer | error / result | reflection lesson |",
             "|---|---|---|---|---|"]
    for h in s["history"]:
        lesson = (h["reflection"] or {}).get("lesson", "")
        tag = f" (injected: {h['injected_fault']})" if h["injected_fault"] else ""
        lines.append(f"| {h['attempt']}{tag} | {h['status']} | {h['stage']} | "
                     f"{(h['headline'] or '').replace('|', '/')[:140]} | {lesson.replace('|', '/')[:120]} |")
    if s["memory"]:
        lines += ["", "## Reflexion memory at termination", *[f"{i}. {l}" for i, l in enumerate(s["memory"], 1)]]
    if s["llm_events"]:
        lines += ["", "## Provider events", *[f"- {e}" for e in s["llm_events"]]]
    lines += ["", "## State trace", "", "| t (s) | state | detail |", "|---|---|---|"]
    for t in s["trace"]:
        detail = ", ".join(f"{k}={v}" for k, v in t.items() if k not in {"t", "state"})
        lines.append(f"| {t['t']} | {t['state']} | {detail} |")
    return "\n".join(lines) + "\n"
