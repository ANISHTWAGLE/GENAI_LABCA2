"""EVALUATE state: layered verdict on one execution.

Layers are checked in order and evaluation stops at the first failing layer, because
later layers are meaningless if earlier ones fail (no point checking values in a JSON
file that was never written):

  STATIC   - syntax + safety guard passed
  RUNTIME  - process exited 0, inside time/memory/output limits, no runtime violations
  ARTIFACT - required output files exist and are valid (JSON parses, PNG magic bytes)
  SCHEMA   - result.json has the required keys and types
  SEMANTIC - values agree with the trusted oracle and domain constraints
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import feedback as fb
from .sandbox import ExecutionResult

LAYERS = ["STATIC", "RUNTIME", "ARTIFACT", "SCHEMA", "SEMANTIC"]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass
class Check:
    name: str
    layer: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.layer}/{self.name}: {self.detail}"


@dataclass
class Evaluation:
    passed: bool
    stage: str                      # first failing layer, or "PASSED"
    checks: list[Check] = field(default_factory=list)
    error: fb.ErrorInfo | None = None
    signature: str | None = None
    result: dict | None = None      # parsed result.json (if any)

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def headline(self) -> str:
        if self.passed:
            return "all checks passed"
        if self.error:
            return f"{self.stage}: {self.error.headline()}"
        return f"{self.stage}: " + "; ".join(f"{c.name} - {c.detail}" for c in self.failed_checks[:2])


def evaluate(task, run: ExecutionResult) -> Evaluation:
    checks: list[Check] = []

    # ---- STATIC --------------------------------------------------------------------------
    if run.status == "SYNTAX":
        se = run.error or {}
        err = fb.ErrorInfo("SyntaxError", se.get("message", ""), se.get("line"))
        return Evaluation(False, "STATIC", [Check("syntax", "STATIC", False, err.headline())], err,
                          fb.signature("SyntaxError", se.get("message", "")))
    if run.status == "BLOCKED" and run.guard and not run.guard.ok:
        vs = run.guard.violations
        checks = [Check(f"guard:{v.rule}", "STATIC", False, str(v)) for v in vs]
        err = fb.ErrorInfo("UnsafeCode", "; ".join(str(v) for v in vs[:4]), vs[0].line if vs else None)
        return Evaluation(False, "STATIC", checks, err,
                          fb.signature("UnsafeCode", ",".join(sorted({v.rule + v.detail for v in vs}))))
    checks.append(Check("syntax+guard", "STATIC", True, "no forbidden constructs"))

    # ---- RUNTIME -------------------------------------------------------------------------
    if not run.ok:
        err = _runtime_error(run)
        checks.append(Check(run.status.lower(), "RUNTIME", False, err.headline()))
        return Evaluation(False, "RUNTIME", checks, err, fb.signature(err.type, err.message))
    checks.append(Check("exit_code", "RUNTIME", True, f"exit 0 in {run.duration_s}s, peak {run.peak_memory_mb} MB"))

    # ---- ARTIFACT ------------------------------------------------------------------------
    result = None
    for rel in task.required_outputs:
        path = run.workdir / rel
        if not path.exists() or path.stat().st_size == 0:
            checks.append(Check(f"exists:{rel}", "ARTIFACT", False, f"'{rel}' was not created (or is empty)"))
            continue
        if rel.endswith(".png"):
            ok = path.read_bytes()[:8] == PNG_MAGIC and path.stat().st_size > 1000
            checks.append(Check(f"png:{rel}", "ARTIFACT", ok, "valid PNG" if ok else "not a valid, non-trivial PNG"))
        elif rel.endswith(".json"):
            try:
                if path.stat().st_size > 2_000_000:
                    raise ValueError("file larger than 2 MB")
                result = json.loads(path.read_text(encoding="utf-8"))
                checks.append(Check(f"json:{rel}", "ARTIFACT", True, "parses"))
            except (ValueError, UnicodeDecodeError) as e:
                checks.append(Check(f"json:{rel}", "ARTIFACT", False, f"invalid JSON: {str(e)[:150]}"))
        else:
            checks.append(Check(f"exists:{rel}", "ARTIFACT", True, f"{path.stat().st_size} bytes"))
    if failed := [c for c in checks if not c.passed]:
        return Evaluation(False, "ARTIFACT", checks, None, fb.signature("Artifact", ",".join(c.name for c in failed)))

    # ---- SCHEMA + SEMANTIC (task-specific) -------------------------------------------------
    for layer, validator in (("SCHEMA", task.schema_check), ("SEMANTIC", task.semantic_check)):
        if validator is None:
            continue
        try:
            new = validator(result, run.workdir)
        except Exception as e:  # a validator crash on malformed output = that layer fails
            new = [Check("validator", layer, False, f"output could not be validated: {type(e).__name__}: {e}")]
        checks.extend(new)
        if failed := [c for c in new if not c.passed]:
            return Evaluation(False, layer, checks, None,
                              fb.signature(layer, ",".join(sorted(c.name for c in failed))), result)
    return Evaluation(True, "PASSED", checks, None, None, result)


def _runtime_error(run: ExecutionResult) -> fb.ErrorInfo:
    if run.status == "TIMEOUT":
        return fb.ErrorInfo("TIMEOUT", f"killed after exceeding the wall-clock limit ({run.duration_s}s)")
    if run.status == "OUTPUT_LIMIT":
        return fb.ErrorInfo("OUTPUT_LIMIT", "killed: stdout/stderr exceeded the size cap")
    if run.status == "BLOCKED" and run.violations:
        base = fb.from_record(run.error) if run.error else fb.ErrorInfo("SandboxViolation", "")
        v = run.violations[0]
        return fb.ErrorInfo("SandboxViolation", f"{v['event']} blocked: {v['detail']}", base.line, base.func)
    if run.error:
        info = fb.from_record(run.error)
    else:
        info = fb.parse_traceback(run.stderr) or fb.ErrorInfo(
            "ProcessError", f"exit code {run.exit_code}; stderr: {fb.trim_middle(run.stderr, 300) or '(empty)'}")
    if run.status == "MEMORY_LIMIT" and info.type != "MemoryError":
        info = fb.ErrorInfo("MEMORY_LIMIT", f"process exceeded the memory cap (peak {run.peak_memory_mb} MB)",
                            info.line, info.func)
    return info
