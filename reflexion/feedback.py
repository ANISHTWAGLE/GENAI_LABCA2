"""Turning raw failures into compact, structured feedback for the LLM.

Design rules (why the feedback looks the way it does):
  * Signal over noise - library-internal frames and boot/runpy frames are dropped; the
    model sees the failing line of ITS code with +/-2 lines of context, the exception
    type/message, and at most the last library frame (e.g. pandas get_loc) as a clue.
  * Bounded size - stderr/stdout are trimmed from the middle so the head (what started)
    and tail (what finally broke) survive; prompt size stays roughly constant per attempt.
  * Normalised signatures - volatile parts (numbers, addresses, paths) are masked so the
    loop can recognise "the same error again" even when line numbers shift.
  * Targeted hints - an error-type -> hint table injects facts the model cannot see
    (real column names, allowed modules, limits).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .guard import ALLOWED_MODULES

_TB_HEADER = "Traceback (most recent call last):"
_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>.+))?$')
_EXC_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Violation|Warning|Iteration)?)"
                     r"(?::\s?(?P<msg>.*))?$")


@dataclass
class ErrorInfo:
    type: str
    message: str
    line: int | None = None           # line in solution.py
    func: str | None = None
    library_frame: str | None = None  # e.g. "base.py:3812 in get_loc"
    cause: str | None = None

    def headline(self) -> str:
        where = f" (solution.py line {self.line})" if self.line else ""
        return f"{self.type}: {self.message}{where}"


def parse_traceback(stderr: str, solution_name: str = "solution.py") -> ErrorInfo | None:
    """Regex fallback parser for raw stderr (used when no structured _error.json exists)."""
    if _TB_HEADER not in stderr:
        return None
    block = stderr[stderr.rfind(_TB_HEADER):]          # the LAST traceback = the final exception
    lines = block.splitlines()[1:]
    user_line = user_func = lib = None
    exc_type, exc_msg = "UnknownError", ""
    for i, raw in enumerate(lines):
        m = _FRAME_RE.match(raw)
        if m:
            fname = m.group("file").replace("\\", "/").rsplit("/", 1)[-1]
            if fname == solution_name:
                user_line, user_func, lib = int(m.group("line")), m.group("func"), None
            else:
                lib = f"{fname}:{m.group('line')} in {m.group('func')}"
            continue
        if raw and not raw.startswith((" ", "\t")):
            em = _EXC_RE.match(raw.strip())
            if em:
                exc_type = em.group("type").rsplit(".", 1)[-1]
                exc_msg = "\n".join([em.group("msg") or ""] + [l for l in lines[i + 1:] if l.strip()][:3]).strip()
                break
    cause = None
    if "The above exception was the direct cause" in stderr or "During handling of the above" in stderr:
        first = stderr[stderr.find(_TB_HEADER):]
        for raw in first.splitlines():
            em = _EXC_RE.match(raw.strip()) if raw and not raw.startswith((" ", "\t", "Traceback")) else None
            if em and em.group("msg") is not None:
                cause = raw.strip()[:300]
                break
    return ErrorInfo(exc_type, exc_msg[:1500], user_line, user_func, lib, cause)


def from_record(record: dict) -> ErrorInfo:
    """Build ErrorInfo from the structured record written by sandbox_boot.py."""
    frames = record.get("user_frames") or []
    last = frames[-1] if frames else {}
    lib = record.get("last_library_frame")
    return ErrorInfo(record.get("type", "UnknownError"), record.get("message", ""),
                     last.get("line"), last.get("func"),
                     f"{lib['file']}:{lib['line']} in {lib['func']}" if lib else None, record.get("cause"))


def normalise(msg: str) -> str:
    msg = re.sub(r"0x[0-9a-fA-F]+", "0x?", msg)
    msg = re.sub(r"[A-Za-z]:\\[^\s'\"]+|/[\w./-]+", "<path>", msg)
    msg = re.sub(r"\d+(\.\d+)?", "N", msg)
    return re.sub(r"\s+", " ", msg).strip()[:200]


def signature(kind: str, detail: str) -> str:
    """Stable identity of a failure, robust to shifted line numbers and changing values."""
    norm = normalise(detail)
    return f"{kind}:{norm[:60]}#{hashlib.sha1(norm.encode()).hexdigest()[:6]}"


def code_excerpt(code: str, line: int | None, radius: int = 2) -> str:
    if not line:
        return ""
    lines = code.splitlines()
    lo, hi = max(1, line - radius), min(len(lines), line + radius)
    width = len(str(hi))
    return "\n".join(f"{'>>' if n == line else '  '} {n:>{width}} | {lines[n - 1]}" for n in range(lo, hi + 1))


def trim_middle(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head, tail = limit // 3, limit - limit // 3
    return f"{text[:head]}\n... [{len(text) - limit} chars trimmed] ...\n{text[-tail:]}"


def hint_for(error_type: str, message: str, ctx: dict) -> str | None:
    """Error-type-aware hints: facts the model needs but cannot see from the traceback alone."""
    t = error_type
    if t == "KeyError" or "not in index" in message or "column" in message.lower() and "not" in message.lower():
        return f"Column names are case-sensitive. The dataset columns are exactly: {ctx.get('columns')}."
    if t in {"ModuleNotFoundError", "ImportError"}:
        return f"Only these modules are available: {', '.join(sorted(ALLOWED_MODULES - {'__future__'}))}."
    if t == "FileNotFoundError":
        return f"Input data is at '{ctx.get('data_path')}'. Write outputs under 'outputs/' (os.makedirs first)."
    if t == "SandboxViolation" or t == "BLOCKED":
        return "The sandbox forbids this operation (network, subprocess, shell, or writing outside the " \
               "sandbox). There is no workaround - remove it and solve the task with local data only."
    if t == "TIMEOUT":
        return f"The script exceeded the {ctx.get('timeout')}s limit. Look for loops whose condition never " \
               "changes, and replace row-by-row loops (iterrows/iloc/apply) with vectorised pandas/numpy."
    if t in {"MemoryError", "MEMORY_LIMIT"}:
        return f"The script exceeded the {ctx.get('memory')} MB memory limit. Do not build huge lists or " \
               "cross-joins; use groupby/aggregations and process only the columns you need."
    if t == "OUTPUT_LIMIT":
        return "The script printed far too much. Print only short summaries, never whole DataFrames in loops."
    if t == "SyntaxError" or t == "IndentationError":
        return "Return the COMPLETE script in a single ```python block; check brackets and indentation."
    if t == "JSONDecodeError" or "not JSON serializable" in message:
        return "Convert numpy types to built-in Python types (int(), float(), str()) before json.dump."
    return None
