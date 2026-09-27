"""Streamlit helpers for HeatDesk: the live step timeline and the "code & its fixes" view."""
from __future__ import annotations


import streamlit as st

from reflexion import dataset
from reflexion.evaluator import LAYERS
from reflexion.llm import build_llm
from reflexion.loop import LoopResult, Termination
from reflexion.prompts import format_failure

STATE_INFO = {
    "GENERATE": (":material/edit_note:", "Generate", "the AI writes a complete Python script"),
    "EXECUTE": (":material/play_circle:", "Execute", "the script runs inside the locked sandbox"),
    "EVALUATE": (":material/fact_check:", "Evaluate", "the result is checked layer by layer"),
    "REFLECT": (":material/psychology:", "Reflect", "the AI studies the error and writes a lesson"),
}
TERMINATION_INFO = {
    Termination.SUCCESS: ("Success", "every check passed", "green"),
    Termination.MAX_ATTEMPTS: ("Out of attempts", "the attempt limit was reached before all checks passed", "red"),
    Termination.REPEATED_ERROR: ("Stuck on the same error", "the same mistake came back several times in a row, "
                                 "so more tries would waste time", "orange"),
    Termination.UNSAFE_CODE_LIMIT: ("Too many unsafe attempts", "the AI kept trying forbidden operations", "red"),
    Termination.TIME_BUDGET: ("Time budget used up", "the total time limit for this question ran out", "orange"),
    Termination.TOKEN_BUDGET: ("Token budget used up", "the AI usage (cost) limit ran out", "orange"),
    Termination.LLM_UNAVAILABLE: ("AI unavailable", "no AI model could be reached, even after retries and fallback", "red"),
    Termination.USER_REJECTED: ("Rejected by reviewer", "a human declined to run the script", "gray"),
}


@st.cache_resource(show_spinner=False)
def get_llm(provider: str):
    return build_llm(provider)


# ============================================================================ live timeline
class LiveTimeline:
    """on_event callback: renders each loop state as a step in a vertical timeline, live."""

    def __init__(self) -> None:
        self.step = None
        self.failed = False

    def _close(self) -> None:
        if self.step is not None:
            self.step.update(state="error" if self.failed else "complete")
        self.step, self.failed = None, False

    def provider_event(self, msg: str) -> None:
        (self.step or st).caption(f":material/hourglass_top: {msg[:170]}")

    def __call__(self, kind: str, d: dict) -> None:
        if kind == "phase":
            self._close()
            st.markdown(f"**{d['name'].capitalize()}**")
        elif kind == "state":
            self._close()
            s = d["state"]
            if s == "DONE":
                name, why, _ = TERMINATION_INFO[Termination(d["termination"])]
                icon = ":material/flag:" if d["termination"] == "SUCCESS" else ":material/block:"
                st.markdown(f"{icon} **Stopped: {name}** - {why}. `{d['termination']}` after {d['t']} s")
                return
            icon, title, what = STATE_INFO[s]
            self.step = st.status(f"{icon} {title} · attempt {d['attempt']} - {what}", type="step", expanded=False)
        elif self.step is None:
            return
        elif kind == "inject":
            self.step.markdown(f":orange[:material/bug_report: Fault injected on purpose:] `{d['fault']}` - "
                               "this first script is deliberately broken; the AI is not called.")
        elif kind == "llm":
            self.step.caption(f"{d['provider']}:{d['model']} · {d['tokens']:,} tokens · {d['latency']} s")
        elif kind == "execution":
            r = d["run"]
            mem = f" · peak memory {r.peak_memory_mb} MB" if r.peak_memory_mb else ""
            self.step.markdown(f"Sandbox result: **{r.status}** · exit code {r.exit_code} · {r.duration_s} s{mem}")
            if r.violations:
                self.step.markdown(f":red[Blocked operations:] {', '.join(v['event'] for v in r.violations)}")
            self.failed = not r.ok
        elif kind == "evaluation":
            ev = d["evaluation"]
            self.step.markdown(_layer_line(ev))
            if not ev.passed:
                self.step.markdown(f":red[Problem:] {ev.headline()[:220]}")
            self.failed = not ev.passed
        elif kind == "reflection":
            r = d["reflection"]
            self.step.markdown(f"**Diagnosis:** {r.diagnosis}\n\n**Lesson saved to memory:** *{r.lesson}*")


def _layer_line(ev) -> str:
    order = (["FORMAT"] if ev.stage == "FORMAT" else []) + LAYERS
    parts, reached = [], True
    for layer in order:
        if not reached:
            parts.append(f":gray[{layer.lower()} –]")
        elif ev.stage == layer:
            parts.append(f":red[{layer.lower()} ✗]")
            reached = False
        else:
            parts.append(f":green[{layer.lower()} ✓]")
    return " → ".join(parts)


# ============================================================================ behind the scenes
# ---------------------------------------------------------------------------- 0. the code
PLAIN_ERRORS = {
    "KeyError": "it asked for a column or item that doesn't exist",
    "NameError": "it used a name it never defined (for example, it forgot an import)",
    "TypeError": "it gave a function the wrong kind of input",
    "ValueError": "it gave a function a value it can't use",
    "AttributeError": "it used a feature that doesn't exist on that object",
    "IndexError": "it used a position that doesn't exist (for example, two lists of different lengths)",
    "ZeroDivisionError": "it divided by zero",
    "FileNotFoundError": "it looked for a file in the wrong place",
    "ModuleNotFoundError": "it used a library that isn't allowed here",
    "ImportError": "it used a library feature that isn't available here",
    "SyntaxError": "it has a typing mistake, so Python couldn't even read it",
    "IndentationError": "its lines are wrongly indented, so Python couldn't read it",
    "MemoryError": "it tried to use more memory than allowed",
    "MEMORY_LIMIT": "it tried to use more memory than allowed",
    "TIMEOUT": "it ran too long (probably a loop that never ends) and was stopped",
    "OUTPUT_LIMIT": "it printed far too much text and was stopped",
    "SandboxViolation": "it tried something forbidden (internet, other programs, or files outside its folder) and was blocked",
    "UnsafeCode": "it contained forbidden commands, so it was never run",
    "FormatError": "the AI's reply contained no code",
}
PLAIN_STAGES = {
    "ARTIFACT": "it ran, but didn't save the required result file",
    "SCHEMA": "it ran, but saved the result in the wrong format",
    "SEMANTIC": "it ran, but the numbers failed a common-sense or correctness check",
}


def plain_reason(ev) -> str:
    if ev.error is not None:
        return PLAIN_ERRORS.get(ev.error.type, f"Python reported a {ev.error.type}")
    return PLAIN_STAGES.get(ev.stage, "a check failed")


def code_steps(code: str) -> list[str]:
    """The AI is asked to mark its stages as '# Step N: ...' comments; list them in plain English."""
    import re
    return [m.group(2).strip() for m in re.finditer(r"(?im)^\s*#\s*step\s*(\d+)\s*[:.)-]\s*(.+)$", code)]


def broken_line_view(code: str, line: int, radius: int = 2) -> str:
    """Rendered with language='diff' so the failing line (prefixed '-') is shown in red."""
    lines = code.splitlines()
    lo, hi = max(1, line - radius), min(len(lines), line + radius)
    return "\n".join(f"{'-' if n == line else ' '} {n:>3} | {lines[n - 1]}" for n in range(lo, hi + 1))


def code_diff(old: str, new: str) -> tuple[str, int, int]:
    import difflib
    import re
    out, added, removed = [], 0, 0
    for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1):
        if l.startswith(("---", "+++")):
            continue
        if l.startswith("@@"):
            m = re.search(r"\+(\d+)", l)
            out.append(f"··· around line {m.group(1) if m else '?'} ···")
            continue
        added += l.startswith("+")
        removed += l.startswith("-")
        out.append(l)
    return "\n".join(out), added, removed


BLOCK_WORDS = {"subprocess": "start another program", "os.system": "run a shell command", "socket": "use the internet",
               "urllib": "use the internet", "open": "write a file outside its folder", "os.remove": "delete a file"}


def safety_line(run, cfg) -> str:
    """One plain sentence about what the sandbox did for this run."""
    if run is None:
        return ":material/shield: **Safety:** not run."
    if run.guard is not None and run.guard.violations and run.exit_code is None:
        return ":material/shield: **Safety:** :red[blocked before running] - the code contained forbidden commands."
    if run.violations:
        what = {next((w for k, w in BLOCK_WORDS.items() if v["event"].startswith(k)), v["event"]) for v in run.violations}
        return f":material/shield: **Safety:** :red[blocked] - it tried to {', '.join(sorted(what))}."
    limits = {"TIMEOUT": "stopped - it ran past the time limit", "MEMORY_LIMIT": "stopped - it hit the memory limit",
              "OUTPUT_LIMIT": "stopped - it printed too much"}
    if run.status in limits:
        return f":material/shield: **Safety:** :orange[{limits[run.status]}]."
    lim = cfg.limits if cfg else None
    mem = f"{run.peak_memory_mb:.0f} MB" + (f" of {lim.memory_mb} MB" if lim else "") if run.peak_memory_mb else "-"
    return (f":material/shield: **Safety:** ran in the locked sandbox (no internet, no access outside its folder, "
            f"no passwords) - {run.duration_s} s" + (f" of {lim.timeout_s:.0f} s" if lim else "") + f", memory {mem}.")


def render_code_tab(result: LoopResult, key: str) -> None:
    st.caption("To answer, the AI **writes a small Python program**. The program is run; if it fails, the AI "
               "reads the error, **corrects the program** and runs it again. Every version is shown below: "
               ":red[red lines] were removed or broke, :green[green lines] were added by the fix.")
    atts = [a for a in result.attempts if a.code]
    if not atts:
        st.info("No code was produced.")
        return
    # --- summary: the loop at a glance (state loop + stopping rule + sandbox limits)
    name, why, colour = TERMINATION_INFO[result.termination]
    cfg = getattr(result, "config", None)
    runs = [a.run for a in atts if a.run is not None]
    peak = max((r.peak_memory_mb or 0 for r in runs), default=0)
    with st.container(horizontal=True):
        st.metric("Tries", f"{len(atts)}" + (f" of {cfg.max_attempts}" if cfg else ""), border=True)
        st.metric("Fixed by itself", f"{sum(1 for a in atts if a.evaluation and not a.evaluation.passed)} time(s)",
                  border=True)
        st.metric("Time", f"{result.elapsed_s:.0f} s", border=True)
        st.metric("Most memory used", f"{peak:.0f} MB" + (f" of {cfg.limits.memory_mb}" if cfg else ""), border=True)
    st.markdown(f"**Stopped because:** :{colour}-badge[{name}] {why}. "
                f"After every failed try the program checks its stopping rules (max tries, same error repeated, "
                f"unsafe code, time and cost limits) before trying again.")
    prev = None
    for a in atts:
        ev = a.evaluation
        ok = bool(ev and ev.passed)
        with st.container(border=True):
            badge = ":green-badge[:material/check: worked]" if ok else ":red-badge[:material/close: failed]"
            planted = " :orange-badge[:material/bug_report: bug planted on purpose]" if a.injected else ""
            st.markdown(f"#### Version {a.n} {badge}{planted}")
            steps = code_steps(a.code)
            if steps:
                st.markdown("**What this program does:**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
            if prev is not None:
                diff, added, removed = code_diff(prev.code, a.code)
                counts = ", ".join(x for x in (f":red[{removed} line(s) removed]" if removed else "",
                                               f":green[{added} line(s) added]" if added else "") if x)
                st.markdown(f":material/edit: **What the AI changed after reading the error** - {counts or 'nothing'}")
                st.code(diff or "(no changes - the AI resubmitted the same program)", language="diff")
            with st.expander(f"Full program, version {a.n} ({len(a.code.splitlines())} lines)",
                             icon=":material/description:", expanded=ok and prev is None):
                st.code(a.code, language="python", line_numbers=True)
            st.markdown(safety_line(a.run, cfg))
            if ev is not None:
                st.markdown("**Checks:** " + _layer_line(ev))
            if not ok and ev is not None:
                st.markdown(f":material/error: **Why it failed:** {plain_reason(ev)}.")
                if ev.error is not None and ev.error.line:
                    st.markdown(f"**The line that broke** (line {ev.error.line}, in red):")
                    st.code(broken_line_view(a.code, ev.error.line), language="diff")
                st.caption(f"Technical detail: {ev.headline()[:300]}")
                ctx = getattr(result, "ctx", None)
                if ctx:
                    with st.expander("What the AI was told (the short error report sent back to it)",
                                     icon=":material/forum:"):
                        st.caption("Not the raw error dump: only the final error, the failing line with 2 lines "
                                   "around it, and a hint with facts the AI can't see.")
                        st.code(format_failure(a, ctx), language="text")
                if a.reflection:
                    st.markdown(f":material/psychology: **The AI's plan to fix it:** "
                                f"{a.reflection.fix_plan or a.reflection.diagnosis}  \n"
                                f"**Lesson it will remember:** *{a.reflection.lesson}*")
        prev = a
    final = atts[-1]
    if final.evaluation and final.evaluation.passed:
        st.success(f"Version {final.n} passed every check - this is the program that produced the answer.",
                   icon=":material/verified:")
        st.download_button("Download the final program", final.code, file_name="heatdesk_solution.py",
                           mime="text/x-python", icon=":material/download:", key=f"dl_{key}")


def active_dataset():
    return dataset.use(st.session_state.get("data_name") or dataset.active().name)
