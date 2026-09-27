"""Rich terminal rendering of the loop's events (the live viva view)."""
from __future__ import annotations

from rich.console import Console
from rich.markup import escape as E
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

console = Console(highlight=False)
STATE_STYLE = {"GENERATE": "cyan", "EXECUTE": "magenta", "EVALUATE": "yellow", "REFLECT": "blue", "DONE": "bold"}
STATUS_STYLE = {"OK": "green", "ERROR": "red", "TIMEOUT": "red", "MEMORY_LIMIT": "red", "OUTPUT_LIMIT": "red",
                "BLOCKED": "bold red", "SYNTAX": "red", "SANDBOX_FAILURE": "bold red"}


class ConsoleView:
    def __init__(self, show_code: bool = False) -> None:
        self.show_code = show_code

    def __call__(self, kind: str, d: dict) -> None:
        getattr(self, f"_{kind}", lambda _d: None)(d)

    def provider_event(self, msg: str) -> None:
        console.print(f"  [dim]llm>[/] [yellow]{E(msg)}[/]")

    def _state(self, d):
        s = d["state"]
        if s == "DONE":
            return
        att = f" attempt {d['attempt']}" if "attempt" in d else ""
        if s == "GENERATE":
            console.rule(f"[bold]Attempt {d['attempt']}[/]", style="grey50")
        console.print(f"[{STATE_STYLE[s]}]>> {s:<9}[/]{att}  [dim]t={d['t']}s[/]")

    def _inject(self, d):
        console.print(f"  [bold magenta]FAULT INJECTED[/]: {d['fault']} (first script replaced, LLM not called)")

    def _llm(self, d):
        console.print(f"  [dim]{d['kind']}: {d['provider']}:{d['model']}  {d['tokens']:,} tok  {d['latency']}s  "
                      f"(total {d['total_tokens']:,} tok)[/]")

    def _execution(self, d):
        r = d["run"]
        style = STATUS_STYLE.get(r.status, "white")
        mem = f", peak {r.peak_memory_mb} MB" if r.peak_memory_mb else ""
        console.print(f"  sandbox: [{style}]{r.status}[/] exit={r.exit_code} in {r.duration_s}s{mem}")
        if self.show_code:
            console.print(Syntax((r.workdir / "solution.py").read_text(encoding="utf-8"), "python",
                                 line_numbers=True, theme="ansi_dark"))

    def _evaluation(self, d):
        ev = d["evaluation"]
        if ev.passed:
            console.print(f"  [green]PASS[/] {len(ev.checks)} checks across all layers")
            return
        body = [f"[bold]{E(ev.headline())}[/]"]
        for c in ev.failed_checks[:5]:
            body.append(f"  [red]x[/] {c.layer}/{E(c.name)}: {E(c.detail[:180])}")
        body.append(f"[dim]signature: {E(str(ev.signature))}[/]")
        console.print(Panel("\n".join(body), title=f"failed at {ev.stage}", border_style="red", expand=False))

    def _reflection(self, d):
        r, mem = d["reflection"], d["memory"]
        console.print(Panel(f"[bold]Diagnosis:[/] {E(r.diagnosis)}\n[bold]Fix plan:[/] {E(r.fix_plan or '-')}\n"
                            f"[bold]Lesson ->memory:[/] {E(r.lesson)}", title="reflection", border_style="blue",
                            expand=False))
        console.print(f"  [dim]memory now holds {len(mem.lessons)} lesson(s)[/]")

    def _done(self, d):
        r = d["result"]
        t = Table(title="Attempt history", show_lines=False)
        for col in ("#", "sandbox", "layer", "error / result", "time"):
            t.add_column(col)
        for a in r.attempts:
            status = a.run.status if a.run else "-"
            t.add_row(str(a.n) + (" *" if a.injected else ""), f"[{STATUS_STYLE.get(status, 'white')}]{status}[/]",
                      a.evaluation.stage if a.evaluation else "-",
                      E((a.evaluation.headline() if a.evaluation else "")[:90]),
                      f"{a.run.duration_s}s" if a.run else "-")
        console.print(t)
        colour = "green" if r.success else "red"
        console.print(Panel(f"[bold {colour}]{r.termination.value}[/]: {E(r.reason)}\n"
                            f"attempts {len(r.attempts)} | tokens {r.tokens_used:,} | time {r.elapsed_s}s\n"
                            f"logs: {r.run_dir}", title="terminated", border_style=colour, expand=False))
        if r.success:
            out = r.attempts[-1].run.workdir / "outputs"
            console.print(f"  outputs: {out}")


    def _phase(self, d):
        console.rule(f"[bold magenta]{d['name']}[/]", style="magenta")

    def _ask_done(self, d):
        r = d["result"]
        colour = {"HIGH": "green", "LOW": "yellow", "UNVERIFIED": "cyan", "FAILED": "red"}[r.confidence]
        if r.answer is None:
            console.print(Panel(f"[red]No verified answer.[/] {E(r.primary.reason)}", title="answer", border_style="red"))
            return
        body = [E(r.answer.get("answer", "")), "", f"[dim]method:[/] {E(str(r.answer.get('method', '')))}"]
        res = r.answer.get("result")
        if isinstance(res, list):
            unit = r.answer.get("unit") or ""
            body += [""] + [f"  {i}. {E(str(x.get('label')))}: {x.get('value')} {E(str(unit))}" for i, x in enumerate(res[:10], 1)]
        if r.check is not None:
            verdict = ("agrees" if r.agreement else "DISAGREES") if r.check_answer else "failed to finish"
            body += ["", f"[{colour}]independent check {verdict}[/] {E(r.agreement_detail)}"]
            if r.check_answer and not r.agreement:
                body += [f"[dim]second answer:[/] {E(r.check_answer.get('answer', ''))}"]
        if r.answer.get("chart") and r.outputs_dir:
            body += ["", f"chart: {r.outputs_dir / 'chart.png'}"]
        body += ["", "[dim]Decision support only - a meteorologist should confirm before any public advisory.[/]"]
        console.print(Panel("\n".join(body), title=f"answer - confidence {r.confidence}", border_style=colour))

    def benchmark_summary(self, rows):
        t = Table(title="Benchmark")
        for col in ("task", "result", "attempts", "tokens", "time (s)"):
            t.add_column(col)
        for task, term, n, tok, sec in rows:
            t.add_row(task, f"[{'green' if term == 'SUCCESS' else 'red'}]{term}[/]", str(n), f"{tok:,}", str(sec))
        console.print(t)


def review(code: str, attempt: int) -> tuple[str, str]:
    """Human-in-the-loop gate (--review): approve, reject, or edit the script before it runs."""
    console.print(Panel(Syntax(code, "python", line_numbers=True, theme="ansi_dark"),
                        title=f"REVIEW attempt {attempt}", border_style="yellow"))
    choice = Prompt.ask("run this script?", choices=["y", "n", "e"], default="y")
    if choice == "n":
        return "reject", code
    if choice == "e":
        from pathlib import Path
        import tempfile
        p = Path(tempfile.gettempdir()) / f"reflexion_review_{attempt}.py"
        p.write_text(code, encoding="utf-8")
        Prompt.ask(f"edit [bold]{p}[/] in your editor, save, then press Enter", default="")
        return "run", p.read_text(encoding="utf-8")
    return "run", code
