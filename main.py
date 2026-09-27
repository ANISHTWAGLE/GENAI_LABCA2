"""Reflexion code-generation loop for IMD heatwave intelligence (use case KJS-CES-01).

Examples:
  python main.py --ask "Which 5 places had the most heatwave days in May 2024?"
  python main.py --ask "How has the average May max temperature in Delhi changed since 2015?" --no-verify
  python main.py --benchmark                      (the 4 heatwave questions with known answers)
  python main.py --task heatwave_detection
  python main.py --task tmax_forecast --inject wrong_column
  python main.py --task advisory_json --provider ollama --review
  python main.py --custom "Plot the 10 hottest days in North Central and save their dates"
  python main.py --task hotspot_severity --provider mock --inject memory_bomb     (fully offline)
  python main.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows' legacy console codepage (cp1252/cp437) cannot encode characters an LLM may
# freely produce (em/en dashes, non-breaking hyphens, arrows, ...). Force UTF-8 on
# stdout/stderr so a stray Unicode character in model output never crashes the CLI.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from reflexion.console import ConsoleView, console, review
from reflexion.faults import FAULTS, inject, strip_markers
from reflexion.llm import GROQ_PREFERENCE, DEFAULT_OLLAMA_MODEL, FallbackLLM, LLMUnavailable, build_llm
from reflexion import dataset
from reflexion.ask import answer_question
from reflexion.loop import LoopConfig, ReflexionLoop
from reflexion.mock import scripted_llm
from reflexion.sandbox import SandboxLimits
from reflexion.tasks import TASKS, custom_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ask", metavar="QUESTION", help="plain-English question about the temperature data")
    g.add_argument("--task", choices=list(TASKS), help="one built-in benchmark task")
    g.add_argument("--benchmark", action="store_true", help="run all 4 benchmark tasks and summarise")
    g.add_argument("--custom", metavar="PROMPT", help="free-form task (generic checks only)")
    g.add_argument("--list", action="store_true", help="list tasks and injectable faults")
    p.add_argument("--data", choices=list(dataset.DATASETS), default=None,
                   help="real IMD data (default when downloaded) or synthetic")
    p.add_argument("--no-verify", action="store_true", help="--ask: skip the independent second solution")
    p.add_argument("--provider", choices=["auto", "groq", "ollama", "mock"], default="auto",
                   help="auto = Groq, falling back to local Ollama (default)")
    p.add_argument("--groq-model", default=GROQ_PREFERENCE[0])
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--inject", choices=list(FAULTS), help="replace the first script with a known-faulty one")
    p.add_argument("--review", action="store_true", help="human approval before each execution")
    p.add_argument("--show-code", action="store_true", help="print every generated script")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--repeat-limit", type=int, default=3, help="stop after N identical consecutive errors")
    p.add_argument("--unsafe-limit", type=int, default=3, help="stop after N sandbox-blocked attempts")
    p.add_argument("--timeout", type=float, default=60.0, help="per-execution wall-clock limit (s)")
    p.add_argument("--memory-mb", type=int, default=None, help="per-execution memory cap (MB); default per dataset")
    p.add_argument("--time-budget", type=float, default=None, help="total loop budget (s); default 600, 1800 on Ollama")
    p.add_argument("--token-budget", type=int, default=60_000)
    return p.parse_args()


def list_all() -> None:
    console.print("[bold]Benchmark tasks[/]")
    for t in TASKS.values():
        console.print(f"  [cyan]{t.id:20s}[/] {t.title}")
    console.print("\n[bold]Injectable faults[/] (caught at)")
    for name, (layer, desc, _) in FAULTS.items():
        console.print(f"  [magenta]{name:20s}[/] [{layer}] {desc}")
    console.print("\n[bold]Datasets[/]")
    for d in dataset.DATASETS.values():
        console.print(f"  [green]{d.name:10s}[/] {'ready' if d.available else 'missing'}  {d.description}")


def make_llm(args, view, task=None):
    if args.provider == "mock":
        if task is None or task.id not in TASKS:
            raise LLMUnavailable("the mock provider only knows the built-in benchmark tasks")
        return FallbackLLM([scripted_llm([strip_markers(task.reference_path.read_text(encoding="utf-8"))])])
    return build_llm(args.provider, args.groq_model, args.ollama_model, on_event=view.provider_event)


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = parse_args()
    if args.list:
        list_all()
        return 0
    if not (args.task or args.custom or args.ask or args.benchmark):
        console.print("[red]choose --ask, --task, --benchmark, --custom or --list[/]")
        return 2
    try:
        ds = dataset.use(args.data or dataset.active().name)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        return 2

    view = ConsoleView(show_code=args.show_code)
    first_task = TASKS[args.task] if args.task else None
    try:
        llm = make_llm(args, view, first_task)
    except LLMUnavailable as e:
        console.print(f"[red]{e}[/]")
        return 3

    active = llm.current
    cfg = LoopConfig(max_attempts=args.max_attempts, repeat_limit=args.repeat_limit, unsafe_limit=args.unsafe_limit,
                     time_budget_s=args.time_budget or (1800.0 if active.name == "ollama" else 600.0),
                     token_budget=args.token_budget,
                     limits=SandboxLimits(timeout_s=args.timeout, memory_mb=args.memory_mb or ds.memory_mb))
    console.print(f"[bold]Data:[/] {ds.description}\n[bold]LLM:[/] {active.name}:{active.model}"
                  f"{'  (fallback: ' + ', '.join(p.name for p in llm.providers[1:]) + ')' if len(llm.providers) > 1 else ''}\n"
                  f"[bold]Limits:[/] {cfg.max_attempts} attempts, {cfg.limits.timeout_s:.0f}s/run, {cfg.limits.memory_mb} MB, "
                  f"repeat {cfg.repeat_limit}, unsafe {cfg.unsafe_limit}, budget {cfg.time_budget_s:.0f}s / "
                  f"{cfg.token_budget:,} tok")
    reviewer = review if args.review else None
    try:
        if args.ask:
            console.print(f"[bold]Question:[/] {args.ask}")
            res = answer_question(args.ask, llm, cfg, verify=not args.no_verify, reviewer=reviewer, on_event=view)
            return 0 if res.primary.success else 1
        if args.benchmark:
            rows = []
            for task in TASKS.values():
                console.rule(f"[bold]{task.title}")
                r = ReflexionLoop(task, llm, cfg, reviewer=reviewer, on_event=view).run()
                rows.append((task.id, r.termination.value, len(r.attempts), r.tokens_used, r.elapsed_s))
            view.benchmark_summary(rows)
            return 0 if all(r[1] == "SUCCESS" for r in rows) else 1
        task = first_task or custom_task(args.custom)
        injected = None
        if args.inject:
            if args.custom:
                console.print("[red]--inject needs a built-in --task (faults mutate its reference solution)[/]")
                return 2
            injected = inject(task.reference_path.read_text(encoding="utf-8"), args.inject)
        console.print(f"[bold]Task:[/] {task.title}")
        result = ReflexionLoop(task, llm, cfg, injected_first_code=injected, injected_fault=args.inject,
                               reviewer=reviewer, on_event=view).run()
        return 0 if result.success else 1
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted by user[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
