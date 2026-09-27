# HeatDesk — a self-correcting code agent for IMD heatwave data

**Assignment:** ReAct / Reflexion loop — take a complex user task, generate Python code, run it in a
safe environment, capture tracebacks on failure, and iteratively fix the code.

**Use case (KJS-CES-01, IMD heatwave intelligence):** officials at disaster-management authorities,
municipal corporations and health departments ask questions about temperature data in plain English.
They cannot write code, and today they wait for an analyst. HeatDesk writes the analysis itself,
runs it safely, fixes its own mistakes, double-checks the answer and says how confident it is.

```bash
streamlit run app.py      # one page: ask a question, see the answer and the code the AI wrote and fixed
python main.py --ask "Which 5 places had the most heatwave days in May 2024?"   # terminal
python main.py --benchmark                                               # 4 questions with known answers
```

## 1. How it maps to the assignment

| Assignment asks for | Where |
|---|---|
| takes a complex user task | a free-form question (`--ask`, web page) or a benchmark task (`--task`) |
| generates Python code | GENERATE state — Groq `openai/gpt-oss-120b`, fallback local `qwen2.5-coder:3b` |
| runs it in a safe environment | EXECUTE state — subprocess + 5 more isolation layers (§5) |
| captures tracebacks on failure | structured error record from the child + regex fallback parser (§4) |
| iteratively fixes the code | REFLECT state + Reflexion memory, explicit termination conditions (§3) |

## 2. The loop

```
                +-----------+   code    +-----------+  result  +-----------+
      start --> | GENERATE  | --------> |  EXECUTE  | -------> | EVALUATE  | --pass--> DONE(SUCCESS)
                +-----------+           +-----------+          +-----------+
                      ^                                              | fail
                      |           lesson -> memory             +-----------+
                      +------------------------------------------ |  REFLECT  |
                                                                  +-----------+
```

* **GENERATE** — prompt = task/question + automatic profile of the data file + Reflexion memory +
  (from attempt 2) the previous script and its formatted failure. The ```python block is extracted.
* **EXECUTE** — sandboxed run (§5).
* **EVALUATE** — layers `STATIC → RUNTIME → ARTIFACT → SCHEMA → SEMANTIC`, stopping at the first failure.
* **REFLECT** — a separate LLM call returns `DIAGNOSIS / FIX_PLAN / LESSON`; the lesson joins a small,
  deduplicated memory shown on every later attempt.

**Checking answers to free-form questions** (no answer key exists):
1. *Sanity checks* — temperatures physically possible (−15…60 °C, catches the IMD `99.9` missing code),
   day counts not larger than the period asked about (catches "125 heatwave days in May"), dates inside
   the data, the answer text states the computed number, the chart is a real image.
2. *Self-consistency* — a second loop solves the question again independently (higher temperature).
   Results agree → **HIGH** confidence; disagree → **LOW**, both answers shown. Comparison ignores label
   wording/order and accepts ties at a top-N cut-off.

The 4 original heatwave tasks (`--benchmark`) *do* have answer keys: `reflexion/oracle.py` recomputes
the truth from the same data and the SEMANTIC layer compares.

## 3. Termination conditions (`reflexion/loop.py`)

| Condition | Default | Why |
|---|---|---|
| `SUCCESS` | — | all layers pass |
| `UNSAFE_CODE_LIMIT` | 3 blocked attempts | model keeps trying forbidden operations |
| `REPEATED_ERROR` | same error signature 3× in a row | model is stuck; signatures mask numbers/paths so line shifts don't hide a repeat |
| `MAX_ATTEMPTS` | 5 | checked before reflecting — no wasted LLM call |
| `TIME_BUDGET` | 600 s (1800 s on the local model) | wall-clock ceiling |
| `TOKEN_BUDGET` | 60,000 | cost ceiling |
| `LLM_UNAVAILABLE` | after retries + fallback | e.g. Groq down and Ollama not running |
| `USER_REJECTED` | — | `--review`: a human declined a script |

Groq's free tier allows ~8,000 tokens/minute; on a rate limit the client waits the time Groq asks for
(up to 60 s) before switching to the local model.

## 4. Error traces fed back to the LLM (`reflexion/feedback.py`, `prompts.py`)

* Only the **last** exception; only frames in the generated script; failing line with ±2 lines
  (`>> 20 | ...`); plus the last library frame as one clue (`base.py:3812 in get_loc`).
* **Hints** for facts the traceback doesn't show: real column names on `KeyError`, allowed modules
  on `ImportError`, the time/memory limits on `TIMEOUT` / `MEMORY_LIMIT`.
* Failed checks are listed as *expected vs got*; passing checks are listed so fixes don't break them.
* History = one line per earlier attempt; only the latest script is replayed in full → the prompt grows
  by about one line per attempt, not one script.

## 5. Sandbox (`sandbox.py`, `sandbox_boot.py`, `guard.py`)

| Layer | Stops |
|---|---|
| static AST guard | non-allowlisted imports, `eval/exec`, `os.system`, dunder escapes, pickle — before running |
| separate process | a crash or hang can't take down the loop |
| PEP 578 audit hook | subprocess, sockets, writes/deletes outside the sandbox — even inside `try/except` |
| Windows Job Object (POSIX `setrlimit`) | memory cap (1024 MB real data / 512 MB synthetic), 1 process |
| supervision | wall-clock timeout (60 s), output-size cap |
| hygiene | fresh folder per attempt, copied (not linked) data, stripped env (no API keys), stdin closed |

## 6. Data

* **Real (default):** IMD Pune 1° gridded daily maximum temperature 2015–2024, downloaded from
  imdpune.gov.in via `imdlib` — 1,296,815 rows, 355 land grid points.
  `python data/prepare_real_data.py --download` fetches and converts it. IMD files contain only
  date/lat/lon/tmax; **region** (IMD's 7 temperature regions), **terrain** and **nearest city** are
  approximate labels added by lat/lon rules. Cross-checked: the late-May-2024 peak appears over
  Rajasthan at ~48 °C (grid average; station records were ~50 °C).
* **Synthetic:** `python data/generate_data.py` — same columns, 35 grid points; used by the tests so they
  are fast and deterministic. Choose with `--data real|synthetic`.

## 7. Run

```bash
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY=...
python data/prepare_real_data.py --download
python data/generate_data.py
streamlit run app.py
python main.py --ask "How many days above 45 degC did the area near Delhi have each year?"
python main.py --task tmax_forecast --inject wrong_column    # force a failure, watch it self-fix
python main.py --list                                        # tasks, faults, datasets
pytest -q
```

Each run writes `runs/<time>_<task>/` with every prompt and response, `trace.json`, `report.md`, and
for questions `ask_summary.json`.

## 8. Limitations (honest)

* 1° grid ≈ 110 km: "near Nagpur" means the nearest grid cells, not the city station; districts,
  rainfall, humidity and minimum temperature are not in the data — the AI is told to say so.
* Heatwave normals use 10 years (IMD uses 30); results are indicative, not official warnings.
* Self-consistency catches unstable answers, not a mistake both solutions make the same way.
* The local model is slow on CPU (2–6 min per call); Groq's free tier rate-limits bursts.
* The sandbox timeout is wall-clock: heavy background load can cause false timeouts.
