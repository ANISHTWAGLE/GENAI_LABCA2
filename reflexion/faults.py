"""Deliberate fault injection for demonstrating failure mechanics.

`--inject <fault>` replaces the FIRST generated script with a mutated copy of the task's
reference solution. Each fault targets a specific evaluation / sandbox layer, so every
branch of the loop can be demonstrated on demand instead of waiting for the LLM to fail.
"""
from __future__ import annotations

import re

# fault -> (layer that must catch it, description, code inserted at the # <<FAULT>> marker)
FAULTS: dict[str, tuple[str, str, str]] = {
    "wrong_column": ("RUNTIME", "KeyError: uses 'Tmax' instead of 'tmax'",
                     'print("mean Tmax:", df["Tmax"].mean())'),
    "infinite_loop": ("RUNTIME/timeout", "row loop whose counter only advances on a rare condition",
                      "i = 0\nwhile i < len(df):\n    if df.iloc[i]['tmax'] > 100:\n        i += 1"),
    "memory_bomb": ("RUNTIME/memory", "caches ~5 GB of copies -> Job Object memory cap",
                    "cache = [bytearray(50_000_000) for _ in range(100)]"),
    "blocked_import": ("STATIC", "imports subprocess (not on the allowlist)",
                       'import subprocess\nsubprocess.run(["whoami"])'),
    "network": ("RUNTIME/audit-hook", "downloads normals over HTTP -> socket blocked by audit hook",
                'normals = pd.read_csv("https://imdpune.gov.in/normals.csv")'),
    "file_escape": ("RUNTIME/audit-hook", "writes a file outside the sandbox directory",
                    'open("../../../leaked_forecast.csv", "w").write("x")'),
    "swallowed_violation": ("RUNTIME/audit-hook", "hides a blocked write inside try/except - still caught",
                            'try:\n    open("C:/Windows/Temp/cache.csv", "w")\nexcept Exception:\n    pass'),
    "syntax_error": ("STATIC", "unbalanced parenthesis", 'print("loaded rows:", len(df)'),
    "sentinel": ("SEMANTIC", "forgets to drop the 99.9 missing-value code -> wrong numbers", None),
    "bad_schema": ("SCHEMA", "omits a required key from result.json", None),
}


def strip_markers(code: str) -> str:
    """Remove the <<...>> mutation markers so they never leak into prompts."""
    out = []
    for l in code.splitlines():
        if l.strip() in {"# <<FAULT>>"}:
            continue
        out.append(re.sub(r"\s*# <<\w+>>", "", l))
    return "\n".join(out) + "\n"


def inject(reference_code: str, fault: str) -> str:
    return strip_markers(_inject(reference_code, fault))


def _inject(reference_code: str, fault: str) -> str:
    if fault not in FAULTS:
        raise ValueError(f"unknown fault '{fault}'. Choose from: {', '.join(FAULTS)}")
    lines = reference_code.splitlines()
    if fault == "sentinel":   # drop the cleaning line, keep only the NaN filter
        return "\n".join(re.sub(r"df\s*=\s*df\[.*\]\s*(\.copy\(\))?\s*# <<CLEAN>>", "df = df.dropna(subset=['tmax'])", l)
                         for l in lines)
    if fault == "bad_schema":  # the key named on the <<SCHEMA>> line never reaches result.json
        key = next(re.search(r'"(\w+)"', l).group(1) for l in lines if "# <<SCHEMA>>" in l)
        return "\n".join(l.replace("json.dump(result,", f"json.dump({{k: v for k, v in result.items() if k != {key!r}}},")
                         for l in lines)
    snippet = FAULTS[fault][2]
    out = []
    for l in lines:
        if "# <<FAULT>>" in l:
            indent = l[: len(l) - len(l.lstrip())]
            out.extend(indent + s for s in snippet.splitlines())
        else:
            out.append(l)
    return "\n".join(out)
