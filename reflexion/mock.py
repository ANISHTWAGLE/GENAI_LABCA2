"""Scripted offline 'LLM' for demos without internet and for deterministic tests."""
from __future__ import annotations

import re

from .llm import MockLLM


def _reflection_from_feedback(prompt: str) -> str:
    err = re.search(r"^ERROR: (.+)$", prompt, re.M)
    layer = re.search(r"^FAILED AT LAYER: (\w+)", prompt, re.M)
    hint = re.search(r"^HINT: (.+)$", prompt, re.M)
    check = re.search(r"^  - (\w[\w:<>=]*): (.+)$", prompt, re.M)
    what = err.group(1) if err else (f"check {check.group(1)} failed: {check.group(2)}" if check else "unknown")
    return (f"DIAGNOSIS: The script failed at the {layer.group(1) if layer else '?'} layer because {what[:200]}.\n"
            f"FIX_PLAN: Remove the faulty statement and follow the task rules exactly.\n"
            f"LESSON: {(hint.group(1) if hint else 'Validate outputs against the stated rules before writing them.')[:160]}")


def scripted_llm(solutions: list[str]) -> MockLLM:
    """Actor calls return `solutions` in order (the last one repeats); reflector calls summarise the feedback."""
    state = {"actor": 0}

    def respond(messages: list[dict], _i: int) -> str:
        if "debugging reviewer" in messages[0]["content"]:
            return _reflection_from_feedback(messages[-1]["content"])
        code = solutions[min(state["actor"], len(solutions) - 1)]
        state["actor"] += 1
        return f"```python\n{code}\n```"

    return MockLLM(respond)
