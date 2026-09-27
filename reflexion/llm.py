"""LLM providers: Groq (primary), Ollama (local fallback), Mock (offline / tests).

All providers expose chat(messages) -> LLMResponse with token usage, so the loop can
enforce a token budget regardless of backend. FallbackLLM retries with exponential
backoff, then permanently switches to the next provider and logs the switch.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

GROQ_PREFERENCE = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
GROQ_OUTPUT_CAP = 4096  # free tier counts prompt + max output against 8,000 tokens/min; 8,192 made retries "too large"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:3b"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


class LLMError(Exception):
    """A single call failed (may be retried). retry_after: seconds the server asked us to wait."""

    def __init__(self, msg: str, retry_after: float = 0.0) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class LLMUnavailable(Exception):
    """No provider can serve requests any more -> loop terminates with LLM_UNAVAILABLE."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class GroqLLM:
    name = "groq"

    def __init__(self, model: str | None = None, reasoning_effort: str = "medium", exact: bool = False) -> None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMUnavailable("GROQ_API_KEY is not set (add it to .env)")
        from groq import Groq
        self.client = Groq(api_key=key, max_retries=0, timeout=120)
        self.reasoning_effort = reasoning_effort
        self.exact = exact
        self.model = self._resolve(model)

    def _resolve(self, wanted: str | None) -> str:
        try:
            available = {m.id for m in self.client.models.list().data}
        except Exception as e:  # network / auth problems surface here, before the loop starts
            raise LLMUnavailable(f"cannot list Groq models: {type(e).__name__}: {e}") from e
        for m in ([wanted] if wanted else []) + ([] if self.exact else GROQ_PREFERENCE):
            if m in available:
                return m
        raise LLMUnavailable(f"none of {[wanted] + GROQ_PREFERENCE} is available on Groq")

    def chat(self, messages: list[dict], max_tokens: int = 8192, temperature: float = 0.2) -> LLMResponse:
        import groq
        kwargs = dict(model=self.model, messages=messages, temperature=temperature,
                      max_completion_tokens=min(max_tokens, GROQ_OUTPUT_CAP))
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = self.reasoning_effort
        t0 = time.monotonic()
        try:
            r = self.client.chat.completions.create(**kwargs)
        except groq.AuthenticationError as e:
            raise LLMUnavailable(f"Groq rejected the API key: {e}") from e
        except groq.RateLimitError as e:  # free tier: tokens-per-minute limit -> wait as instructed
            raise LLMError(f"rate limited: {str(e)[:160]}", _retry_after(e)) from e
        except groq.APIStatusError as e:
            if e.status_code == 413:  # request exceeds this model's per-minute budget
                raise LLMError(f"request too large for {self.model}: {str(e)[:120]}", 60.0) from e
            raise LLMError(f"{type(e).__name__}: {str(e)[:300]}") from e
        except (groq.APIConnectionError, groq.APITimeoutError) as e:
            raise LLMError(f"{type(e).__name__}: {str(e)[:300]}") from e
        u = r.usage
        return LLMResponse(r.choices[0].message.content or "", self.name, self.model,
                           u.prompt_tokens if u else 0, u.completion_tokens if u else 0, time.monotonic() - t0)


def _retry_after(e) -> float:
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        m = re.search(r"try again in ((?:[\d.]+(?:h|ms|m|s))+)", str(e))
        if not m:
            return 10.0
        unit = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
        return sum(float(v) * unit[u] for v, u in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)))


class GroqPool:
    """Several Groq models behind one provider. Each model has its OWN rate limits, so when one is
    limited we switch to the next immediately instead of waiting; we only wait when all are limited."""
    name = "groq"

    def __init__(self, models: list[str] | None = None, max_wait_s: float = 90.0) -> None:
        self.clients, seen = [], set()
        for m in models or GROQ_PREFERENCE:
            try:
                c = GroqLLM(m, exact=True)
            except LLMUnavailable as e:
                if "GROQ_API_KEY" in str(e) or "API key" in str(e):
                    raise
                continue  # model not on this account
            if c.model not in seen:
                self.clients.append(c)
                seen.add(c.model)
        if not self.clients:
            raise LLMUnavailable("no Groq model from the pool is available on this account")
        self.cooldown: dict[str, float] = {}
        self.max_wait_s = max_wait_s
        self.model = " > ".join(c.model for c in self.clients)
        self.log: Callable[[str], None] = lambda msg: None

    def chat(self, messages: list[dict], max_tokens: int = GROQ_OUTPUT_CAP, temperature: float = 0.2) -> LLMResponse:
        while True:
            now = time.monotonic()
            ready = [c for c in self.clients if self.cooldown.get(c.model, 0) <= now]
            if not ready:
                wait = min(self.cooldown.values()) - now
                if wait > self.max_wait_s:
                    raise LLMError(f"all Groq models are rate-limited for {wait:.0f}s", wait)
                self.log(f"groq: all models busy - waiting {wait:.0f}s for the first to free up")
                time.sleep(max(wait, 0.5))
                continue
            c = ready[0]
            try:
                return c.chat(messages, max_tokens=max_tokens, temperature=temperature)
            except LLMError as e:
                # "too large" = this request can never fit that model's per-minute cap -> rest it longer
                rest = 300.0 if "too large" in str(e).lower() or "too lar" in str(e).lower() else max(e.retry_after, 5.0)
                self.cooldown[c.model] = time.monotonic() + rest
                nxt = [x.model for x in self.clients if self.cooldown.get(x.model, 0) <= time.monotonic()]
                self.log(f"groq: {c.model} limited ({str(e)[:70]}) -> "
                         + (f"switching to {nxt[0]}" if nxt else "no other model free"))


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str | None = None, num_ctx: int = 8192) -> None:
        self.model = model or DEFAULT_OLLAMA_MODEL
        self.num_ctx = num_ctx
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
                names = {m["name"] for m in json.load(r)["models"]}
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
            raise LLMUnavailable(f"Ollama server not reachable at {OLLAMA_URL} ({e})") from e
        if self.model not in names and f"{self.model}:latest" not in names:
            raise LLMUnavailable(f"Ollama model '{self.model}' not pulled (run: ollama pull {self.model})")

    def chat(self, messages: list[dict], max_tokens: int = 4096, temperature: float = 0.2) -> LLMResponse:
        body = json.dumps({"model": self.model, "messages": messages, "stream": False, "keep_alive": "10m",
                           "options": {"temperature": temperature, "num_ctx": self.num_ctx,
                                       "num_predict": max_tokens}}).encode()
        req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body, headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.load(r)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise LLMError(f"Ollama call failed: {e}") from e
        return LLMResponse(data["message"]["content"], self.name, self.model, data.get("prompt_eval_count", 0),
                           data.get("eval_count", 0), time.monotonic() - t0)


class MockLLM:
    """Deterministic offline LLM. `responder(messages, call_index) -> str`."""
    name = "mock"

    def __init__(self, responder: Callable[[list[dict], int], str], model: str = "scripted") -> None:
        self.responder, self.model, self.calls = responder, model, 0

    def chat(self, messages: list[dict], max_tokens: int = 0, temperature: float = 0.0) -> LLMResponse:
        text = self.responder(messages, self.calls)
        self.calls += 1
        prompt_chars = sum(len(m["content"]) for m in messages)
        return LLMResponse(text, self.name, self.model, prompt_chars // 4, len(text) // 4, 0.0)


@dataclass
class FallbackLLM:
    """Try providers in order; retry transient errors with backoff; switch on persistent failure."""
    providers: list
    retries: int = 4
    base_delay_s: float = 2.0
    max_wait_s: float = 60.0      # longest we wait for a rate-limit window before switching provider
    events: list[str] = field(default_factory=list)
    on_event: Callable[[str], None] | None = None
    _i: int = 0

    @property
    def current(self):
        return self.providers[self._i] if self._i < len(self.providers) else None

    def _log(self, msg: str) -> None:
        self.events.append(msg)
        if self.on_event:
            self.on_event(msg)

    def chat(self, messages: list[dict], **kw) -> LLMResponse:
        while self.current is not None:
            p = self.current
            for attempt in range(1, self.retries + 1):
                try:
                    return p.chat(messages, **kw)
                except LLMUnavailable as e:
                    self._log(f"{p.name}: unavailable ({e})")
                    break
                except LLMError as e:
                    delay = min(self.max_wait_s, max(self.base_delay_s * 2 ** (attempt - 1), e.retry_after + 1))
                    self._log(f"{p.name}: call {attempt}/{self.retries} failed ({e})"
                              + (f"; retrying in {delay:.0f}s" if attempt < self.retries else ""))
                    if attempt < self.retries:
                        time.sleep(delay)
            self._i += 1
            if self.current is not None:
                self._log(f"switching provider: {p.name} -> {self.current.name}:{self.current.model}")
        self._i = 0  # the next request starts again with the preferred provider (limits are temporary)
        raise LLMUnavailable("all LLM providers failed")


def build_llm(provider: str = "auto", groq_model: str | None = None, ollama_model: str | None = None,
              on_event: Callable[[str], None] | None = None) -> FallbackLLM:
    """provider: auto (groq -> ollama) | groq | ollama."""
    chain, notes = [], []
    wanted = {"auto": ["groq", "ollama"], "groq": ["groq"], "ollama": ["ollama"]}[provider]
    for name in wanted:
        try:
            chain.append((GroqPool([groq_model] + [m for m in GROQ_PREFERENCE if m != groq_model])
                          if groq_model else GroqPool()) if name == "groq" else OllamaLLM(ollama_model))
        except LLMUnavailable as e:
            notes.append(f"{name}: skipped - {e}")
    llm = FallbackLLM(chain, on_event=on_event)
    for p in chain:
        if isinstance(p, GroqPool):
            p.log = llm._log
    for n in notes:
        llm._log(n)
    if not chain:
        raise LLMUnavailable("no LLM provider available:\n  " + "\n  ".join(notes))
    return llm
