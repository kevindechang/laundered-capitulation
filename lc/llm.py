"""Model calls: one interface, a mock for dry runs, and Vertex for real runs.

Every real call is priced and added to a ledger; the ledger refuses calls past
the budget, so a run stops at the limit rather than overspending.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

# USD per 1M tokens (input, output; thinking billed as output). UNCONFIRMED
# placeholders set high on purpose; the pilot replaces them with Google's live
# price and checks them against the billing report before the lock.
PRICES = {
    "gemini-3.8-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.25, 1.50),
    "gemma-4-26b-a4b-it-maas": (0.15, 0.60),
}
FLEX_DISCOUNT = 0.5
THINKING_LEVEL = {"gemini-3.8-flash": "low"}  # others: "minimal"
MAX_RETRIES = 4


@dataclass(frozen=True)
class Reply:
    text: str
    tokens_in: int
    tokens_out: int
    tokens_thought: int
    cost: float
    failed: bool = False


class Model(Protocol):
    def complete(self, model: str, system: str, user: str, *, max_tokens: int,
                 as_json: bool = False) -> Reply: ...


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Thread-safe spend tracker with a hard cap, appended to a JSONL file."""

    def __init__(self, cap_usd: float, path: Path | None = None):
        self.cap = cap_usd
        self.spent = 0.0
        self.calls = 0
        self.failures = 0
        self._path = path
        self._lock = threading.Lock()

    def check(self) -> None:
        if self.spent >= self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.4f} of cap ${self.cap:.2f}")

    def add(self, model: str, reply: Reply, tag: str) -> None:
        with self._lock:
            self.spent += reply.cost
            self.calls += 1
            self.failures += int(reply.failed)
            if self._path:
                with self._path.open("a") as fh:
                    fh.write(json.dumps({"t": time.time(), "model": model, "tag": tag,
                                         "in": reply.tokens_in, "out": reply.tokens_out,
                                         "thought": reply.tokens_thought, "cost": reply.cost,
                                         "failed": reply.failed}) + "\n")


def price(model: str, tokens_in: int, tokens_out: int, *, flex: bool) -> float:
    p_in, p_out = PRICES[model]
    cost = (tokens_in * p_in + tokens_out * p_out) / 1e6
    return cost * FLEX_DISCOUNT if flex and not model.startswith("gemma") else cost


class Metered:
    """Wraps a Model so every call is checked against and added to the ledger."""

    def __init__(self, inner: Model, ledger: Ledger):
        self.inner = inner
        self.ledger = ledger

    def complete(self, model: str, system: str, user: str, *, max_tokens: int,
                 as_json: bool = False, tag: str = "") -> Reply:
        self.ledger.check()
        reply = self.inner.complete(model, system, user, max_tokens=max_tokens, as_json=as_json)
        self.ledger.add(model, reply, tag)
        return reply


class VertexModel:
    def __init__(self, project: str, *, flex: bool = False, location: str = "global"):
        from google import genai
        from google.genai import types

        headers = {"X-Vertex-AI-LLM-Shared-Request-Type": "flex"} if flex else {}
        self._types = types
        self._flex = flex
        self._client = genai.Client(vertexai=True, project=project, location=location,
                                    http_options=types.HttpOptions(headers=headers))

    def _config(self, model: str, system: str, max_tokens: int, as_json: bool):
        t = self._types
        kwargs = dict(system_instruction=system or None, max_output_tokens=max_tokens,
                      temperature=0.0)
        if model.startswith("gemini"):
            kwargs["thinking_config"] = t.ThinkingConfig(
                thinking_level=THINKING_LEVEL.get(model, "minimal"))
        if as_json:
            kwargs["response_mime_type"] = "application/json"
        return t.GenerateContentConfig(**kwargs)

    def complete(self, model: str, system: str, user: str, *, max_tokens: int,
                 as_json: bool = False) -> Reply:
        config = self._config(model, system, max_tokens, as_json)
        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.models.generate_content(model=model, contents=user, config=config)
                usage = resp.usage_metadata
                t_in = usage.prompt_token_count or 0
                t_out = usage.candidates_token_count or 0
                t_th = usage.thoughts_token_count or 0
                return Reply(resp.text or "", t_in, t_out, t_th,
                             price(model, t_in, t_out + t_th, flex=self._flex))
            except Exception as exc:  # retried, then recorded as a failed call
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 ** attempt + random.random())
        return Reply(f"ERROR {last_error}", 0, 0, 0, 0.0, failed=True)


class MockModel:
    """Deterministic stand-in. `policy(system, user, as_json, rng)` returns text."""

    def __init__(self, policy: Callable[[str, str, bool, random.Random], str]):
        self.policy = policy

    def complete(self, model: str, system: str, user: str, *, max_tokens: int,
                 as_json: bool = False) -> Reply:
        seed = int(hashlib.sha256(f"{model}|{system}|{user}".encode()).hexdigest()[:12], 16)
        text = self.policy(system, user, as_json, random.Random(seed))
        n_in = (len(system) + len(user)) // 4
        return Reply(text, n_in, len(text) // 4, 0, price(model if model in PRICES else
                     "gemini-3.1-flash-lite", n_in, len(text) // 4, flex=False))


def parse_json(text: str) -> dict:
    """Parse a model's JSON answer, tolerating code fences. Empty dict if unusable."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        body = body[body.find("{"):]
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < 0:
        return {}
    try:
        value = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
