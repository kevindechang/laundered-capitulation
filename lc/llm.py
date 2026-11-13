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
    # Fireworks list price, uncached input (cached input is cheaper, so this overstates).
    "accounts/fireworks/models/deepseek-v4p1-flash": (0.30, 1.20),
    "deepseek/deepseek-v4.1-flash": (0.30, 1.20),  # OpenRouter, pinned to Fireworks
}
DEEPSEEK = "accounts/fireworks/models/deepseek-v4p1-flash"
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
HTTP_TIMEOUT_S = 120
FIREWORKS_REASONING = {DEEPSEEK: "none"}
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DEEPSEEK = "deepseek/deepseek-v4.1-flash"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_PROVIDER = {"order": ["fireworks"], "allow_fallbacks": False, "require_parameters": True}
OPENROUTER_REASONING = {OPENROUTER_DEEPSEEK: {"enabled": False}}  # thinking off; it otherwise eats the token limit
FLEX_DISCOUNT = 0.5
FLEX_MODELS = frozenset({"gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"})
THINKING_LEVEL = {"gemini-3.8-flash": "low"}  # others: "minimal"
MAX_RETRIES = 7
MAX_BACKOFF_S = 60  # quota errors (429) clear within about a minute


@dataclass(frozen=True)
class Reply:
    text: str
    tokens_in: int
    tokens_out: int
    tokens_thought: int
    cost: float
    failed: bool = False
    finish: str = ""


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
        if path and path.exists():  # the cap holds across restarts and commands
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.spent += row.get("cost", 0.0)
                    self.calls += 1
                    self.failures += int(row.get("failed", False))

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
                                         "failed": reply.failed, "finish": reply.finish}) + "\n")


def price(model: str, tokens_in: int, tokens_out: int, *, flex: bool) -> float:
    p_in, p_out = PRICES[model]
    cost = (tokens_in * p_in + tokens_out * p_out) / 1e6
    return cost * FLEX_DISCOUNT if flex and model in FLEX_MODELS else cost


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

        self._types = types
        self._flex = flex
        self._standard = genai.Client(vertexai=True, project=project, location=location)
        self._flex_client = genai.Client(
            vertexai=True, project=project, location=location,
            http_options=types.HttpOptions(headers={"X-Vertex-AI-LLM-Shared-Request-Type": "flex"}),
        ) if flex else None

    def _uses_flex(self, model: str) -> bool:
        return self._flex and model in FLEX_MODELS

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
                flex = self._uses_flex(model)
                client = self._flex_client if flex else self._standard
                resp = client.models.generate_content(model=model, contents=user, config=config)
                usage = resp.usage_metadata
                t_in = usage.prompt_token_count or 0
                t_out = usage.candidates_token_count or 0
                t_th = usage.thoughts_token_count or 0
                finish = str(resp.candidates[0].finish_reason) if resp.candidates else "NONE"
                text = resp.text or ""
                return Reply(text, t_in, t_out, t_th, price(model, t_in, t_out + t_th, flex=flex),
                             failed=not text.strip(), finish=finish)
            except Exception as exc:  # retried, then recorded as a failed call
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2 ** attempt, MAX_BACKOFF_S) * (1 + random.random()))
        return Reply(f"ERROR {last_error}", 0, 0, 0, 0.0, failed=True, finish=f"ERROR {last_error[:200]}")


class FireworksModel:
    """Fireworks chat-completions client (OpenAI-compatible), stdlib only.
    `post` is injectable for tests; it takes the JSON payload and returns the JSON reply."""

    url = FIREWORKS_URL

    def __init__(self, api_key: str, *, post: Callable[[dict], dict] | None = None):
        self._key = api_key
        self._post = post or self._http_post

    def _http_post(self, payload: dict) -> dict:
        import urllib.request

        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self._key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read())

    def complete(self, model: str, system: str, user: str, *, max_tokens: int,
                 as_json: bool = False) -> Reply:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        payload.update(self._route(model))
        if as_json:
            payload["response_format"] = {"type": "json_object"}
        last_error = ""
        for attempt in range(MAX_RETRIES):
            try:
                data = self._post(payload)
                self._check_provider(data)
                choice = data["choices"][0]
                text = choice["message"].get("content") or ""
                usage = data.get("usage", {})
                t_in, t_out = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                return Reply(text, t_in, t_out, 0, self._cost(model, usage, t_in, t_out),
                             failed=not text.strip(), finish=str(choice.get("finish_reason", "")))
            except Exception as exc:  # retried, then recorded as a failed call
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2 ** attempt, MAX_BACKOFF_S) * (1 + random.random()))
        return Reply(f"ERROR {last_error}", 0, 0, 0, 0.0, failed=True, finish=f"ERROR {last_error[:200]}")

    def _route(self, model: str) -> dict:
        return {"reasoning_effort": FIREWORKS_REASONING[model]} if model in FIREWORKS_REASONING else {}

    def _check_provider(self, data: dict) -> None:
        """Direct Fireworks calls have no routing to check."""

    def _cost(self, model: str, usage: dict, t_in: int, t_out: int) -> float:
        return price(model, t_in, t_out, flex=False)


class OpenRouterModel(FireworksModel):
    """Same client through OpenRouter, pinned to Fireworks (no fallbacks), reasoning off.
    A reply served by any other provider raises, so it is retried and never recorded as data."""

    url = OPENROUTER_URL

    def _route(self, model: str) -> dict:
        route = {"provider": dict(OPENROUTER_PROVIDER)}
        if model in OPENROUTER_REASONING:
            route["reasoning"] = dict(OPENROUTER_REASONING[model])
        return route

    def _cost(self, model: str, usage: dict, t_in: int, t_out: int) -> float:
        """What OpenRouter actually billed, else list price."""
        billed = usage.get("cost")
        return float(billed) if isinstance(billed, (int, float)) else price(model, t_in, t_out, flex=False)

    def _check_provider(self, data: dict) -> None:
        served = str(data.get("provider", ""))
        if served.lower() != OPENROUTER_PROVIDER["order"][0]:
            raise RuntimeError(f"served by {served or 'unknown provider'}, not Fireworks")


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
