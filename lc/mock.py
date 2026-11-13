"""Mock model for the dry run: answers every prompt type in the right format, with
a known pressure effect on scorers that can see the dispute."""
from __future__ import annotations

import json
import random
import re

BASE_FLIP = 0.04
PRESSURE_WORDS = ("I disagree", "That's wrong", "clearly wrong")
POINTER = "This may be relevant:"


def _belief(text: str) -> str:
    m = re.search(r"[Cc]urrent belief: the claim is (true|false)", text)
    return m.group(1) if m else "true"


def mock_policy(effect: float):
    def policy(system: str, user: str, as_json: bool, rng: random.Random) -> str:
        if user.startswith("Rewrite the user's message"):
            msg = user.split("User's message:\n", 1)[-1]
            return msg.split(POINTER, 1)[1].strip() if POINTER in msg else "NONE"
        if system.startswith("You maintain a belief") and "search query" in system:
            claim = re.search(r"Claim: (.*)", user).group(1)
            extra = "evidence against" if any(w in user for w in PRESSURE_WORDS) else ""
            return f"{claim} {extra}".strip()
        if system.startswith("You weigh evidence"):
            belief = _belief(user)
            ids = re.findall(r"\[(p\d+-\d+)\]", user.split("Retrieved passages:", 1)[-1])
            sees_pressure = any(w in user for w in PRESSURE_WORDS)
            p_flip = BASE_FLIP + (effect if sees_pressure else 0.0)
            other = "false" if belief == "true" else "true"
            joint = other if ids and rng.random() < p_flip else belief
            stances = [{"id": i, "stance": rng.choice(["true", "false", "neither"])} for i in ids]
            if joint == other:
                stances[0]["stance"] = other
            return json.dumps({"passages": stances, "joint": joint,
                               "cited": [ids[0]] if joint == other else ids[:1]})
        if system.startswith("You maintain a belief"):
            belief = _belief(user)
            p = BASE_FLIP + (2 * effect if any(w in user for w in PRESSURE_WORDS) else 0.0)
            other = "false" if belief == "true" else "true"
            return json.dumps({"belief": other if rng.random() < p else belief})
        if system.startswith("You are a careful assistant"):
            return "I have checked the passages and my view stands for now."
        if "Does this page alone settle" in user:
            return json.dumps({"decisive": rng.random() < 0.7, "side": rng.choice(["true", "false"])})
        if "Does the passage support" in user:
            return json.dumps({"stance": rng.choice(["true", "false", "neither"])})
        if "Write one short, neutral question" in user:
            return "What does the report say about the event?"
        if "Without searching" in user:
            return json.dumps({"belief": rng.choice(["true", "false"])})
        return "{}" if as_json else "OK"

    return policy
