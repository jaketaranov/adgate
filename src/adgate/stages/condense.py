"""Stage 0 — context condensation.

Default: last-2-turns concatenation (what host apps typically send anyway).
Optional: LLM query rewriting for multi-turn intent ("what about vegetarian
options?" after a restaurant discussion). Its `None` output doubles as an
early abstain signal.
"""

from __future__ import annotations

from ..config import AdGateConfig
from ..prompts import QUERY_REWRITE_PROMPT
from ..schemas import Message


def condense_last_turns(messages: list[Message], n: int = 2) -> str:
    """Concatenate the last n non-system turns into one intent-query string.

    IMPROVEMENT: replace with rewrite_query_llm for multi-turn conversations
    where the current need isn't in the last turns.
    """
    turns = [m for m in messages if m.role != "system"][-n:]
    return " ".join(m.content.strip() for m in turns if m.content.strip())


def rewrite_query_llm(messages: list[Message], config: AdGateConfig) -> str | None:
    """LLM rewrite for multi-turn intent. Returns None when the model
    outputs NONE (doubles as an early abstain signal) or on any failure
    (precision-first: fail toward abstention).

    Enabled with config.query_rewrite="llm" (needs an API key)."""
    import litellm  # deferred: only needed for this backend

    convo = "\n".join(f"{m.role}: {m.content}" for m in messages[-6:] if m.role != "system")
    try:
        resp = litellm.completion(
            model=config.rewrite_model,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_PROMPT},
                {"role": "user", "content": convo},
            ],
            temperature=0,
            max_tokens=60,
        )
        out = resp.choices[0].message.content.strip()
        return None if not out or out.upper() == "NONE" else out
    except Exception:  # noqa: BLE001 — fail closed by design
        return None
