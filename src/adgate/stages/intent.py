"""Gate A — commercial intent classification.

First run ships two backends, selected by config.intent_backend:

  "heuristic"  pattern rules, zero cost, no keys — the default so the whole
               pipeline runs offline
  "llm"        few-shot LLM classification via litellm, used when an API
               key is available
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from ..config import AdGateConfig
from ..prompts import INTENT_FEW_SHOT_PROMPT
from ..schemas import IntentGrade


class IntentResult(BaseModel):
    grade: IntentGrade
    confidence: float  # 0..1
    reason: str = ""


# Patterns that mark a query as informational / non-commercial.
_INFORMATIONAL = re.compile(
    r"^(what|why|how|who|when|where|explain|summarize|translate|prove|convert"
    r"|write|help me|give me|tell me)\b"
    r"|\b(meaning of|difference between|history of|definition of|rhymes with"
    r"|grammar check|debug|function|segfault)\b",
    re.IGNORECASE,
)

# Patterns that mark active transactional intent.
_TRANSACTIONAL = re.compile(
    r"\b(buy|purchase|order|near me|cheapest|discount|deal|coupon|price of"
    r"|best .* under|shop for|book a)\b",
    re.IGNORECASE,
)


def keyword_prefilter(query: str) -> bool:
    """True if the query is an obvious non-commercial and can short-circuit
    to abstain(NO_INTENT) with zero model cost."""
    return bool(_INFORMATIONAL.search(query)) and not _TRANSACTIONAL.search(query)


class HeuristicIntentClassifier:
    """Pattern-rule classifier so the first run needs no API key.

    Logic: informational patterns -> NONE; transactional patterns ->
    TRANSACTIONAL; everything else (bare noun phrases like "bathroom fan
    without light") -> RESEARCH, because product searches rarely contain
    commercial keywords — they're just nouns.

    IMPROVEMENT: this is dataset-shaped and brittle. Replace with
    LLMIntentClassifier (below) or the zero-shot NLI fallback
    (MoritzLaurer/deberta-v3-large-zeroshot-v2.0) for real use. It also
    cannot produce grade ADJACENT (1) at all.
    """

    def __init__(self, config: AdGateConfig) -> None:
        self.config = config

    def classify(self, query: str) -> IntentResult:
        if _TRANSACTIONAL.search(query):
            return IntentResult(grade=IntentGrade.TRANSACTIONAL, confidence=0.9,
                                reason="transactional pattern")
        if _INFORMATIONAL.search(query):
            return IntentResult(grade=IntentGrade.NONE, confidence=0.9,
                                reason="informational pattern")
        # IMPROVEMENT: a fixed confidence carries no signal — an LLM's
        # logprobs or NLI entailment score would make tau_a meaningful.
        return IntentResult(grade=IntentGrade.RESEARCH, confidence=0.8,
                            reason="default: looks like a product/service search")


class LLMIntentClassifier:
    """Few-shot LLM classifier via litellm.

    IMPROVEMENT: confidence should come from the grade token's logprobs
    (or a 3-sample self-consistency vote), not the fixed 0.85 here.
    """

    def __init__(self, config: AdGateConfig) -> None:
        self.config = config

    def classify(self, query: str) -> IntentResult:
        import litellm  # deferred: only needed for this backend

        resp = litellm.completion(
            model=self.config.intent_model,
            messages=[
                {"role": "system", "content": INTENT_FEW_SHOT_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            grade = IntentGrade(int(data["grade"]))
            return IntentResult(grade=grade, confidence=0.85,
                                reason=data.get("reason", ""))
        except (json.JSONDecodeError, KeyError, ValueError):
            # Unparseable response: fail toward abstention (precision-first).
            return IntentResult(grade=IntentGrade.NONE, confidence=0.5,
                                reason="unparseable LLM response")


def make_intent_classifier(config: AdGateConfig):
    if config.intent_backend == "llm":
        return LLMIntentClassifier(config)
    return HeuristicIntentClassifier(config)
