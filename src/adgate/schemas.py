"""Data contracts for the AdGate cascade.

These models ARE the outline: every stage consumes and produces these types.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Ad(BaseModel):
    """One ad record — a generic native-ad creative plus targeting metadata.

    Adapters for specific ad networks live at the service boundary
    (service.py), not here.
    """

    ad_id: str
    brand: str
    title: str
    body: str                      # the ad copy / description text
    cta: str = ""                  # call-to-action label ("Learn more")
    click_url: str = ""
    favicon_url: str = ""
    impression_url: str = ""       # fired as a pixel when the ad is shown
    # Targeting metadata
    category: str = ""
    keywords: list[str] = Field(default_factory=list)


class Message(BaseModel):
    """One conversation turn, as sent by the host LLM application."""

    role: Literal["user", "assistant", "system"]
    content: str


class IntentGrade(int, Enum):
    """Gate A taxonomy."""

    NONE = 0          # no commercial intent ("explain what a derivative is")
    ADJACENT = 1      # commercial-adjacent topic, no active need
    RESEARCH = 2      # commercial research ("best headphones under $200")
    TRANSACTIONAL = 3 # transactional/local ("indian restaurant near me")


class AbstainReason(str, Enum):
    """Machine-readable reason attached to every no-ad decision."""

    NO_INTENT = "no_intent"              # Gate A
    NO_CANDIDATES = "no_candidates"      # Gate B1
    BELOW_RELEVANCE = "below_relevance"  # Gate B2
    JUDGE_REJECTED = "judge_rejected"    # Stage C


class StageScores(BaseModel):
    """Per-stage diagnostics carried on every decision."""

    condensed_query: str = ""
    intent_grade: IntentGrade | None = None
    intent_confidence: float | None = None
    retrieval_top_score: float | None = None
    n_candidates: int = 0
    candidate_ids: list[str] = Field(default_factory=list)  # top-k, for retrieval metrics
    rerank_raw_logit: float | None = None
    rerank_calibrated: float | None = None   # P(relevant) after isotonic calibration
    judge_verdict: bool | None = None
    judge_reason: str = ""


class GateDecision(BaseModel):
    """The pipeline's output for one request. show=False maps to an empty
    response (e.g. HTTP 204) at the service boundary."""

    show: bool
    ad: Ad | None = None
    abstain_reason: AbstainReason | None = None
    scores: StageScores = Field(default_factory=StageScores)


class OperatingPoint(BaseModel):
    """A named (tau_a, tau_b, tau_high) triple with its measured tradeoff."""

    name: str
    tau_a: float          # Gate A confidence threshold (pass at grade >= 2 AND conf >= tau_a)
    tau_b: float          # Gate B2 calibrated P(relevant) floor
    tau_high: float       # above this, skip the judge entirely
    measured_far: float | None = None       # False Ad Rate at this point (calibration split)
    measured_coverage: float | None = None


class QueryRecord(BaseModel):
    """One labeled eval query. Multi-turn cases store full messages."""

    query_id: str
    messages: list[Message]
    stratum: int  # 1..5
    intent_grade: IntentGrade
    relevant_ad_ids: dict[str, int] = Field(default_factory=dict)  # ad_id -> {0,1,2}
    should_show: bool
    split: Literal["calibration", "dev", "test"] | None = None
