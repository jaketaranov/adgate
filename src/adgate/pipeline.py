"""The 4-stage cascade orchestrator.

Stage 0 → Gate A → Gate B1 → Gate B2 → Stage C, with early-exit abstention
at every stage. Each abstain carries a machine-readable reason.
"""

from __future__ import annotations

import math
from pathlib import Path

from .calibration import CalibrationModel
from .config import OPERATING_POINTS, AdGateConfig
from .schemas import AbstainReason, GateDecision, Message, StageScores
from .stages.condense import condense_last_turns, rewrite_query_llm
from .stages.intent import make_intent_classifier
from .stages.judge import LLMJudge
from .stages.rerank import CrossEncoderReranker
from .stages.retrieve import AdIndex


class AdGate:
    """decide(messages) -> GateDecision. show=False maps to HTTP 204 upstream."""

    def __init__(self, config: AdGateConfig, index: AdIndex) -> None:
        self.config = config
        self.op = OPERATING_POINTS[config.operating_point]
        self.index = index
        self.intent = make_intent_classifier(config)
        self.reranker = (
            CrossEncoderReranker(config) if config.rerank_backend == "cross-encoder" else None
        )
        self.judge = LLMJudge(config) if config.use_judge else None

        # Calibration: if adgate.calibrate has run, its isotonic mapping +
        # Wilson-selected tau_b override the eyeballed fallbacks. An empty
        # calibration_path disables loading entirely (hermetic tests).
        self.calib: CalibrationModel | None = None
        self.tau_b = self.op.tau_b
        if config.calibration_path and Path(config.calibration_path).exists():
            self.calib = CalibrationModel.load(Path(config.calibration_path))
            if self.op.name in self.calib.taus:
                self.tau_b = self.calib.taus[self.op.name]

    def decide(self, messages: list[Message]) -> GateDecision:
        # Stage 0: condense the conversation into one query string
        if self.config.query_rewrite == "llm":
            query = rewrite_query_llm(messages, self.config)
        else:
            query = condense_last_turns(messages)
        scores = StageScores(condensed_query=query or "")
        if not query:
            return self._abstain(AbstainReason.NO_INTENT, scores)

        # Gate A: commercial intent
        intent = self.intent.classify(query)
        scores.intent_grade = intent.grade
        scores.intent_confidence = intent.confidence
        if intent.grade < self.config.min_intent_grade or intent.confidence < self.op.tau_a:
            return self._abstain(AbstainReason.NO_INTENT, scores)

        # Gate B1: retrieve candidate ads
        candidates = self.index.search(query)
        scores.n_candidates = len(candidates)
        scores.candidate_ids = [c.ad.ad_id for c in candidates]
        if not candidates:
            return self._abstain(AbstainReason.NO_CANDIDATES, scores)

        # Gate B2: is the best candidate actually good enough?
        if self.reranker is not None:
            # Rerank the head of the retrieval pool; the winner may differ
            # from retrieval's #1 (that reshuffle is half the value).
            head = candidates[: self.config.rerank_top_n]
            raw = self.reranker.score(query, [c.ad for c in head])
            best_i = max(range(len(raw)), key=raw.__getitem__)
            top, raw_best = head[best_i], raw[best_i]
            scores.rerank_raw_logit = raw_best
            if self.calib is not None:
                p = self.calib.calibrate(raw_best)
            else:
                # IMPROVEMENT: sigmoid is NOT calibration — run
                # `python -m adgate.calibrate` to replace this fallback.
                p = 1.0 / (1.0 + math.exp(-raw_best))
        else:
            # No-reranker path (first-run behavior): squash retrieval score.
            # IMPROVEMENT: not a probability; kept for comparison.
            top = candidates[0]
            scores.retrieval_top_score = top.score
            p = top.score / (1.0 + top.score)

        scores.rerank_calibrated = p
        if p < self.tau_b:
            return self._abstain(AbstainReason.BELOW_RELEVANCE, scores)

        # Stage C: LLM judge on the borderline band only
        if self.judge is not None and p < self.op.tau_high:
            verdict, reason = self.judge.judge(messages, top.ad)
            scores.judge_verdict = verdict
            scores.judge_reason = reason
            if not verdict:
                return self._abstain(AbstainReason.JUDGE_REJECTED, scores)

        return GateDecision(show=True, ad=top.ad, scores=scores)

    @staticmethod
    def _abstain(reason: AbstainReason, scores: StageScores) -> GateDecision:
        return GateDecision(show=False, abstain_reason=reason, scores=scores)
