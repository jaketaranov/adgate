"""Gate B2 — cross-encoder reranking.

The cross-encoder reads query+ad JOINTLY (full attention between them) and
separates relevant from near-miss far more sharply than retrieval scores.

⚠ Raw scores are UNCALIBRATED and not comparable across query types. They
must pass through CalibrationModel.calibrate() (fit by adgate.calibrate)
before any threshold comparison. This module returns raw scores.
"""

from __future__ import annotations

from ..config import AdGateConfig
from ..schemas import Ad
from .retrieve import ad_to_document


class CrossEncoderReranker:
    def __init__(self, config: AdGateConfig) -> None:
        self.config = config
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # deferred: heavy

            # IMPROVEMENT: default is the small ms-marco MiniLM for CPU
            # speed; BAAI/bge-reranker-v2-m3 is the quality pick.
            self._model = CrossEncoder(self.config.reranker_model)
        return self._model

    def score(self, query: str, candidates: list[Ad]) -> list[float]:
        """Raw reranker scores for (query, ad_document) pairs, same order as
        candidates. Higher = more relevant, scale is model-specific."""
        if not candidates:
            return []
        pairs = [(query, ad_to_document(ad)) for ad in candidates]
        return [float(s) for s in self._get_model().predict(pairs)]
