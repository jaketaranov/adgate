"""Configuration: model picks, retrieval knobs, named operating points.

Model choices and defaults sized for CPU-only machines. IMPROVEMENT: the
full-quality picks are BAAI/bge-large-en-v1.5 (embeddings) and
BAAI/bge-reranker-v2-m3 (reranker) — swap them in when a GPU or more
patience is available; nothing else changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from .schemas import OperatingPoint


class AdGateConfig(BaseModel):
    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge/e5 family requires an asymmetric query prefix; passages get none
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # litellm model ids for Gate A / judge / rewriting — provider-agnostic.
    # Family separation rule: keep generator/labeler/judge distinct.
    intent_model: str = "claude-haiku-4-5-20251001"
    judge_model: str = "gpt-4o-mini"
    rewrite_model: str = "claude-haiku-4-5-20251001"
    nli_fallback_model: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

    # Stage 0
    # "last_turns" = concat last 2 turns; "llm" = standalone-query rewrite
    query_rewrite: str = "last_turns"

    # Gate A
    # "heuristic" = pattern rules (offline, default); "llm" = few-shot via litellm
    intent_backend: str = "heuristic"
    min_intent_grade: int = 2    # pass at RESEARCH or above; grade-1 is a business knob

    # Gate B1: retrieval
    # "bm25" (offline, no models), "dense" (embeddings), "hybrid" (RRF fusion)
    retrieval_backend: str = "hybrid"
    top_k: int = 20
    score_floor: float = 0.0     # raw BM25 garbage guard
    cosine_floor: float = 0.35   # dense garbage guard
    rrf_k: int = 60

    # Gate B2: rerank + calibration
    # "none" = squash the retrieval score (first-run behavior); "cross-encoder"
    rerank_backend: str = "cross-encoder"
    rerank_top_n: int = 10       # rerank only the best-retrieved N of top_k
    # Written by adgate.calibrate; empty string disables calibration loading.
    calibration_path: str = "data/calibration.json"

    # Stage C: judge
    use_judge: bool = False      # needs an LLM API key (or a local Ollama model)

    # Operating point
    # "conservative" | "balanced" | "aggressive"
    operating_point: str = "balanced"

    # Data paths
    ads_path: str = "data/ads.jsonl"
    queries_path: str = "data/queries.jsonl"


# Fallback triples used ONLY when data/calibration.json doesn't exist yet.
# Once `python -m adgate.calibrate` has run, its Wilson-selected tau_b values
# (stored in calibration.json) override tau_b per operating point.
OPERATING_POINTS: dict[str, OperatingPoint] = {
    "conservative": OperatingPoint(name="conservative", tau_a=0.85, tau_b=0.90, tau_high=0.97),
    "balanced":     OperatingPoint(name="balanced",     tau_a=0.75, tau_b=0.80, tau_high=0.97),
    "aggressive":   OperatingPoint(name="aggressive",   tau_a=0.60, tau_b=0.70, tau_high=0.97),
}

# Shown-ad precision targets used by adgate.calibrate to derive each
# operating point's tau_b (smallest tau whose Wilson lower bound clears this).
PRECISION_TARGETS: dict[str, float] = {
    "conservative": 0.95,
    "balanced": 0.90,
    "aggressive": 0.80,
}

DEFAULT_CONFIG = AdGateConfig()

# Offline config: no model downloads, no API keys — what the unit tests use.
# calibration_path="" keeps tests hermetic: whatever data/calibration.json
# happens to be on disk must not change unit-test behavior.
OFFLINE_CONFIG = AdGateConfig(retrieval_backend="bm25", rerank_backend="none",
                              calibration_path="")

# Local testing via Ollama — no API key needed, just `ollama pull <model>` and
# `ollama serve` running. litellm routes "ollama_chat/*" to Ollama's chat API
# (better instruction-following than the raw "ollama/*" completion route) and
# reads OLLAMA_API_BASE from the environment if the server isn't on localhost.
# Swap the model name for whatever you've pulled (e.g. "ollama_chat/mistral").
OLLAMA_CONFIG = AdGateConfig(
    intent_backend="llm",
    intent_model="ollama_chat/llama3.1",
    judge_model="ollama_chat/llama3.1",
    rewrite_model="ollama_chat/llama3.1",
    query_rewrite="llm",
    use_judge=True,
)


def load_config(path: str | os.PathLike | None = None) -> AdGateConfig:
    """Load a deployment's config from YAML (default: config.yaml, or the
    ADGATE_CONFIG env var). The YAML holds only the keys that differ from
    the defaults above — copy config.example.yaml to config.yaml and
    uncomment what you change."""
    resolved = Path(path or os.environ.get("ADGATE_CONFIG", "config.yaml"))
    if not resolved.exists():
        raise FileNotFoundError(
            f"{resolved} not found — copy config.example.yaml to {resolved.name} "
            "and edit it (or set ADGATE_CONFIG to a different path)."
        )
    data = yaml.safe_load(resolved.read_text()) or {}
    return AdGateConfig.model_validate(data)
