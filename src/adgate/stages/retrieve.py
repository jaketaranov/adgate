"""Gate B1 — retrieval over the ad index.

Three backends, selected by config.retrieval_backend:

  "bm25"    lexical only — pure Python, no model downloads (offline/tests)
  "dense"   sentence-transformers embeddings, exact cosine search
  "hybrid"  both, fused with reciprocal-rank fusion (the default: dense
            handles paraphrases, BM25 handles brand names / exact terms)

B1's job is recall; the relevance decision belongs to B2.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from ..config import AdGateConfig
from ..schemas import Ad


class Candidate(BaseModel):
    ad: Ad
    # Meaning depends on backend: BM25 score, cosine similarity, or RRF score.
    # Never treated as a probability — that's Gate B2's job.
    score: float
    rank: int


def ad_to_document(ad: Ad) -> str:
    """Compose the searchable text for one ad."""
    keywords = " ".join(ad.keywords)
    return f"{ad.brand} — {ad.title}. {ad.body} Category: {ad.category}. Keywords: {keywords}"


def _tokenize(text: str) -> list[str]:
    # IMPROVEMENT: plain lowercase word-split; a stemmer or subword tokenizer
    # would help BM25 ("fans" vs "fan" currently don't match).
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """RRF: score(d) = sum over rankings of 1 / (k + rank(d)).
    Returns (id, fused_score) sorted best-first. Ids missing from a ranking
    simply contribute nothing from it."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])


class AdIndex:
    """Retrieval index over the ad database (BM25 + optional dense)."""

    def __init__(self, config: AdGateConfig) -> None:
        self.config = config
        self.ads: list[Ad] = []
        self._by_id: dict[str, int] = {}
        self._bm25 = None
        self._embeddings = None   # np.ndarray [n_ads, dim], L2-normalized
        self._encoder = None

    # build

    def build(self, ads: list[Ad]) -> None:
        from rank_bm25 import BM25Okapi  # cheap; always built (hybrid + strawman)

        self.ads = ads
        self._by_id = {ad.ad_id: i for i, ad in enumerate(ads)}
        docs = [ad_to_document(ad) for ad in ads]
        self._bm25 = BM25Okapi([_tokenize(d) for d in docs])

        if self.config.retrieval_backend in ("dense", "hybrid"):
            self._build_dense(docs)

    def _build_dense(self, docs: list[str]) -> None:
        import hashlib

        import numpy as np

        # Disk cache keyed on (model, ad ids): embedding 5k ads takes ~5 min
        # on CPU, so re-encoding on every process start is unacceptable.
        cache_dir = Path(self.config.ads_path).parent / ".cache"
        key = hashlib.md5(
            (self.config.embedding_model + "|" + "|".join(a.ad_id for a in self.ads)).encode()
        ).hexdigest()[:16]
        cache = cache_dir / f"emb-{key}.npy"
        if cache.exists():
            self._embeddings = np.load(cache)
            return
        encoder = self._get_encoder()
        # Passages get NO prefix; only queries do (asymmetric embeddings).
        self._embeddings = encoder.encode(
            docs, normalize_embeddings=True, batch_size=64,
            convert_to_numpy=True, show_progress_bar=True,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, self._embeddings)

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # deferred: heavy

            self._encoder = SentenceTransformer(self.config.embedding_model)
        return self._encoder

    # search

    def search(self, query: str, k: int | None = None) -> list[Candidate]:
        if self._bm25 is None:
            raise RuntimeError("AdIndex.build() must be called first")
        k = k or self.config.top_k
        backend = self.config.retrieval_backend
        if backend == "bm25":
            return self._search_bm25(query, k)
        if backend == "dense":
            return self._search_dense(query, k)
        if backend == "hybrid":
            return self._search_hybrid(query, k)
        raise ValueError(f"unknown retrieval_backend: {backend}")

    def _search_bm25(self, query: str, k: int) -> list[Candidate]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            Candidate(ad=self.ads[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(ranked)
            if scores[i] > self.config.score_floor
        ]

    def _search_dense(self, query: str, k: int) -> list[Candidate]:
        # Query side DOES get the bge instruction prefix (asymmetry gotcha).
        qvec = self._get_encoder().encode(
            self.config.query_prefix + query,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        # Exact search: 5k x 384 dot product is microseconds; IMPROVEMENT:
        # FAISS IndexFlatIP / an ANN index at real inventory scale.
        cos = self._embeddings @ qvec
        ranked = cos.argsort()[::-1][:k]
        return [
            Candidate(ad=self.ads[i], score=float(cos[i]), rank=rank)
            for rank, i in enumerate(ranked)
            if cos[i] >= self.config.cosine_floor
        ]

    def _search_hybrid(self, query: str, k: int) -> list[Candidate]:
        # Fetch a deeper pool from each retriever, fuse, cut to k.
        pool = k * 2
        bm25 = self._search_bm25(query, pool)
        dense = self._search_dense(query, pool)
        fused = reciprocal_rank_fusion(
            [[c.ad.ad_id for c in bm25], [c.ad.ad_id for c in dense]],
            k=self.config.rrf_k,
        )[:k]
        return [
            Candidate(ad=self.ads[self._by_id[ad_id]], score=score, rank=rank)
            for rank, (ad_id, score) in enumerate(fused)
        ]
