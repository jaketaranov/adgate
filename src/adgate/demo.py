"""Try single queries against the pipeline and watch the gate decisions.

    python -m adgate.demo "best bathroom exhaust fan"
    python -m adgate.demo            # interactive loop
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import load_config
from .pipeline import AdGate
from .schemas import Ad, Message
from .stages.retrieve import AdIndex


def show(gate: AdGate, query: str) -> None:
    d = gate.decide([Message(role="user", content=query)])
    s = d.scores
    print(f"\n  query: {query!r}")
    print(f"  Gate A   intent grade={s.intent_grade.name if s.intent_grade is not None else '-'}"
          f" conf={s.intent_confidence}")
    if s.n_candidates:
        print(f"  Gate B1  {s.n_candidates} candidates retrieved")
    if s.rerank_raw_logit is not None:
        print(f"  Gate B2  reranker raw={s.rerank_raw_logit:.2f} -> P(relevant)={s.rerank_calibrated:.3f}")
    elif s.rerank_calibrated is not None:
        print(f"  Gate B2  pseudo_p={s.rerank_calibrated:.3f} (no reranker)")
    if s.judge_verdict is not None:
        print(f"  Stage C  judge={'show' if s.judge_verdict else 'reject'} ({s.judge_reason})")
    if d.show:
        print(f"SHOW: [{d.ad.brand}] {d.ad.title}")
    else:
        print(f"NO AD (abstain: {d.abstain_reason.value})")


def main() -> None:
    data = Path("data/ads.jsonl")
    if not data.exists():
        sys.exit("data/ads.jsonl not found — run: python -m adgate.data.download")
    cfg = load_config()
    print(f"Loading {data} ...")
    ads = [Ad.model_validate_json(line) for line in open(data)]
    index = AdIndex(cfg)
    index.build(ads)
    gate = AdGate(cfg, index)
    print(f"{len(ads)} ads indexed. Operating point: {cfg.operating_point}")

    if len(sys.argv) > 1:
        show(gate, " ".join(sys.argv[1:]))
        return
    print("Type a query (empty line to quit):")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(gate, q)


if __name__ == "__main__":
    main()
