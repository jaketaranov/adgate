"""Fit Gate B2's calibration on the calibration split.

    python -m adgate.calibrate --data-dir data/

Flow:
  1. For every calibration-split query: retrieve top-k, rerank the head,
     record (raw score of the winning candidate, was it actually relevant?).
  2. fit_isotonic: raw score -> P(relevant).
  3. select_threshold_wilson per operating point's precision target.
  4. conformal_risk_control as the distribution-free alternative (reported).
  5. Save everything to data/calibration.json — the pipeline picks it up
     automatically and stops using its eyeballed fallback thresholds.

Rerun this whenever the reranker, retrieval backend, or ad inventory
changes — calibration is a procedure, not a constant.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from .calibration import (
    conformal_risk_control,
    expected_calibration_error,
    fit_isotonic,
    select_threshold_wilson,
)
from .config import PRECISION_TARGETS, AdGateConfig
from .schemas import Ad, QueryRecord
from .stages.condense import condense_last_turns
from .stages.rerank import CrossEncoderReranker
from .stages.retrieve import AdIndex


def load_split(
    queries_path: Path, split: Literal["calibration", "dev", "test"]
) -> list[QueryRecord]:
    records = []
    with open(queries_path) as f:
        for line in f:
            r = QueryRecord.model_validate_json(line)
            if r.split == split:
                records.append(r)
    return records


def collect_scores(
    records: list[QueryRecord], index: AdIndex, reranker: CrossEncoderReranker,
    rerank_top_n: int,
) -> tuple[list[float], list[bool], list[bool]]:
    """One (raw winning score, top-ad-was-relevant, top-ad-was-judged) sample
    per query that produced candidates. Gate A is deliberately NOT applied
    here: B2's calibration must cover everything that could reach it,
    including commercial queries with no relevant inventory (stratum 2).

    The judged mask matters: ESCI judgments only cover ~8 ads per query, so
    an unjudged top ad has UNKNOWN relevance, not zero. Calibrating with
    unjudged-counted-as-irrelevant makes every precision target unattainable
    (observed empirically: base rate drops to 0.23). Standard pooled-relevance
    practice: fit and threshold on judged pairs only.

    IMPROVEMENT: route a sample of unjudged (query, top-ad) pairs to an
    LLM judge + human review to close the coverage gap properly.
    """
    raws: list[float] = []
    labels: list[bool] = []
    judged: list[bool] = []
    for i, r in enumerate(records):
        query = condense_last_turns(r.messages)
        candidates = index.search(query)
        if not candidates:
            continue
        head = candidates[:rerank_top_n]
        scores = reranker.score(query, [c.ad for c in head])
        best_i = max(range(len(scores)), key=scores.__getitem__)
        top_id = head[best_i].ad.ad_id
        raws.append(scores[best_i])
        labels.append(r.relevant_ad_ids.get(top_id, 0) >= 1)
        judged.append(top_id in r.relevant_ad_ids)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(records)} queries scored")
    return raws, labels, judged


def main() -> None:
    parser = argparse.ArgumentParser(prog="adgate.calibrate", description=__doc__)
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--conformal-alpha", type=float, default=0.05)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cfg = AdGateConfig(ads_path=str(data_dir / "ads.jsonl"),
                       queries_path=str(data_dir / "queries.jsonl"),
                       calibration_path=str(data_dir / "calibration.json"))

    print("Loading data ...")
    ads = [Ad.model_validate_json(line) for line in open(cfg.ads_path)]
    records = load_split(Path(cfg.queries_path), "calibration")
    print(f"  {len(ads)} ads, {len(records)} calibration queries")

    index = AdIndex(cfg)
    index.build(ads)
    reranker = CrossEncoderReranker(cfg)

    print("Scoring calibration split (retrieve + rerank) ...")
    raws, labels, judged = collect_scores(records, index, reranker, cfg.rerank_top_n)
    # Fit on judged pairs only — unjudged top ads have unknown relevance, not
    # zero (see collect_scores docstring).
    j_raws = [r for r, j in zip(raws, judged) if j]
    j_labels = [l for l, j in zip(labels, judged) if j]
    n, base = len(j_raws), sum(j_labels)
    print(f"  {len(raws)} samples, {n} with judged top ad "
          f"({len(raws) - n} unjudged dropped), base relevance rate {base / n:.3f}")

    print("Fitting isotonic calibration ...")
    model = fit_isotonic(j_raws, j_labels)
    calibrated = [model.calibrate(r) for r in j_raws]
    labels = j_labels
    ece = expected_calibration_error(calibrated, labels)

    print("Selecting thresholds (Wilson lower bound) ...")
    for op_name, target in PRECISION_TARGETS.items():
        tau = select_threshold_wilson(calibrated, labels, target_precision=target)
        if tau is None:
            print(f"  {op_name}: target precision {target:.2f} UNATTAINABLE — no tau stored")
            continue
        shown = [(c, l) for c, l in zip(calibrated, labels) if c >= tau]
        prec = sum(1 for _, l in shown if l) / len(shown)
        cov = len(shown) / n
        model.taus[op_name] = tau
        print(f"  {op_name}: tau_b={tau:.3f} -> precision {prec:.3f}, "
              f"share-shown {cov:.3f} (target {target:.2f})")

    conf_tau = conformal_risk_control(calibrated, labels, alpha=args.conformal_alpha)
    if conf_tau is not None:
        print(f"  conformal (alpha={args.conformal_alpha}): tau_b={conf_tau:.3f} "
              f"[distribution-free P(show | irrelevant) <= {args.conformal_alpha}]")
        model.meta["conformal_tau"] = conf_tau
        model.meta["conformal_alpha"] = args.conformal_alpha

    model.meta.update({"ece": ece, "n": float(n), "base_rate": base / n})
    model.save(cfg.calibration_path)
    print(f"\nECE after calibration: {ece:.4f} (self-evaluated — optimistic; "
          f"judge on dev split)")
    print(f"Saved -> {cfg.calibration_path}")
    print("The pipeline now uses these thresholds automatically.")


if __name__ == "__main__":
    main()
