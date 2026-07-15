"""Threshold calibration — the statistics.

All fitting happens on the CALIBRATION split only; the test split is
untouched until final numbers. Thresholds are procedures to rerun on real
traffic, never fixed constants.

The flow (driven by `python -m adgate.calibrate`):
  raw reranker scores + binary relevance labels
    -> fit_isotonic()             raw score -> P(relevant)
    -> select_threshold_wilson()  tau_b per precision target
    -> conformal_risk_control()   alternative tau with a distribution-free guarantee
    -> CalibrationModel saved to data/calibration.json, loaded by the pipeline
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field


class CalibrationModel(BaseModel):
    """Serialized isotonic mapping + the thresholds selected on top of it."""

    x: list[float]                 # raw-score breakpoints (sorted ascending)
    y: list[float]                 # calibrated P(relevant) at each breakpoint
    taus: dict[str, float] = Field(default_factory=dict)   # operating point -> tau_b
    meta: dict[str, float] = Field(default_factory=dict)   # ece, n, base_rate, ...

    def calibrate(self, raw: float) -> float:
        """Map a raw reranker score to P(relevant) by piecewise-linear
        interpolation over the isotonic breakpoints (clipped at the ends)."""
        xs, ys = self.x, self.y
        if not xs:
            return 0.0
        if raw <= xs[0]:
            return ys[0]
        if raw >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_right(xs, raw) - 1
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        return y0 if x1 == x0 else y0 + (y1 - y0) * (raw - x0) / (x1 - x0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationModel":
        return cls.model_validate_json(Path(path).read_text())


def fit_isotonic(
    raw_scores: Sequence[float], labels: Sequence[bool]
) -> CalibrationModel:
    """Fit isotonic regression mapping raw reranker score -> P(relevant).
    Isotonic = monotone step function; we keep its breakpoints and
    interpolate between them at inference time.

    IMPROVEMENT: with very small calibration sets (<~200), Platt scaling
    (logistic) is more stable than isotonic; not implemented here.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(list(raw_scores), [1.0 if b else 0.0 for b in labels])
    return CalibrationModel(
        x=[float(v) for v in iso.X_thresholds_],
        y=[float(v) for v in iso.y_thresholds_],
    )


def wilson_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion."""
    from statsmodels.stats.proportion import proportion_confint

    lo, _ = proportion_confint(successes, n, alpha=1 - confidence, method="wilson")
    return float(lo)


def select_threshold_wilson(
    calibrated_scores: Sequence[float],
    labels: Sequence[bool],
    target_precision: float = 0.90,
    confidence: float = 0.95,
    min_support: int = 10,
) -> float | None:
    """Smallest tau whose shown-ad precision LOWER Wilson bound clears the
    target. The CI bound, not the point estimate, protects against
    overfitting the threshold to calibration noise.

    Returns None if no threshold with >= min_support shown samples reaches
    the target — a signal the target is unattainable with this scorer.
    """
    pairs = sorted(zip(calibrated_scores, labels), key=lambda p: p[0])
    n_total = len(pairs)
    # Sweep candidate thresholds ascending. Suffix sums make each check O(1).
    suffix_pos = [0] * (n_total + 1)
    for i in range(n_total - 1, -1, -1):
        suffix_pos[i] = suffix_pos[i + 1] + (1 if pairs[i][1] else 0)
    for i in range(n_total):
        if i > 0 and pairs[i][0] == pairs[i - 1][0]:
            continue  # same threshold as previous
        tau = pairs[i][0]
        n_shown = n_total - i
        if n_shown < min_support:
            return None  # thresholds only get less supported from here
        if wilson_lower_bound(suffix_pos[i], n_shown, confidence) >= target_precision:
            return float(tau)
    return None


def conformal_risk_control(
    calibrated_scores: Sequence[float],
    labels: Sequence[bool],
    alpha: float = 0.05,
) -> float | None:
    """Split-conformal threshold with a distribution-free guarantee:
    P(score >= tau | top ad is IRRELEVANT) <= alpha on exchangeable data.

    Mechanics: take the scores of the irrelevant (label=False) calibration
    samples and return their ceil((n+1)(1-alpha))-th order statistic. A fresh
    irrelevant example then exceeds tau with probability <= alpha.
    Reference: Angelopoulos & Bates 2021.
    """
    neg = sorted(s for s, is_rel in zip(calibrated_scores, labels) if not is_rel)
    n = len(neg)
    if n == 0:
        return None
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return None  # not enough negatives for a guarantee at this alpha
    return float(neg[k - 1])


def expected_calibration_error(
    calibrated_scores: Sequence[float], labels: Sequence[bool], n_bins: int = 10
) -> float:
    """ECE: bin predictions, compare mean confidence vs observed accuracy,
    weight by bin size (reliability diagram companion number)."""
    if not calibrated_scores:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for s, l in zip(calibrated_scores, labels):
        idx = min(int(s * n_bins), n_bins - 1)
        bins[idx].append((s, l))
    n = len(calibrated_scores)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(s for s, _ in b) / len(b)
        acc = sum(1 for _, l in b if l) / len(b)
        ece += (len(b) / n) * abs(conf - acc)
    return ece
