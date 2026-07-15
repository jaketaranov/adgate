"""Hand-computed cases for the calibration statistics and RRF fusion."""

import math

from adgate.calibration import (
    CalibrationModel,
    conformal_risk_control,
    expected_calibration_error,
    fit_isotonic,
    select_threshold_wilson,
)
from adgate.stages.retrieve import reciprocal_rank_fusion


def test_isotonic_is_monotone_and_interpolates():
    # scores cleanly separate: low scores irrelevant, high scores relevant
    raws = [0.0, 0.1, 0.2, 0.3, 2.0, 2.1, 2.2, 2.3]
    labels = [False, False, False, False, True, True, True, True]
    model = fit_isotonic(raws, labels)

    assert model.calibrate(0.1) < 0.5 < model.calibrate(2.1)
    # monotone over a sweep
    xs = [i / 10 for i in range(-5, 30)]
    ys = [model.calibrate(x) for x in xs]
    assert all(a <= b + 1e-9 for a, b in zip(ys, ys[1:]))
    # clipped at the ends
    assert model.calibrate(-100) == ys[0]
    assert model.calibrate(+100) == ys[-1]


def test_calibration_model_roundtrip(tmp_path):
    m = CalibrationModel(x=[0.0, 1.0], y=[0.2, 0.8], taus={"balanced": 0.6})
    p = tmp_path / "cal.json"
    m.save(p)
    m2 = CalibrationModel.load(p)
    assert m2 == m
    assert abs(m2.calibrate(0.5) - 0.5) < 1e-9  # linear midpoint


def test_wilson_threshold_hand_computed():
    # 20 samples: scores >= 0.9 are 10/10 relevant; below, 0/10 relevant.
    scores = [0.1] * 10 + [0.9] * 10
    labels = [False] * 10 + [True] * 10
    # Wilson 95% lower bound for 10/10 is ~0.722 -> target 0.7 attainable at 0.9
    tau = select_threshold_wilson(scores, labels, target_precision=0.70)
    assert tau == 0.9
    # target 0.95 is NOT attainable with only 10 shown samples (LB ~0.722)
    assert select_threshold_wilson(scores, labels, target_precision=0.95) is None


def test_conformal_hand_computed():
    # 9 irrelevant scores 0.1..0.9; alpha=0.2 -> k=ceil(10*0.8)=8 -> tau=0.8
    scores = [i / 10 for i in range(1, 10)] + [0.95]
    labels = [False] * 9 + [True]
    tau = conformal_risk_control(scores, labels, alpha=0.2)
    assert abs(tau - 0.8) < 1e-9
    # alpha too strict for n=9 negatives -> no guarantee possible
    assert conformal_risk_control(scores, labels, alpha=0.01) is None


def test_ece_perfect_and_awful():
    # perfectly calibrated at 0.5: half the 0.5-confidence predictions are true
    assert expected_calibration_error([0.5, 0.5], [True, False]) < 0.01
    # maximally miscalibrated: confident 0.95 but always wrong
    assert expected_calibration_error([0.95] * 4, [False] * 4) > 0.9


def test_rrf_hand_computed():
    # doc "a" is rank 0 in both lists; "b" rank 1 in one, absent in other
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "a"
    scores = dict(fused)
    assert abs(scores["a"] - 2 / 61) < 1e-9
    assert abs(scores["b"] - 1 / 62) < 1e-9
    assert set(ids) == {"a", "b", "c"}
