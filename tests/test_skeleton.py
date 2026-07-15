"""Skeleton coherence test: every module imports without model downloads,
schemas instantiate, and the pipeline wires stage interfaces."""



def test_all_modules_import():
    import adgate  # noqa: F401
    import adgate.calibration  # noqa: F401
    import adgate.config  # noqa: F401
    import adgate.pipeline  # noqa: F401
    import adgate.schemas  # noqa: F401
    import adgate.stages.condense  # noqa: F401
    import adgate.stages.intent  # noqa: F401
    import adgate.stages.judge  # noqa: F401
    import adgate.stages.rerank  # noqa: F401
    import adgate.stages.retrieve  # noqa: F401


def test_schemas_instantiate():
    from adgate.schemas import (
        AbstainReason,
        Ad,
        GateDecision,
        IntentGrade,
        Message,
        QueryRecord,
        StageScores,
    )

    ad = Ad(
        ad_id="ad-001",
        brand="Acme Coffee",
        title="Small-batch beans, delivered fresh",
        body="Single-origin coffee roasted to order and shipped within 48 hours.",
        cta="Learn more",
        click_url="https://example.com/acme",
        category="food-beverage",
    )
    decision = GateDecision(show=True, ad=ad, scores=StageScores(n_candidates=5))
    assert decision.ad.brand == "Acme Coffee"

    abstain = GateDecision(show=False, abstain_reason=AbstainReason.NO_INTENT)
    assert abstain.abstain_reason == "no_intent"

    record = QueryRecord(
        query_id="q-001",
        messages=[Message(role="user", content="what is a derivative")],
        stratum=3,
        intent_grade=IntentGrade.NONE,
        should_show=False,
    )
    assert record.intent_grade == 0
    # round-trips through JSON (harness serializes decisions)
    assert GateDecision.model_validate_json(decision.model_dump_json()) == decision


def test_operating_points_defined():
    from adgate.config import DEFAULT_CONFIG, OPERATING_POINTS

    assert set(OPERATING_POINTS) == {"conservative", "balanced", "aggressive"}
    assert DEFAULT_CONFIG.operating_point in OPERATING_POINTS
    for op in OPERATING_POINTS.values():
        assert op.tau_b <= op.tau_high


def _tiny_index():
    from adgate.config import OFFLINE_CONFIG as CFG
    from adgate.schemas import Ad
    from adgate.stages.retrieve import AdIndex

    # Note: BM25's IDF is ~0 on a 2-doc corpus (term in 1 of 2 docs ->
    # ln(1.5/1.5)=0), so the fixture needs a few filler ads to be realistic.
    ads = [
        Ad(ad_id="a1", brand="Taj Spice", title="Indian restaurant downtown",
           body="Authentic indian curry and tandoori, dine in or delivery.",
           category="restaurants"),
        Ad(ad_id="a2", brand="PedalPro", title="Carbon road bike sale",
           body="Lightweight carbon frame road bikes for racing.",
           category="cycling"),
        Ad(ad_id="a3", brand="SleepWell", title="Memory foam mattress",
           body="Cooling gel memory foam mattress with free shipping.",
           category="home"),
        Ad(ad_id="a4", brand="CodeCamp", title="Learn programming online",
           body="Interactive coding bootcamp with mentor support.",
           category="education"),
        Ad(ad_id="a5", brand="FitFuel", title="Whey protein powder",
           body="Grass fed whey protein for muscle recovery.",
           category="fitness"),
        Ad(ad_id="a6", brand="SkyTours", title="Hot air balloon rides",
           body="Sunrise hot air balloon tours over wine country.",
           category="travel"),
    ]
    index = AdIndex(CFG)
    index.build(ads)
    return index


def test_pipeline_end_to_end():
    from adgate.config import OFFLINE_CONFIG
    from adgate.pipeline import AdGate
    from adgate.schemas import AbstainReason, Message

    gate = AdGate(OFFLINE_CONFIG, _tiny_index())

    # Commercial query with a matching ad -> shown
    d = gate.decide([Message(role="user", content="indian restaurant curry delivery near me")])
    assert d.show and d.ad.ad_id == "a1"

    # Non-commercial query -> Gate A abstains
    d = gate.decide([Message(role="user", content="what is a derivative")])
    assert not d.show and d.abstain_reason == AbstainReason.NO_INTENT

    # Empty conversation -> abstain, never crash
    d = gate.decide([])
    assert not d.show
