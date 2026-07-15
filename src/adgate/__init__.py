"""AdGate — two-gate ad suitability & relevance middleware.

Layout:

    schemas.py        Data contracts
    config.py         Model names, thresholds, operating points
    pipeline.py       The 4-stage cascade orchestrator
    stages/           Stage 0 + Gates A, B1, B2 + judge
    calibration.py    Isotonic / Wilson / conformal threshold selection
    data/             ESCI download -> ads.jsonl + queries.jsonl
    service.py        FastAPI serving shell: ad JSON or 204
"""

__version__ = "0.1.0"
