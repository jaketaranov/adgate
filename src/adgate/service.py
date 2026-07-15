"""FastAPI serving shell.

    pip install 'adgate[service]'
    uvicorn --factory adgate.service:create_app --port 8399

POST /v1/ad  {messages: [{role, content}], session_id?}
  -> 200 ad JSON (+ decision diagnostics)
  -> 204 with X-Adgate-Abstain-Reason header when AdGate abstains
GET /healthz -> {ok, ads, operating_point}

The 204-on-no-ad shape matches common ad-network conventions. Adapters for
specific ad networks (field renames, upstream inventory proxies) belong here
at the service boundary — the core pipeline stays network-agnostic.

IMPROVEMENT: per-deployment operating-point config, GateDecision logging to
durable storage (future training data + drift monitoring), and caching of
Gate A verdicts for repeated queries.
"""

from __future__ import annotations

from pathlib import Path

try:
    from fastapi import FastAPI, Response
    from pydantic import BaseModel
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "service extras not installed — pip install 'adgate[service]'"
    ) from _e

from .config import AdGateConfig, load_config
from .pipeline import AdGate
from .schemas import Ad, Message
from .stages.retrieve import AdIndex


class AdRequest(BaseModel):
    messages: list[Message]
    session_id: str = ""


def create_app(config: AdGateConfig | None = None, gate: AdGate | None = None) -> FastAPI:
    """App factory. Pass a prebuilt `gate` for tests; otherwise ads are
    loaded and the index built once at startup from config paths."""
    cfg = config or load_config()

    if gate is None:
        ads_path = Path(cfg.ads_path)
        if not ads_path.exists():
            raise FileNotFoundError(
                f"{ads_path} not found — run: python -m adgate.data.download"
            )
        ads = [Ad.model_validate_json(line) for line in open(ads_path)]
        index = AdIndex(cfg)
        index.build(ads)
        gate = AdGate(cfg, index)

    app = FastAPI(title="adgate", version="0.1.0")

    @app.post("/v1/ad")
    def serve_ad(req: AdRequest, response: Response):
        decision = gate.decide(req.messages)
        if not decision.show:
            response.status_code = 204
            response.headers["X-Adgate-Abstain-Reason"] = (
                decision.abstain_reason.value if decision.abstain_reason else "unknown"
            )
            return Response(status_code=204, headers=dict(response.headers))
        return {
            "ad": decision.ad.model_dump(),
            "diagnostics": decision.scores.model_dump(),
        }

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "ads": len(gate.index.ads),
                "operating_point": gate.op.name}

    return app
