"""Service boundary test: ad JSON on show, HTTP 204 + reason header on abstain."""

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from adgate.config import OFFLINE_CONFIG  # noqa: E402
from adgate.pipeline import AdGate  # noqa: E402
from adgate.service import create_app  # noqa: E402
from test_skeleton import _tiny_index  # noqa: E402


@pytest.fixture(scope="module")
def client():
    gate = AdGate(OFFLINE_CONFIG, _tiny_index())
    return TestClient(create_app(config=OFFLINE_CONFIG, gate=gate))


def test_serve_ad_success(client):
    r = client.post("/v1/ad", json={
        "messages": [{"role": "user", "content": "indian restaurant curry delivery near me"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ad"]["ad_id"] == "a1"
    assert body["diagnostics"]["intent_grade"] == 3  # TRANSACTIONAL


def test_serve_204_on_noncommercial(client):
    r = client.post("/v1/ad", json={
        "messages": [{"role": "user", "content": "what is a derivative"}],
    })
    assert r.status_code == 204
    assert r.headers["x-adgate-abstain-reason"] == "no_intent"


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True and r.json()["ads"] == 6
