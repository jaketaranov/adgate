# AdGate

A project exploring ad relevance gating for LLM applications: decide whether to show an ad in a conversation, and show nothing unless the query has commercial intent (Gate A) and the inventory actually contains a relevant ad (Gate B).

Two failure modes motivate the two gates: `"explain what a derivative is"` has no commercial intent, so no ad should show — and `"good indian restaurant near me"` is commercial, but if there are no restaurant ads in the inventory, the closest available match is still wrong. The pipeline is built to abstain at every stage rather than force a match.

## Architecture

A 4-stage cascade where every stage can abstain with a machine-readable reason — cheap stages first, the expensive LLM judgment confined to a bounded borderline band:

```
conversation messages (last N turns)
   │
   ▼
[Stage 0] Context condensation → standalone "intent query" string
   │
   ▼
[Gate A]  Commercial-intent classifier (graded 0–3)
   │         grade/conf below τ_A ──────► ABSTAIN (no_intent)
   ▼
[Gate B1] Hybrid retrieval (BM25 + dense embeddings, RRF-fused), top-k=20
   │         no candidates ────────────► ABSTAIN (no_candidates)
   ▼
[Gate B2] Cross-encoder rerank → calibrated P(relevant)
   │         p < τ_B ──────────────────► ABSTAIN (below_relevance)
   │         p ∈ [τ_B, τ_high) ────────► borderline band ─┐
   │         p ≥ τ_high ───────────────► SHOW             │
   ▼                                                      ▼
[Stage C] LLM-as-judge (borderline band only)
             "no" ─────────────────────► ABSTAIN (judge_rejected)
             "yes" ────────────────────► SHOW top ad (JSON) — else HTTP 204
```

Everything runs on off-the-shelf models (no custom training): `sentence-transformers` embeddings + cross-encoder for retrieval/rerank, and any [litellm](https://github.com/BerriAI/litellm)-supported model (Anthropic, OpenAI, local Ollama, …) for the intent classifier, judge, and query rewriter.

## Example decisions

Real pipeline runs against the bundled ESCI product inventory (`aggressive` operating point, LLM gates backed by a small local Ollama model). `—` means the query abstained before reaching that stage; a 204 is the "no ad" response.

| Conversation | Gate A: intent | Gate B1: retrieve | Gate B2: P(relevant) | Stage C: judge | Result |
|---|---|---|---|---|---|
| *"best lightweight running shoes for daily training"* | RESEARCH | 20 candidates | 0.84 | show | **Ad:** Xero Shoes Prio Cross Training Shoe |
| *"my bathroom mirror fogs up every morning"* → … → *"ok, which quiet exhaust fan should I get?"* | RESEARCH | 20 candidates | 0.84 | show | **Ad:** Broan Very Quiet Ceiling Bathroom Exhaust Fan |
| *"write a python function to reverse a string"* | ADJACENT | — | — | — | 204 (`no_intent`) |
| *"good indian restaurant near me"* | RESEARCH | 20 candidates | 0.25 | — | 204 (`below_relevance`) |
| *"can you recommend a whey protein powder?"* → … → *"nevermind, I just ordered one. thanks!"* | ADJACENT | — | — | — | 204 (`no_intent`) |

Each row exercises a different path: a straightforward product-research hit; a multi-turn conversation where the intent only exists in context (Stage 0 condenses the recent turns into one query); a non-commercial query stopped at the intent gate; the pitch's own failure case — a perfectly commercial query correctly abstained because the inventory contains no restaurants; and a completed purchase, where showing an ad would arrive too late.

## Quickstart

### Docker

```bash
docker compose up --build
```

First boot bootstraps everything automatically (idempotent, persisted in the `./data` volume): streams the ad corpus, builds `ads.jsonl` + `queries.jsonl`, fits Gate B2's calibration, then serves. Takes a few minutes once; restarts skip straight to serving.

```bash
curl -s -X POST localhost:8399/v1/ad \
  -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "buy a quiet bathroom exhaust fan"}]}'
```

An ad decision returns JSON (`ad` + per-stage `diagnostics`); an abstention returns **HTTP 204** — the standard ad-network "no fill" signal — with the reason in the `X-Adgate-Abstain-Reason` header.

### Local

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only, skips ~2.5GB CUDA wheels
pip install -e ".[data,service]"
cp config.example.yaml config.yaml            # then edit to taste

python -m adgate.data.download                # ESCI corpus -> data/ads.jsonl + data/queries.jsonl
python -m adgate.calibrate --data-dir data/   # fit Gate B2 thresholds -> data/calibration.json

python -m adgate.demo "buy a quiet bathroom exhaust fan"   # single query, per-gate trace
uvicorn --factory adgate.service:create_app --port 8399    # or serve
```

The demo prints each gate's decision:

```
  query: 'buy a quiet bathroom exhaust fan'
  Gate A   intent grade=RESEARCH conf=0.85
  Gate B1  20 candidates retrieved
  Gate B2  reranker raw=7.04 -> P(relevant)=0.867
  Stage C  judge=show (matches the live need)
SHOW: [Broan-NuTone] Broan Very Quiet Ceiling Bathroom Exhaust Fan ...
```

### Local models via Ollama (no API key)

Point the litellm model fields in `config.yaml` at `ollama_chat/<model>` (e.g. `ollama_chat/llama3.1`) with `ollama serve` running — see the comments in [config.example.yaml](config.example.yaml). Note that models below ~3B parameters grade commercial intent unreliably; the `heuristic` intent backend is often a stronger choice than a very small LLM.

## Configuration

All knobs live in `config.yaml` (copy from [config.example.yaml](config.example.yaml); every key is optional and falls back to the defaults in `adgate/config.py`). The highlights:

| Knob | Values | What it controls |
|---|---|---|
| `operating_point` | `conservative` / `balanced` / `aggressive` | The precision↔coverage tradeoff — a named (τ_A, τ_B, τ_high) threshold triple applied across the whole cascade |
| `intent_backend` | `heuristic` / `llm` | Gate A: offline pattern rules vs few-shot LLM classification |
| `retrieval_backend` | `bm25` / `dense` / `hybrid` | Gate B1: lexical-only (offline), embeddings, or both fused with reciprocal-rank fusion |
| `rerank_backend` | `cross-encoder` / `none` | Gate B2: joint query+ad scoring vs a squashed retrieval score |
| `use_judge` | `true` / `false` | Stage C: LLM sanity check on borderline scores |
| `query_rewrite` | `last_turns` / `llm` | Stage 0: concatenate recent turns vs LLM standalone-query rewriting for multi-turn intent |
| `intent_model` / `judge_model` / `rewrite_model` | any litellm model id | Which LLM backs each LLM-powered stage |

The fully-offline configuration (`intent_backend: heuristic`, `retrieval_backend: bm25`, `rerank_backend: none`, `use_judge: false`) runs with no model downloads and no API keys — it's what the unit tests use.

**Calibration is a procedure, not a constant.** `python -m adgate.calibrate` fits an isotonic mapping from raw reranker scores to P(relevant) and selects each operating point's τ_B as the smallest threshold whose Wilson 95% lower confidence bound clears that point's precision target. Rerun it whenever the reranker, retrieval backend, or ad inventory changes.

## Data

`python -m adgate.data.download` streams the [Amazon ESCI "Shopping Queries" dataset](https://github.com/amazon-science/esci-data) (Apache-2.0) — real products become the ad inventory, and its human-graded query↔product relevance judgments (Exact/Substitute/Complement/Irrelevant) become the labels used for calibration. One command produces both `ads.jsonl` and `queries.jsonl`; no LLM-generated data is involved.

## Project layout

```
src/adgate/
  schemas.py        Data contracts (Ad, Message, GateDecision, ...)
  config.py         Pydantic config + YAML loading, operating points
  pipeline.py       The 4-stage cascade orchestrator (AdGate.decide)
  prompts.py        All LLM prompts in one place
  stages/           condense (Stage 0), intent (A), retrieve (B1), rerank (B2), judge (C)
  calibration.py    Isotonic fit, Wilson-bound threshold selection, conformal alternative
  calibrate.py      `python -m adgate.calibrate` CLI
  data/download.py  `python -m adgate.data.download` CLI
  service.py        FastAPI shell: POST /v1/ad -> ad JSON or 204
```

## Development

```bash
pip install -e ".[dev]"
./run_tests.sh        # downloads data on first run, then pytest
```
