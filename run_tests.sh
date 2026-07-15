#!/usr/bin/env bash
# Run the AdGate test suite, downloading/building data first if it's missing.
set -euo pipefail
cd "$(dirname "$0")"

# Use the project venv if present and not already activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for v in venv .venv; do
    if [[ -x "$v/bin/python" ]]; then PY="$v/bin/python"; break; fi
  done
fi
PY="${PY:-python3}"

export PYTHONPATH=src

if [[ ! -f data/ads.jsonl || ! -f data/queries.jsonl ]]; then
  echo "==> data missing — downloading ESCI (this streams, a few minutes)"
  "$PY" -m adgate.data.download --out data/
else
  echo "==> data present ($(wc -l < data/ads.jsonl) ads, $(wc -l < data/queries.jsonl) queries)"
fi

echo "==> running tests"
"$PY" -m pytest tests/ -q
