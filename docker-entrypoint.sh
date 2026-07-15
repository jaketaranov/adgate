#!/usr/bin/env bash
# Idempotent data bootstrap: download + calibrate only run once. Mount
# ./data as a volume (see docker-compose.yml) so restarts skip both and
# go straight to serving.
set -euo pipefail
cd /app

if [[ ! -f config.yaml ]]; then
  echo "==> no config.yaml — copying config.example.yaml as a starting default"
  cp config.example.yaml config.yaml
fi

if [[ ! -f data/ads.jsonl || ! -f data/queries.jsonl ]]; then
  echo "==> data missing — downloading ESCI (streams, a few minutes)"
  python -m adgate.data.download --out data/
else
  echo "==> data present ($(wc -l < data/ads.jsonl) ads, $(wc -l < data/queries.jsonl) queries)"
fi

if [[ ! -f data/calibration.json ]]; then
  echo "==> no calibration file — fitting Gate B2 thresholds"
  python -m adgate.calibrate --data-dir data/
else
  echo "==> calibration.json present, skipping"
fi

exec "$@"
