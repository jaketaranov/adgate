FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first — sentence-transformers otherwise pulls the ~2.5GB
# CUDA build (see pyproject.toml's note). No GPU in this image.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir ".[data,service]"

COPY config.example.yaml ./

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8399
VOLUME ["/app/data"]

# First boot streams ESCI + fits calibration (a few minutes); persist
# /app/data as a volume so restarts skip straight to serving.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD curl -f http://localhost:8399/healthz || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "--factory", "adgate.service:create_app", "--host", "0.0.0.0", "--port", "8399"]
