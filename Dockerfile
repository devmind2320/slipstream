# Slipstream gateway + dashboard.
#
# The image installs only requirements.txt — the pure-Python core. No model
# weights are baked in and none are fetched at runtime, so a container started
# from this image satisfies INV-5 (offline, no API key) exactly as a clean clone
# does. The optional heavier backends are deliberately NOT installed here; see
# requirements-optional.txt if you want them.
#
#   docker build -t slipstream .
#   docker run --rm -p 8000:8000 slipstream
#   open http://127.0.0.1:8000/

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so a source-only change does not invalidate the wheel layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY slipstream/ ./slipstream/
COPY README.md ASSUMPTIONS.md ./

# The ledger and any generated data are written under /app/data. Mount a volume
# here to keep the hash chain across container restarts; without one the chain
# starts fresh each run, which is fine for a demo and wrong for an audit.
ENV SLIPSTREAM_DATA_DIR=/app/data
RUN mkdir -p /app/data && \
    useradd --create-home --uid 10001 slipstream && \
    chown -R slipstream:slipstream /app
USER slipstream

EXPOSE 8000

# Hits the gateway's own health endpoint, which reports the provider, the live
# detector set and the registry fingerprint rather than merely that a port is open.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "slipstream.gateway.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
