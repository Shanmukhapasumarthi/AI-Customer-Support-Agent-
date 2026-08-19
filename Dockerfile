# --- Stage 1: build dependencies --------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy requirements FIRST, before the source code. Docker caches each layer, and
# invalidates a layer only when its inputs change. Because source changes far
# more often than dependencies, this ordering means editing a .py file does not
# trigger a full re-install of torch. This one line saves minutes per rebuild.
COPY requirements.txt .

RUN pip install --prefix=/install -r requirements.txt


# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Cache the HuggingFace model inside a named volume so it is downloaded once
    # rather than on every container start.
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/
COPY data/knowledge_base/ ./data/knowledge_base/
COPY main.py ./

# Run as a non-root user. If the process is ever compromised, the attacker gets
# an unprivileged account rather than root inside the container.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/vector_store /app/database /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Docker restarts the container when this fails. It checks the DEPENDENCIES via
# our /health endpoint, not just that the port is open.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["python", "main.py"]
