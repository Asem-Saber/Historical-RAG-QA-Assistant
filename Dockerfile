# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt


FROM python:3.12-slim AS model-fetcher

COPY --from=builder /install /install

ENV PYTHONPATH=/install/lib/python3.12/site-packages \
    HF_HOME=/root/.cache/huggingface

RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')"


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/install/lib/python3.12/site-packages \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY --from=builder /install /install
COPY --from=model-fetcher /root/.cache/huggingface /app/.cache/huggingface

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser && \
    mkdir -p /app/data /app/logs /app/.cache/huggingface && \
    chown -R appuser:appuser /app/data /app/logs /app/.cache/huggingface

COPY --chmod=755 --chown=appuser:appuser docker-entrypoint.sh /app/docker-entrypoint.sh
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser frontend/ frontend/

USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["/install/bin/uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
