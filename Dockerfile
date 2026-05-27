FROM python:3.13-slim AS builder

WORKDIR /build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv pip install --system --no-cache .

FROM python:3.13-slim AS runner

WORKDIR /app

RUN groupadd -r nanoserve && useradd -r -g nanoserve nanoserve

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/src /app/src

USER nanoserve

EXPOSE 8000

ENTRYPOINT ["python", "-m", "nanoserve.server.app", "--host", "0.0.0.0", "--port", "8000", "--model", "toy", "--device", "cpu"]
