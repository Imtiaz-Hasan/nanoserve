"""Prometheus metrics collectors for the serving engine."""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest

router = APIRouter()

# Counters
REQUESTS_TOTAL = Counter(
    "nanoserve_requests_total",
    "Total number of inference requests",
    ["method", "endpoint", "status"],
)
TOKENS_GENERATED_TOTAL = Counter(
    "nanoserve_tokens_generated_total",
    "Total tokens generated across all requests",
)

# Histograms
TTFT_SECONDS = Histogram(
    "nanoserve_ttft_seconds",
    "Time to first token in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
TPOT_SECONDS = Histogram(
    "nanoserve_tpot_seconds",
    "Time per output token in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)
E2E_LATENCY_SECONDS = Histogram(
    "nanoserve_e2e_latency_seconds",
    "End-to-end request latency in seconds",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# Gauges
RUNNING_SEQUENCES = Gauge(
    "nanoserve_running_sequences",
    "Number of currently running sequences",
)
WAITING_SEQUENCES = Gauge(
    "nanoserve_waiting_sequences",
    "Number of sequences waiting to be scheduled",
)
KV_CACHE_USAGE_RATIO = Gauge(
    "nanoserve_kv_cache_usage_ratio",
    "Fraction of KV cache blocks currently in use",
)


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus metrics exposition endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
