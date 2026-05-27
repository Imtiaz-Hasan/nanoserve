# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] — 2026-05-27

### Added
- Initial vertical slice: weight loading, Llama forward pass, greedy generation
- OpenAI-compatible API (`/v1/completions`, `/v1/chat/completions`)
- Naive contiguous KV cache (to be replaced by paged cache in Week 2)
- CPU-testable toy model (2-layer, 4-head, 64-dim)
- Prometheus metrics endpoint (`/metrics`)
- Structured logging with `structlog`
- CI pipeline: ruff, mypy --strict, pytest
- Dockerfile
