.PHONY: test check lint format serve bench clean

test:
	uv run pytest -v

check: lint format typecheck

lint:
	uv run ruff check src tests benchmarks

format:
	uv run ruff format --check src tests benchmarks

typecheck:
	uv run mypy --strict src/nanoserve/core src/nanoserve/sampling src/nanoserve/config.py

serve:
	uv run nanoserve --model toy --device cpu

bench:
	uv run python benchmarks/harness.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
