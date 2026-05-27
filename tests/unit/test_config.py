"""Unit tests for configuration and KV memory arithmetic."""

from nanoserve.config import CacheConfig, EngineConfig, ModelConfig


def test_kv_bytes_per_token_fp16() -> None:
    """Verify KV bytes/token formula: 2 × layers × kv_heads × head_dim × dtype_bytes."""
    model = ModelConfig(
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
        dtype="float16",
    )
    cache = CacheConfig(cache_dtype="auto")

    # 2 × 28 × 4 × 128 × 2 = 57,344
    assert cache.kv_bytes_per_token(model) == 57_344


def test_kv_bytes_per_token_fp32() -> None:
    """Verify KV bytes/token with float32 dtype."""
    model = ModelConfig(
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        dtype="float32",
    )
    cache = CacheConfig(cache_dtype="auto")

    # 2 × 2 × 4 × 64 × 4 = 4,096
    assert cache.kv_bytes_per_token(model) == 4_096


def test_kv_bytes_per_block() -> None:
    """Block bytes = token bytes × block_size."""
    model = ModelConfig(num_layers=2, num_kv_heads=4, head_dim=64, dtype="float32")
    cache = CacheConfig(block_size=16, cache_dtype="auto")

    assert cache.kv_bytes_per_block(model) == 4_096 * 16


def test_estimate_num_blocks() -> None:
    """Estimate how many blocks fit in a given memory budget."""
    model = ModelConfig(num_layers=2, num_kv_heads=4, head_dim=64, dtype="float32")
    cache = CacheConfig(block_size=16, cache_dtype="auto")

    bpb = cache.kv_bytes_per_block(model)
    budget = bpb * 100  # exactly 100 blocks worth
    assert cache.estimate_num_blocks(model, budget) == 100


def test_engine_config_log_memory_budget() -> None:
    """Verify the human-readable memory summary contains expected values."""
    config = EngineConfig(
        model=ModelConfig(num_layers=2, num_kv_heads=4, head_dim=64),
        cache=CacheConfig(num_gpu_blocks=256, block_size=16),
    )
    summary = config.log_memory_budget()
    assert "KV bytes/token:" in summary
    assert "GPU blocks: 256" in summary
    assert "Block size: 16 tokens" in summary


def test_model_config_is_toy() -> None:
    """Verify toy model detection."""
    assert ModelConfig(model_name_or_path="toy").is_toy
    assert not ModelConfig(model_name_or_path="Qwen/Qwen2.5-0.5B").is_toy


def test_default_configs_are_valid() -> None:
    """Ensure default configs can be constructed without errors."""
    config = EngineConfig()
    assert config.device == "cpu"
    assert config.model.vocab_size == 256
    assert config.cache.block_size == 16
    assert config.scheduler.max_num_seqs == 128
