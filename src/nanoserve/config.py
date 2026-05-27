"""Engine, model, and scheduler configuration with KV memory arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Model identity and dtype configuration."""

    model_name_or_path: str = "toy"
    dtype: str = "float32"
    max_model_len: int = 2048
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 4
    head_dim: int = 64
    hidden_size: int = 256
    intermediate_size: int = 688
    vocab_size: int = 256
    rope_theta: float = 10000.0
    rope_scaling: dict[str, object] | None = None
    tie_word_embeddings: bool = True

    @property
    def is_toy(self) -> bool:
        """Whether this is the built-in randomly-initialized toy model."""
        return self.model_name_or_path == "toy"


@dataclass(frozen=True)
class CacheConfig:
    """KV cache sizing and block configuration."""

    block_size: int = 16
    num_gpu_blocks: int | None = None
    num_cpu_blocks: int = 256
    cache_dtype: str = "auto"

    def kv_bytes_per_token(self, model: ModelConfig) -> int:
        """Calculate KV cache memory per token in bytes.

        Formula: 2 (K and V) × num_layers × num_kv_heads × head_dim × dtype_bytes
        """
        dtype_bytes = _dtype_size(self.cache_dtype if self.cache_dtype != "auto" else model.dtype)
        return 2 * model.num_layers * model.num_kv_heads * model.head_dim * dtype_bytes

    def kv_bytes_per_block(self, model: ModelConfig) -> int:
        """KV cache memory per block in bytes."""
        return self.kv_bytes_per_token(model) * self.block_size

    def estimate_num_blocks(self, model: ModelConfig, available_bytes: int) -> int:
        """Estimate how many KV blocks fit in the given memory budget."""
        bpb = self.kv_bytes_per_block(model)
        if bpb == 0:
            return 0
        return available_bytes // bpb


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler budget constraints."""

    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    max_paddings: int = 256


@dataclass(frozen=True)
class EngineConfig:
    """Top-level configuration composing model, cache, and scheduler configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    device: str = "cpu"
    seed: int = 42

    def log_memory_budget(self) -> str:
        """Return a human-readable summary of KV memory arithmetic."""
        bpt = self.cache.kv_bytes_per_token(self.model)
        bpb = self.cache.kv_bytes_per_block(self.model)
        num_blocks = self.cache.num_gpu_blocks or 0
        total_kv_mb = (num_blocks * bpb) / (1024 * 1024)
        max_tokens = num_blocks * self.cache.block_size

        lines = [
            f"Model: {self.model.model_name_or_path}",
            f"  Layers: {self.model.num_layers}, KV heads: {self.model.num_kv_heads}, "
            f"Head dim: {self.model.head_dim}",
            f"  KV bytes/token: {bpt:,} ({bpt / 1024:.1f} KiB)",
            f"  Block size: {self.cache.block_size} tokens, "
            f"Block bytes: {bpb:,} ({bpb / 1024:.1f} KiB)",
            f"  GPU blocks: {num_blocks:,}, Total KV: {total_kv_mb:.1f} MiB",
            f"  Max concurrent tokens: {max_tokens:,}",
        ]
        return "\n".join(lines)


def _dtype_size(dtype_str: str) -> int:
    """Return byte size for a dtype string."""
    sizes: dict[str, int] = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
    }
    return sizes.get(dtype_str, 4)
