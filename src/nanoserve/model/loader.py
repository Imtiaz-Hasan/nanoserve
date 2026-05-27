"""Model loader: instantiate LlamaForCausalLM from HuggingFace or create a toy model."""

from __future__ import annotations

import logging

import torch

from nanoserve.config import ModelConfig
from nanoserve.model.llama import LlamaForCausalLM
from nanoserve.model.weights import load_weights_from_safetensors

logger = logging.getLogger(__name__)


def create_toy_model(
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    head_dim: int = 64,
    vocab_size: int = 256,
    seed: int = 42,
) -> tuple[LlamaForCausalLM, ModelConfig]:
    """Create a small randomly-initialized model for CPU testing.

    Returns:
        (model, config) — the model and its corresponding ModelConfig.
    """
    hidden_size = num_heads * head_dim
    # SwiGLU intermediate size: roughly 8/3 * hidden_size, rounded to multiple of 8
    intermediate_size = ((hidden_size * 8 // 3 + 7) // 8) * 8

    config = ModelConfig(
        model_name_or_path="toy",
        dtype="float32",
        max_model_len=512,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        tie_word_embeddings=True,
    )

    torch.manual_seed(seed)
    model = LlamaForCausalLM(config)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Created toy model: %d layers, %d heads, hidden=%d, vocab=%d, params=%s",
        num_layers,
        num_heads,
        hidden_size,
        vocab_size,
        f"{num_params:,}",
    )

    return model, config


def load_model(
    model_config: ModelConfig,
    device: str = "cpu",
) -> LlamaForCausalLM:
    """Load a model from HuggingFace weights or create a toy model.

    Args:
        model_config: model configuration
        device: target device ('cpu' or 'cuda')

    Returns:
        Initialized LlamaForCausalLM.
    """
    if model_config.is_toy:
        model, _ = create_toy_model(
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
            vocab_size=model_config.vocab_size,
        )
        return model.to(device)

    dtype_map: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(model_config.dtype, torch.float32)

    model = LlamaForCausalLM(model_config)
    state_dict = load_weights_from_safetensors(
        model_config.model_name_or_path,
        dtype=dtype,
        device=device,
    )

    # Load weights with strict=False to handle tied embeddings gracefully
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        logger.warning("Unexpected weights: %s", unexpected)
    if missing:
        # Tied embeddings will show lm_head as missing — that's expected
        non_tied = [k for k in missing if k != "lm_head.weight"]
        if non_tied:
            logger.warning("Missing weights: %s", non_tied)

    model = model.to(device=torch.device(device), dtype=dtype)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded model %s: %s params, dtype=%s, device=%s",
        model_config.model_name_or_path,
        f"{num_params:,}",
        model_config.dtype,
        device,
    )

    return model
