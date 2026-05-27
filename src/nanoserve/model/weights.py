"""Weight loading utilities: safetensors, HuggingFace Hub, and random initialization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


# Llama-family weight name mappings from HuggingFace format to our format
_HF_WEIGHT_MAP: dict[str, str] = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}


def _build_layer_map(layer_idx: int) -> dict[str, str]:
    """Build weight name mapping for a single transformer layer."""
    hf_prefix = f"model.layers.{layer_idx}"
    our_prefix = f"layers.{layer_idx}"
    suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]
    return {f"{hf_prefix}.{s}": f"{our_prefix}.{s}" for s in suffixes}


def load_weights_from_safetensors(
    model_path: str | Path,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load weights from safetensors files in a model directory.

    Args:
        model_path: path to a directory containing .safetensors files
        dtype: target dtype for all weights
        device: target device

    Returns:
        State dict with our naming convention.
    """
    model_path = Path(model_path)
    state_dict: dict[str, torch.Tensor] = {}

    # Find all safetensors files
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        msg = f"No .safetensors files found in {model_path}"
        raise FileNotFoundError(msg)

    # Load config to determine number of layers
    config_path = model_path / "config.json"
    num_layers = 32  # default
    if config_path.exists():
        with open(config_path) as f:
            hf_config: dict[str, Any] = json.load(f)
        num_layers = int(hf_config.get("num_hidden_layers", 32))

    # Build full name mapping
    name_map = dict(_HF_WEIGHT_MAP)
    for i in range(num_layers):
        name_map.update(_build_layer_map(i))

    # Load and remap
    for sf_path in safetensor_files:
        raw = load_file(str(sf_path), device=device)
        for hf_name, tensor in raw.items():
            our_name = name_map.get(hf_name, hf_name)
            state_dict[our_name] = tensor.to(dtype=dtype)

    logger.info("Loaded %d weight tensors from %s", len(state_dict), model_path)
    return state_dict


def read_hf_config(model_path: str | Path) -> dict[str, Any]:
    """Read a HuggingFace config.json and return model parameters."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        msg = f"config.json not found in {model_path}"
        raise FileNotFoundError(msg)

    with open(config_path) as f:
        hf_config: dict[str, Any] = json.load(f)

    return {
        "num_layers": int(hf_config.get("num_hidden_layers", 32)),
        "num_heads": int(hf_config.get("num_attention_heads", 32)),
        "num_kv_heads": int(
            hf_config.get("num_key_value_heads", hf_config.get("num_attention_heads", 32))
        ),
        "hidden_size": int(hf_config.get("hidden_size", 4096)),
        "intermediate_size": int(hf_config.get("intermediate_size", 11008)),
        "vocab_size": int(hf_config.get("vocab_size", 32000)),
        "head_dim": int(
            hf_config.get(
                "head_dim",
                int(hf_config.get("hidden_size", 4096))
                // int(hf_config.get("num_attention_heads", 32)),
            )
        ),
        "rope_theta": float(hf_config.get("rope_theta", 10000.0)),
        "max_position_embeddings": int(hf_config.get("max_position_embeddings", 2048)),
        "tie_word_embeddings": bool(hf_config.get("tie_word_embeddings", True)),
    }
