"""Unit tests for SafeTensors weight loading (single-file, sharded, parameter mapping)."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from nanoserve.config import ModelConfig
from nanoserve.model.llama import LlamaForCausalLM
from nanoserve.model.weight_loader import (
    find_safetensors_files,
    load_model_config,
    load_safetensors_weights,
    normalize_param_name,
)


def test_normalize_param_name() -> None:
    """Verify HuggingFace parameter names are correctly normalized."""
    assert normalize_param_name("model.embed_tokens.weight") == "embed_tokens.weight"
    assert (
        normalize_param_name("model.layers.0.self_attn.q_proj.weight")
        == "layers.0.self_attn.q_proj.weight"
    )
    assert normalize_param_name("model.norm.weight") == "norm.weight"
    assert normalize_param_name("lm_head.weight") == "lm_head.weight"


def test_load_model_config_from_json(tmp_path: Path) -> None:
    """Verify config.json is parsed correctly into ModelConfig."""
    config_dict = {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 512,
        "intermediate_size": 1376,
        "vocab_size": 1000,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
    }
    with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f)

    config = load_model_config(tmp_path)
    assert config.num_layers == 4
    assert config.num_heads == 8
    assert config.num_kv_heads == 2
    assert config.hidden_size == 512
    assert config.head_dim == 64
    assert config.tie_word_embeddings is False
    assert config.dtype == "float16"


def test_single_safetensors_weight_loading(tmp_path: Path) -> None:
    """Verify loading weights from a single model.safetensors file."""
    config = ModelConfig(
        model_name_or_path=str(tmp_path),
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        head_dim=16,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=100,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)

    # Synthetic weight dictionary with HF naming schema
    embed_w = torch.randn(100, 32)
    q_w = torch.randn(32, 32)
    norm_w = torch.ones(32)

    weights = {
        "model.embed_tokens.weight": embed_w,
        "model.layers.0.self_attn.q_proj.weight": q_w,
        "model.norm.weight": norm_w,
    }
    save_file(weights, str(tmp_path / "model.safetensors"))

    load_safetensors_weights(model, tmp_path)

    torch.testing.assert_close(model.embed_tokens.weight, embed_w)
    torch.testing.assert_close(model.layers[0].self_attn.q_proj.weight, q_w)
    torch.testing.assert_close(model.norm.weight, norm_w)
    # Tied embeddings check
    torch.testing.assert_close(model.lm_head.weight, embed_w)


def test_sharded_safetensors_weight_loading(tmp_path: Path) -> None:
    """Verify loading weights across sharded safetensors files via index JSON."""
    config = ModelConfig(
        model_name_or_path=str(tmp_path),
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=16,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=100,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)

    # Shard 1
    w1 = {
        "model.embed_tokens.weight": torch.randn(100, 32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
    }
    save_file(w1, str(tmp_path / "model-00001-of-00002.safetensors"))

    # Shard 2
    w2 = {
        "model.layers.1.self_attn.q_proj.weight": torch.randn(32, 32),
        "lm_head.weight": torch.randn(100, 32),
    }
    save_file(w2, str(tmp_path / "model-00002-of-00002.safetensors"))

    # Index
    index = {
        "metadata": {"total_size": 1024},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        },
    }
    with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)

    shard_files = find_safetensors_files(tmp_path)
    assert len(shard_files) == 2

    load_safetensors_weights(model, tmp_path)
    torch.testing.assert_close(model.embed_tokens.weight, w1["model.embed_tokens.weight"])
    torch.testing.assert_close(
        model.layers[1].self_attn.q_proj.weight, w2["model.layers.1.self_attn.q_proj.weight"]
    )
    torch.testing.assert_close(model.lm_head.weight, w2["lm_head.weight"])


def test_safetensors_shape_mismatch_raises(tmp_path: Path) -> None:
    """Verify ValueError is raised if safetensors tensor shape doesn't match model architecture."""
    config = ModelConfig(
        model_name_or_path=str(tmp_path),
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        head_dim=16,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=100,
    )
    model = LlamaForCausalLM(config)

    # Incompatible shape for embed_tokens (expected 100x32, giving 50x32)
    bad_weights = {"model.embed_tokens.weight": torch.randn(50, 32)}
    save_file(bad_weights, str(tmp_path / "model.safetensors"))

    with pytest.raises(ValueError, match=r"Shape mismatch for parameter embed_tokens\.weight"):
        load_safetensors_weights(model, tmp_path)
