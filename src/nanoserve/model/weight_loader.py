"""Weight loader for SafeTensors checkpoints (single-file and sharded) with HF Hub support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file

from nanoserve.config import ModelConfig

if TYPE_CHECKING:
    from nanoserve.model.llama import LlamaForCausalLM


def load_model_config(model_name_or_path: str | Path) -> ModelConfig:
    """Load model architecture configuration from config.json."""
    model_path = Path(model_name_or_path)

    if model_path.is_dir() and (model_path / "config.json").exists():
        config_file = model_path / "config.json"
    else:
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            config_file = Path(
                hf_hub_download(repo_id=str(model_name_or_path), filename="config.json")
            )
        except Exception as err:
            msg = f"Failed to locate config.json in {model_name_or_path}: {err}"
            raise FileNotFoundError(msg) from err

    with open(config_file, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    num_layers = int(data.get("num_hidden_layers", data.get("num_layers", 2)))
    num_heads = int(data.get("num_attention_heads", data.get("num_heads", 4)))
    num_kv_heads = int(data.get("num_key_value_heads", data.get("num_kv_heads", num_heads)))
    hidden_size = int(data.get("hidden_size", 256))
    head_dim = int(data.get("head_dim", hidden_size // num_heads))
    intermediate_size = int(data.get("intermediate_size", 688))
    vocab_size = int(data.get("vocab_size", 256))
    rope_theta = float(data.get("rope_theta", 10000.0))
    tie_word_embeddings = bool(data.get("tie_word_embeddings", True))
    max_model_len = int(data.get("max_position_embeddings", 2048))
    dtype = str(data.get("torch_dtype", "float32"))

    return ModelConfig(
        model_name_or_path=str(model_name_or_path),
        dtype=dtype,
        max_model_len=max_model_len,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        rope_theta=rope_theta,
        tie_word_embeddings=tie_word_embeddings,
    )


def find_safetensors_files(model_name_or_path: str | Path) -> list[Path]:
    """Find all safetensors shard files for a given model directory or HF repo."""
    model_path = Path(model_name_or_path)

    if model_path.is_dir():
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            unique_shards = sorted(set(weight_map.values()))
            return [model_path / shard for shard in unique_shards]

        single_file = model_path / "model.safetensors"
        if single_file.exists():
            return [single_file]

        files = sorted(model_path.glob("*.safetensors"))
        if files:
            return files

        msg = f"No .safetensors files found in {model_path}"
        raise FileNotFoundError(msg)

    # HF Hub path
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        try:
            index_path = hf_hub_download(
                repo_id=str(model_name_or_path), filename="model.safetensors.index.json"
            )
            with open(index_path, encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            unique_shards = sorted(set(weight_map.values()))
            return [
                Path(hf_hub_download(repo_id=str(model_name_or_path), filename=shard))
                for shard in unique_shards
            ]
        except Exception:
            single_path = hf_hub_download(
                repo_id=str(model_name_or_path), filename="model.safetensors"
            )
            return [Path(single_path)]
    except Exception as err:
        msg = f"Failed to download SafeTensors weights from HF Hub for {model_name_or_path}: {err}"
        raise FileNotFoundError(msg) from err


def normalize_param_name(hf_name: str) -> str:
    """Translate HuggingFace tensor names to nanoserve parameter names."""
    # Strip 'model.' prefix
    if hf_name.startswith("model."):
        return hf_name[len("model.") :]
    return hf_name


def load_safetensors_weights(
    model: LlamaForCausalLM,
    model_name_or_path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> None:
    """Load SafeTensors weights into nanoserve LlamaForCausalLM instance."""
    shard_files = find_safetensors_files(model_name_or_path)
    state_dict = model.state_dict()

    for shard_file in shard_files:
        tensors = load_file(str(shard_file), device="cpu")
        for hf_key, tensor in tensors.items():
            param_key = normalize_param_name(hf_key)

            if param_key in state_dict:
                target_param = state_dict[param_key]
                if target_param.shape != tensor.shape:
                    msg = (
                        f"Shape mismatch for parameter {param_key}: "
                        f"expected {target_param.shape}, got {tensor.shape}"
                    )
                    raise ValueError(msg)
                target_param.copy_(tensor.to(device=device, dtype=dtype))

    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
