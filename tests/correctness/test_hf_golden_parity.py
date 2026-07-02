"""Golden parity testing: validates nanoserve Llama forward pass against SafeTensors checkpoint."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import save_file

from nanoserve.model.llama import LlamaForCausalLM


def test_hf_golden_parity_forward_pass(tmp_path: Path) -> None:
    """Verify numerical forward pass parity of model loaded via from_pretrained."""
    torch.manual_seed(42)

    vocab_size = 128
    hidden_size = 64
    intermediate_size = 128
    num_heads = 4
    num_kv_heads = 4
    head_dim = 16
    num_layers = 1

    # 1. Create config.json
    config_dict = {
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }
    with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f)

    # 2. Create synthetic weights
    weights = {
        "model.embed_tokens.weight": torch.randn(vocab_size, hidden_size),
        "model.layers.0.input_layernorm.weight": torch.ones(hidden_size),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden_size, hidden_size),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden_size, hidden_size),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden_size, hidden_size),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, hidden_size),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden_size),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(intermediate_size, hidden_size),
        "model.layers.0.mlp.up_proj.weight": torch.randn(intermediate_size, hidden_size),
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, intermediate_size),
        "model.norm.weight": torch.ones(hidden_size),
        "lm_head.weight": torch.randn(vocab_size, hidden_size),
    }
    save_file(weights, str(tmp_path / "model.safetensors"))

    # 3. Load via nanoserve from_pretrained
    model = LlamaForCausalLM.from_pretrained(str(tmp_path), dtype="float32", device="cpu")

    # 4. Run forward pass on test prompt
    input_ids = torch.tensor([[12, 34, 56, 78]], dtype=torch.long)
    positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        actual_logits, _ = model.forward(input_ids=input_ids, positions=positions)

    # 5. Compute reference forward pass manually
    with torch.no_grad():
        # Embedding
        x = F.embedding(input_ids, weights["model.embed_tokens.weight"])

        # Layer 0 RMSNorm
        x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

        # Q, K, V
        q = F.linear(x_norm, weights["model.layers.0.self_attn.q_proj.weight"])
        k = F.linear(x_norm, weights["model.layers.0.self_attn.k_proj.weight"])
        v = F.linear(x_norm, weights["model.layers.0.self_attn.v_proj.weight"])

        q = q.view(1, 4, num_heads, head_dim).transpose(1, 2)
        k = k.view(1, 4, num_heads, head_dim).transpose(1, 2)
        v = v.view(1, 4, num_heads, head_dim).transpose(1, 2)

        # Rotary Embedding
        q_rot, k_rot = model.rope.forward(q, k, positions)

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q_rot, k_rot, v, attn_mask=None, is_causal=True, scale=1.0 / (head_dim**0.5)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(1, 4, hidden_size)
        attn_proj = F.linear(attn_out, weights["model.layers.0.self_attn.o_proj.weight"])

        # Residual 1
        x = x + attn_proj

        # MLP Pre-norm
        x_mlp_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        gate = F.linear(x_mlp_norm, weights["model.layers.0.mlp.gate_proj.weight"])
        up = F.linear(x_mlp_norm, weights["model.layers.0.mlp.up_proj.weight"])
        mlp_act = F.silu(gate) * up
        mlp_out = F.linear(mlp_act, weights["model.layers.0.mlp.down_proj.weight"])

        # Residual 2
        x = x + mlp_out

        # Final norm
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

        # LM Head
        expected_logits = F.linear(x, weights["lm_head.weight"])

    # 6. Assert exact golden parity
    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-4, rtol=1e-4)
