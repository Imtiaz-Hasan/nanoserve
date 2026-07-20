"""Unit tests for LoraManager (adapter registration, loading, unloading, and query)."""

import torch
import torch.nn as nn

from nanoserve.lora.config import LoraConfig
from nanoserve.lora.manager import LoraManager
from nanoserve.lora.models import LoraLinear, LoraWeight


def test_lora_manager_registration_and_info() -> None:
    """Verify adapter configuration registration and metadata retrieval."""
    manager = LoraManager()
    cfg = LoraConfig(
        adapter_name="med_lora",
        r=16,
        lora_alpha=32.0,
        target_modules=["q_proj", "v_proj"],
    )

    manager.register_adapter_config(cfg)
    assert cfg.scaling == 2.0

    info = manager.get_adapter_info("med_lora")
    assert info is not None
    assert info["adapter_name"] == "med_lora"
    assert info["r"] == 16
    assert info["scaling"] == 2.0


def test_lora_manager_lifecycle_load_and_unload() -> None:
    """Verify loading and unloading adapter weights across registered layers."""
    manager = LoraManager()

    q_proj = LoraLinear(nn.Linear(32, 32))
    v_proj = LoraLinear(nn.Linear(32, 32))

    manager.register_lora_layer("q_proj", q_proj)
    manager.register_lora_layer("v_proj", v_proj)

    # Prepare weights
    q_weight = LoraWeight(lora_a=torch.randn(8, 32), lora_b=torch.randn(32, 8), scaling=1.0)
    v_weight = LoraWeight(lora_a=torch.randn(8, 32), lora_b=torch.randn(32, 8), scaling=1.0)

    manager.load_adapter("sql_adapter", {"q_proj": q_weight, "v_proj": v_weight})

    assert manager.is_adapter_loaded("sql_adapter")
    assert "sql_adapter" in manager.list_adapters()
    assert "sql_adapter" in q_proj.adapters
    assert "sql_adapter" in v_proj.adapters

    # Unload adapter
    manager.unload_adapter("sql_adapter")
    assert not manager.is_adapter_loaded("sql_adapter")
    assert "sql_adapter" not in manager.list_adapters()
    assert "sql_adapter" not in q_proj.adapters
    assert "sql_adapter" not in v_proj.adapters
