"""LoRA Manager: dynamic registration, loading, and memory lifecycle management."""

from __future__ import annotations

from typing import Any

from nanoserve.lora.config import LoraConfig
from nanoserve.lora.models import LoraLinear, LoraWeight


class LoraManager:
    """Manages dynamic lifecycle of multiple LoRA adapters on top of a base model."""

    def __init__(self) -> None:
        self.configs: dict[str, LoraConfig] = {}
        self.lora_layers: dict[str, LoraLinear] = {}

    def register_lora_layer(self, layer_name: str, lora_layer: LoraLinear) -> None:
        """Register a LoraLinear layer to receive adapter weight updates."""
        self.lora_layers[layer_name] = lora_layer

    def register_adapter_config(self, config: LoraConfig) -> None:
        """Register a new LoRA adapter configuration."""
        self.configs[config.adapter_name] = config

    def load_adapter(self, name: str, weights_by_layer: dict[str, LoraWeight]) -> None:
        """Load adapter weights and inject into corresponding registered layers."""
        for layer_name, weight in weights_by_layer.items():
            if layer_name in self.lora_layers:
                self.lora_layers[layer_name].add_adapter(name, weight)

    def unload_adapter(self, name: str) -> None:
        """Unload and free adapter weights across all registered layers."""
        for lora_layer in self.lora_layers.values():
            lora_layer.remove_adapter(name)
        self.configs.pop(name, None)

    def is_adapter_loaded(self, name: str) -> bool:
        """Check if an adapter is active in registered layers."""
        return any(name in layer.adapters for layer in self.lora_layers.values())

    def list_adapters(self) -> list[str]:
        """List all currently loaded adapter names."""
        loaded: set[str] = set()
        for layer in self.lora_layers.values():
            loaded.update(layer.adapters.keys())
        return sorted(loaded)

    def get_adapter_info(self, name: str) -> dict[str, Any] | None:
        """Retrieve metadata for a loaded adapter."""
        if name in self.configs:
            cfg = self.configs[name]
            return {
                "adapter_name": cfg.adapter_name,
                "r": cfg.r,
                "lora_alpha": cfg.lora_alpha,
                "scaling": cfg.scaling,
                "target_modules": cfg.target_modules,
            }
        return None
