"""LoRA configuration definitions for dynamic multi-adapter serving."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoraConfig:
    """Configuration for a Low-Rank Adaptation (LoRA) adapter."""

    adapter_name: str
    r: int = 8
    lora_alpha: float = 16.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    @property
    def scaling(self) -> float:
        """Calculate the LoRA scaling factor s = alpha / r."""
        return self.lora_alpha / self.r if self.r > 0 else 1.0
