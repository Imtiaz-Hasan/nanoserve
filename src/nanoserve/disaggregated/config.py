"""Configuration definitions for Disaggregated Prefill-Decode serving clusters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DisaggregatedConfig:
    """Configuration for disaggregated prefill and decode worker pools."""

    role: str = "router"  # "router", "prefill", "decode"
    transfer_protocol: str = "shared_memory"  # "shared_memory", "tcp"
    prefill_workers: list[str] = field(default_factory=lambda: ["http://prefill-1:8000"])
    decode_workers: list[str] = field(default_factory=lambda: ["http://decode-1:8000"])
