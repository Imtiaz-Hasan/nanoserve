"""Unit tests for Disaggregated Prefill-Decode Serving and KV Transfer Protocol."""

import torch

from nanoserve.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from nanoserve.disaggregated.kv_transfer import KVTransferPayload
from nanoserve.disaggregated.router import DecodeWorker, DisaggregatedRouter, PrefillWorker
from nanoserve.sampling.params import SamplingParams


def test_kv_transfer_payload_serialization_roundtrip() -> None:
    """Verify KVTransferPayload serializes to binary bytes and recovers tensor shapes faithfully."""
    torch.manual_seed(42)
    k_blocks = torch.randn(4, 16, 2, 32)
    v_blocks = torch.randn(4, 16, 2, 32)

    payload = KVTransferPayload(
        request_id="req_123",
        prompt_token_ids=[1, 2, 3, 4, 5],
        first_token_id=42,
        k_blocks=k_blocks,
        v_blocks=v_blocks,
        num_tokens=5,
    )

    data_bytes = payload.to_bytes()
    assert isinstance(data_bytes, bytes)
    assert len(data_bytes) > 0

    recovered = KVTransferPayload.from_bytes(data_bytes)
    assert recovered.request_id == "req_123"
    assert recovered.prompt_token_ids == [1, 2, 3, 4, 5]
    assert recovered.first_token_id == 42
    assert recovered.num_tokens == 5
    assert torch.allclose(recovered.k_blocks, k_blocks, atol=1e-6)
    assert torch.allclose(recovered.v_blocks, v_blocks, atol=1e-6)


def test_disaggregated_prefill_and_decode_workers() -> None:
    """Verify PrefillWorker computes KV state and DecodeWorker ingests it to generate tokens."""
    torch.manual_seed(42)
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=16, block_size=4),
        scheduler=SchedulerConfig(max_num_seqs=4),
        device="cpu",
    )

    prefill_worker = PrefillWorker(config)
    decode_worker = DecodeWorker(config)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=8)
    prompt = [10, 20, 30, 40]

    # 1. Prefill
    payload = prefill_worker.prefill_request("test_req", prompt, sampling_params)
    assert payload.num_tokens == 4
    assert payload.k_blocks.shape[0] >= 1

    # 2. Decode
    output_tokens = decode_worker.ingest_and_decode(payload, sampling_params, max_new_tokens=4)
    assert len(output_tokens) >= 1
    assert output_tokens[0] == payload.first_token_id


def test_disaggregated_router_end_to_end() -> None:
    """Verify DisaggregatedRouter coordinates request lifecycle from prefill to decode."""
    torch.manual_seed(42)
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=16, block_size=4),
        scheduler=SchedulerConfig(max_num_seqs=4),
        device="cpu",
    )

    prefill_worker = PrefillWorker(config)
    decode_worker = DecodeWorker(config)
    router = DisaggregatedRouter(prefill_worker, decode_worker)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=6)
    res = router.generate("route_test", [1, 2, 3, 4, 5], sampling_params, max_new_tokens=4)

    assert res["request_id"] == "route_test"
    assert len(res["output_token_ids"]) >= 1
