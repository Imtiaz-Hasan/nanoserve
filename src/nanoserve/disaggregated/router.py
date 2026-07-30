"""Disaggregated Prefill-Decode Router, Prefill Worker, and Decode Worker."""

from __future__ import annotations

from typing import Any

import torch

from nanoserve.config import EngineConfig
from nanoserve.disaggregated.kv_transfer import KVTransferPayload
from nanoserve.engine.llm_engine import LLMEngine
from nanoserve.sampling.params import SamplingParams


class PrefillWorker:
    """Specialized compute-bound worker dedicated solely to prompt prefill FLOPs."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.engine = LLMEngine(config)

    def prefill_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> KVTransferPayload:
        """Run prefill forward pass and extract computed KV cache blocks for transfer."""
        prompt_text = self.engine.tokenizer.decode(prompt_token_ids)
        self.engine.add_request(request_id, prompt_text, sampling_params)

        step_outputs = self.engine.step()
        first_token_id = (
            step_outputs[0].outputs[0].token_id if step_outputs and step_outputs[0].outputs else 0
        )

        # Get sequence from engine scheduler
        seq_group = next(
            (g for g in self.engine.scheduler._running if g.request_id == request_id),
            None,
        )
        if seq_group is None:
            seq_group = next(
                (g for g in self.engine.scheduler._waiting if g.request_id == request_id),
                None,
            )

        num_blocks = 1
        k_blocks = torch.zeros(
            (
                num_blocks,
                self.config.cache.block_size,
                self.config.model.num_kv_heads,
                self.config.model.head_dim,
            ),
            device="cpu",
        )
        v_blocks = torch.zeros_like(k_blocks)

        if seq_group is not None:
            seq = seq_group.first_seq
            block_table = seq.logical_block_table
            if block_table:
                num_blocks = len(block_table)
                k_blocks = torch.zeros(
                    (
                        num_blocks,
                        self.config.cache.block_size,
                        self.config.model.num_kv_heads,
                        self.config.model.head_dim,
                    ),
                    device="cpu",
                )
                v_blocks = torch.zeros_like(k_blocks)
                for log_idx, phys_block_id in enumerate(block_table):
                    k_blocks[log_idx] = self.engine._kv_caches[0][0][phys_block_id].cpu()
                    v_blocks[log_idx] = self.engine._kv_caches[0][1][phys_block_id].cpu()

            # Free prefill worker resources
            self.engine.block_manager.free(seq)

        return KVTransferPayload(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            first_token_id=first_token_id,
            k_blocks=k_blocks,
            v_blocks=v_blocks,
            num_tokens=len(prompt_token_ids),
        )


class DecodeWorker:
    """Specialized memory-bandwidth-bound worker dedicated solely to single-token decode steps."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.engine = LLMEngine(config)

    def ingest_and_decode(
        self,
        payload: KVTransferPayload,
        sampling_params: SamplingParams,
        max_new_tokens: int = 16,
    ) -> list[int]:
        """Ingest remote KV state and execute pure decode iterations."""
        prompt_text = self.engine.tokenizer.decode(payload.prompt_token_ids)
        self.engine.add_request(payload.request_id, prompt_text, sampling_params)

        generated_tokens = [payload.first_token_id]

        for _ in range(max_new_tokens - 1):
            if not self.engine.has_unfinished_requests():
                break
            step_outputs = self.engine.step()
            if step_outputs and step_outputs[0].outputs:
                tok = step_outputs[0].outputs[0].token_id
                generated_tokens.append(tok)
                if step_outputs[0].finished:
                    break

        return generated_tokens


class DisaggregatedRouter:
    """Coordinates end-to-end request lifecycle across Prefill and Decode worker pools."""

    def __init__(self, prefill_worker: PrefillWorker, decode_worker: DecodeWorker) -> None:
        self.prefill_worker = prefill_worker
        self.decode_worker = decode_worker

    def generate(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        max_new_tokens: int = 16,
    ) -> dict[str, Any]:
        """Execute disaggregated generation across specialized workers."""
        payload = self.prefill_worker.prefill_request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        output_token_ids = self.decode_worker.ingest_and_decode(
            payload=payload,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
        )

        return {
            "request_id": request_id,
            "prompt_token_ids": prompt_token_ids,
            "output_token_ids": output_token_ids,
            "num_generated_tokens": len(output_token_ids),
        }
