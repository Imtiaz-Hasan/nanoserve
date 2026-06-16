"""LLMEngine: the synchronous heart of the serving engine.

One call to step() = one forward pass = one token per running sequence.
"""

from __future__ import annotations

import logging

import torch

from nanoserve.config import EngineConfig
from nanoserve.core.block_manager import BlockManager
from nanoserve.core.scheduler import Scheduler
from nanoserve.core.sequence import Sequence, SequenceGroup, SequenceStatus
from nanoserve.engine.output import CompletionOutput, RequestOutput
from nanoserve.kernels.reshape_cache import swap_blocks
from nanoserve.model.llama import LlamaForCausalLM
from nanoserve.model.loader import load_model
from nanoserve.sampling.params import SamplingParams
from nanoserve.sampling.sampler import Sampler
from nanoserve.sampling.stop import StopChecker

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """Byte-level tokenizer for the toy model. No sentencepiece, no BPE.

    For real models, this would be replaced by the HF tokenizer.
    Each byte value 0-255 is its own token. Vocab size = 256.
    """

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids (byte values)."""
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back to text."""
        # Clamp to valid byte range
        clamped = [max(0, min(255, t)) for t in token_ids]
        return bytes(clamped).decode("utf-8", errors="replace")

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id."""
        return self.decode([token_id])


class LLMEngine:
    """Synchronous LLM serving engine.

    Manages the model, scheduler, and KV cache. Each call to step()
    runs one forward pass and returns outputs for all active requests.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._device = torch.device(config.device)
        self._next_seq_id = 0

        # Load model
        logger.info("Loading model: %s on %s", config.model.model_name_or_path, config.device)
        self.model: LlamaForCausalLM = load_model(config.model, config.device)

        # Tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=config.model.vocab_size)

        # Block manager and scheduler
        self.num_blocks = config.cache.num_gpu_blocks or 256
        self.block_size = config.cache.block_size
        self.block_manager = BlockManager.from_config(config.model, config.cache)
        self.scheduler = Scheduler(config.scheduler, self.block_manager)

        # Sampler
        self.sampler = Sampler()

        # Physical GPU KV caches: pre-allocated per layer
        # Tensor shape: (num_blocks, block_size, num_kv_heads, head_dim)
        self._kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(config.model.num_layers):
            k_cache = torch.zeros(
                (
                    self.num_blocks,
                    self.block_size,
                    config.model.num_kv_heads,
                    config.model.head_dim,
                ),
                dtype=torch.float32,
                device=self._device,
            )
            v_cache = torch.zeros(
                (
                    self.num_blocks,
                    self.block_size,
                    config.model.num_kv_heads,
                    config.model.head_dim,
                ),
                dtype=torch.float32,
                device=self._device,
            )
            self._kv_caches.append((k_cache, v_cache))

        # Physical CPU KV caches for swap preemption
        self._cpu_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(config.model.num_layers):
            k_cpu = torch.zeros(
                (
                    config.cache.num_cpu_blocks,
                    self.block_size,
                    config.model.num_kv_heads,
                    config.model.head_dim,
                ),
                dtype=torch.float32,
                device="cpu",
            )
            v_cpu = torch.zeros_like(k_cpu)
            self._cpu_kv_caches.append((k_cpu, v_cpu))

        # Stop checkers per request
        self._stop_checkers: dict[str, StopChecker] = {}
        # Generated text per sequence (for stop-string detection)
        self._generated_text: dict[int, str] = {}

        logger.info("Engine initialized.\n%s", config.log_memory_budget())

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> None:
        """Submit a new generation request."""
        prompt_token_ids = self.tokenizer.encode(prompt)

        seq_id = self._next_seq_id
        self._next_seq_id += 1

        seq = Sequence(seq_id=seq_id, prompt_token_ids=prompt_token_ids)
        seq_group = SequenceGroup(
            request_id=request_id,
            sequences=[seq],
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
        )

        self._stop_checkers[request_id] = StopChecker(
            stop_strings=sampling_params.stop,
            stop_token_ids=sampling_params.stop_token_ids,
        )
        self._generated_text[seq_id] = ""

        self.scheduler.add_seq_group(seq_group)
        logger.debug(
            "Added request %s: %d prompt tokens, max_tokens=%d",
            request_id,
            len(prompt_token_ids),
            sampling_params.max_tokens,
        )

    def abort_request(self, request_id: str) -> None:
        """Abort a request and free its resources."""
        self.scheduler.abort_seq_group(request_id)
        self._stop_checkers.pop(request_id, None)

    def step(self) -> list[RequestOutput]:
        """Execute one engine iteration with continuous batching and preemption.

        Prefill requests run prompt prefill. Decode requests run batched
        multi-sequence decode in a single forward pass.
        """
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.is_empty:
            return []

        # 1. Execute swap-in transfers (CPU -> GPU)
        if scheduler_output.blocks_to_swap_in:
            swap_in_map = dict(scheduler_output.blocks_to_swap_in)
            for l_idx in range(len(self._kv_caches)):
                swap_blocks(
                    self._cpu_kv_caches[l_idx][0],
                    self._cpu_kv_caches[l_idx][1],
                    self._kv_caches[l_idx][0],
                    self._kv_caches[l_idx][1],
                    swap_in_map,
                )
            return []

        # 2. Execute swap-out transfers (GPU -> CPU)
        if scheduler_output.blocks_to_swap_out:
            swap_out_map = dict(scheduler_output.blocks_to_swap_out)
            for l_idx in range(len(self._kv_caches)):
                swap_blocks(
                    self._kv_caches[l_idx][0],
                    self._kv_caches[l_idx][1],
                    self._cpu_kv_caches[l_idx][0],
                    self._cpu_kv_caches[l_idx][1],
                    swap_out_map,
                )

        outputs: list[RequestOutput] = []

        scheduled = [
            sg
            for sg in scheduler_output.scheduled_seq_groups
            if not sg.first_seq.status.is_finished
        ]
        if not scheduled:
            return []

        all_input_ids: list[int] = []
        all_positions: list[int] = []
        all_slots: list[int] = []
        block_tables_list: list[list[int]] = []
        seq_lens_list: list[int] = []
        seq_token_ranges: list[tuple[int, int]] = []
        current_offset = 0

        for sg in scheduled:
            seq = sg.first_seq
            chunk_len = scheduler_output.seq_chunk_lens.get(seq.seq_id, 1)

            if chunk_len == 1 and seq.is_prefill_done:
                # Running decode token
                token_ids = [seq.last_token_id]
                positions = [seq.num_total_tokens - 1]
                total_kv_len = seq.num_total_tokens
            else:
                # Chunked prefill tokens
                start_pos = seq.num_computed_tokens
                token_ids = seq.all_token_ids[start_pos : start_pos + chunk_len]
                positions = list(range(start_pos, start_pos + chunk_len))
                total_kv_len = start_pos + chunk_len

            table = self.block_manager.get_block_table(seq)
            block_tables_list.append(table.get_all_physical_blocks() if table else [])
            seq_lens_list.append(total_kv_len)

            slots = self.block_manager.get_slot_mapping(seq, positions)
            all_input_ids.extend(token_ids)
            all_positions.extend(positions)
            all_slots.extend(slots)

            seq_token_ranges.append((current_offset, current_offset + chunk_len))
            current_offset += chunk_len

        input_ids = torch.tensor([all_input_ids], dtype=torch.long, device=self._device)
        pos_tensor = torch.tensor(all_positions, dtype=torch.long, device=self._device)
        slot_mapping_tensor = torch.tensor(all_slots, dtype=torch.long, device=self._device)

        with torch.no_grad():
            logits, _ = self.model.forward(
                input_ids=input_ids,
                positions=pos_tensor,
                kv_caches=self._kv_caches,
                slot_mapping=slot_mapping_tensor,
                block_tables=block_tables_list,
                seq_lens=seq_lens_list,
                seq_token_ranges=seq_token_ranges,
            )

        # Process each sequence's results
        for i, sg in enumerate(scheduled):
            seq = sg.first_seq
            start_idx, end_idx = seq_token_ranges[i]
            chunk_len = end_idx - start_idx
            last_token_pos = end_idx - 1
            last_logits = logits[:, last_token_pos, :]

            if not seq.is_prefill_done:
                seq.num_computed_tokens += chunk_len
                if seq.num_computed_tokens < len(seq.all_token_ids):
                    # Intermediate prefill chunk: don't sample yet
                    continue

            # Prefill completed on this step or in decode phase: sample next token
            sampled_tokens = self.sampler.sample(
                logits=last_logits,
                sampling_params_list=[sg.sampling_params],
                history_tokens_list=[seq.all_token_ids],
                output_tokens_list=[seq.output_token_ids],
            )
            new_token_id = sampled_tokens[0]

            seq.append_token(new_token_id)
            token_text = self.tokenizer.decode_token(new_token_id)
            self._generated_text[seq.seq_id] = self._generated_text.get(seq.seq_id, "") + token_text

            finish_reason = self._check_finish(seq, sg, new_token_id)
            outputs.append(self._make_output(seq, sg, new_token_id, token_text, finish_reason))

        return outputs

    def _check_finish(
        self,
        seq: Sequence,
        seq_group: SequenceGroup,
        new_token_id: int,
    ) -> str | None:
        """Check stopping criteria for a sequence and update status."""
        finish_reason: str | None = None
        stop_checker = self._stop_checkers.get(seq_group.request_id)

        if stop_checker and stop_checker.should_stop_token(new_token_id):
            finish_reason = "stop"
        elif stop_checker:
            matched = stop_checker.should_stop_string(self._generated_text.get(seq.seq_id, ""))
            if matched:
                finish_reason = "stop"

        if seq.num_output_tokens >= seq_group.sampling_params.max_tokens:
            finish_reason = "length"

        if finish_reason:
            seq.status = (
                SequenceStatus.FINISHED_STOPPED
                if finish_reason == "stop"
                else SequenceStatus.FINISHED_LENGTH
            )
            self.block_manager.free(seq)
            self._generated_text.pop(seq.seq_id, None)
            self._stop_checkers.pop(seq_group.request_id, None)

        return finish_reason

    def _make_output(
        self,
        seq: Sequence,
        seq_group: SequenceGroup,
        new_token_id: int,
        token_text: str,
        finish_reason: str | None,
    ) -> RequestOutput:
        """Construct RequestOutput dataclass."""
        return RequestOutput(
            request_id=seq_group.request_id,
            prompt=self.tokenizer.decode(seq_group.prompt_token_ids),
            outputs=[
                CompletionOutput(
                    index=0,
                    token_id=new_token_id,
                    text=token_text,
                    finish_reason=finish_reason,
                )
            ],
            finished=finish_reason is not None,
            prompt_token_ids=seq_group.prompt_token_ids,
            num_output_tokens=seq.num_output_tokens,
        )

    def has_unfinished_requests(self) -> bool:
        """Whether there are any requests still being processed."""
        return self.scheduler.has_unfinished_seqs()
