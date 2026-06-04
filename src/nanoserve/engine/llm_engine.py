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
        self.block_manager = BlockManager(num_blocks=self.num_blocks, block_size=self.block_size)
        self.scheduler = Scheduler(config.scheduler, self.block_manager)

        # Sampler
        self.sampler = Sampler()

        # Physical KV caches: pre-allocated per layer
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
        """Execute one engine iteration: schedule → forward → sample → update.

        Returns outputs for all requests that produced a token this step.
        """
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.is_empty:
            return []

        outputs: list[RequestOutput] = []

        for seq_group in scheduler_output.scheduled_seq_groups:
            seq = seq_group.first_seq
            if seq.status.is_finished:
                continue

            is_prefill = seq.num_output_tokens == 0

            if is_prefill:
                # Prefill: process all prompt tokens
                token_ids = seq.prompt_token_ids
                positions = list(range(len(token_ids)))
                current_total_len = len(token_ids)
            else:
                # Decode: process only the last token
                token_ids = [seq.last_token_id]
                positions = [seq.num_total_tokens - 1]
                current_total_len = seq.num_total_tokens

            # Forward pass
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
            pos_tensor = torch.tensor(positions, dtype=torch.long, device=self._device)

            block_table_obj = self.block_manager.get_block_table(seq)
            if block_table_obj is None:
                continue
            physical_block_ids = block_table_obj.get_all_physical_blocks()
            slot_mapping_list = self.block_manager.get_slot_mapping(seq, positions)
            slot_mapping_tensor = torch.tensor(
                slot_mapping_list, dtype=torch.long, device=self._device
            )

            with torch.no_grad():
                logits, _ = self.model.forward(
                    input_ids=input_ids,
                    positions=pos_tensor,
                    kv_caches=self._kv_caches,
                    slot_mapping=slot_mapping_tensor,
                    block_tables=[physical_block_ids],
                    seq_lens=[current_total_len],
                )

            # Sample from the last position's logits
            last_logits = logits[:, -1, :]
            sampled = self.sampler.sample(
                logits=last_logits,
                sampling_params_list=[seq_group.sampling_params],
                history_tokens_list=[seq.all_token_ids],
                output_tokens_list=[seq.output_token_ids],
            )
            new_token_id = sampled[0]

            # Append token and decode
            seq.append_token(new_token_id)
            token_text = self.tokenizer.decode_token(new_token_id)
            self._generated_text[seq.seq_id] = self._generated_text.get(seq.seq_id, "") + token_text

            # Check stopping conditions
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
                # Free sequence's physical blocks
                self.block_manager.free(seq)
                self._generated_text.pop(seq.seq_id, None)

            output = RequestOutput(
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
            outputs.append(output)

            if finish_reason:
                self._stop_checkers.pop(seq_group.request_id, None)

        return outputs

    def has_unfinished_requests(self) -> bool:
        """Whether there are any requests still being processed."""
        return self.scheduler.has_unfinished_seqs()
