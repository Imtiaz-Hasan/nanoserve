"""Speculative decoding engine: coordinates proposer, target verification, and rollback."""

from __future__ import annotations

import torch

from nanoserve.config import EngineConfig
from nanoserve.engine.llm_engine import LLMEngine, RequestOutput
from nanoserve.sampling.params import SamplingParams
from nanoserve.speculative.proposer import NgramProposer, SpeculativeProposer
from nanoserve.speculative.verifier import SpeculativeVerifier


class SpeculativeEngine:
    """Speculative decoding serving engine.

    Orchestrates:
      1. Fast speculative candidate generation via N-gram or Draft Model proposer
      2. Parallel target model candidate verification forward pass
      3. KV cache state rollback on candidate rejection
      4. Performance and acceptance rate metric tracking
    """

    def __init__(
        self,
        config: EngineConfig,
        proposer: SpeculativeProposer | None = None,
    ) -> None:
        self.config = config
        self.target_engine = LLMEngine(config)

        if proposer is not None:
            self.proposer: SpeculativeProposer = proposer
        elif config.speculative.proposer_type == "ngram":
            self.proposer = NgramProposer(ngram_size=config.speculative.ngram_size)
        else:
            self.proposer = NgramProposer(ngram_size=3)

        self.verifier = SpeculativeVerifier()
        self.k_tokens = config.speculative.num_speculative_tokens

        # Metrics
        self.total_drafted_tokens: int = 0
        self.total_accepted_tokens: int = 0
        self.total_steps: int = 0
        self.total_emitted_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Ratio of accepted speculative tokens over total proposed draft tokens."""
        if self.total_drafted_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_drafted_tokens

    @property
    def tokens_per_step(self) -> float:
        """Average number of emitted output tokens per engine step iteration."""
        if self.total_steps == 0:
            return 0.0
        return self.total_emitted_tokens / self.total_steps

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        """Add request to the target engine."""
        self.target_engine.add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params or SamplingParams(),
        )

    def has_unfinished_requests(self) -> bool:
        """Whether there are active or waiting requests in the engine."""
        return self.target_engine.has_unfinished_requests()

    def abort_request(self, request_id: str) -> None:
        """Abort request and reclaim resources."""
        self.target_engine.abort_request(request_id)

    def step(self) -> list[RequestOutput]:
        """Execute one speculative iteration."""
        # Clean up finished sequence groups from scheduler running queue
        self.target_engine.scheduler._running = [
            sg
            for sg in self.target_engine.scheduler._running
            if not sg.first_seq.status.is_finished
        ]

        # Find if we have a running sequence eligible for speculative proposal
        if not self.target_engine.scheduler._running:
            # Prefill or swap iteration: execute standard engine step
            initial_outputs = self.target_engine.step()
            if initial_outputs:
                self.total_steps += 1
                self.total_emitted_tokens += sum(len(o.outputs) for o in initial_outputs)
            return initial_outputs

        sg = self.target_engine.scheduler._running[0]
        seq = sg.first_seq

        if not seq.is_prefill_done or seq.status.is_finished:
            running_outputs = self.target_engine.step()
            if running_outputs:
                self.total_steps += 1
                self.total_emitted_tokens += sum(len(o.outputs) for o in running_outputs)
            return running_outputs

        # Running decode sequence: attempt speculative proposal
        draft_candidates = self.proposer.propose(seq, self.k_tokens)
        if not draft_candidates:
            # Fallback to standard 1-token decode step
            fallback_outputs = self.target_engine.step()
            if fallback_outputs:
                self.total_steps += 1
                self.total_emitted_tokens += sum(len(o.outputs) for o in fallback_outputs)
            return fallback_outputs

        # Speculative execution path
        self.total_steps += 1
        num_draft = len(draft_candidates)
        self.total_drafted_tokens += num_draft

        verify_tokens = [seq.last_token_id, *draft_candidates]
        start_pos = seq.num_total_tokens - 1
        positions = list(range(start_pos, start_pos + len(verify_tokens)))

        # Ensure block allocation for candidate window
        orig_total_tokens = seq.num_total_tokens
        # Temporarily append draft tokens to sequence for block allocation
        for tok in draft_candidates:
            seq.append_token(tok)
        self.target_engine.block_manager.allocate(seq)

        # Build forward tensors
        input_ids = torch.tensor(
            [verify_tokens], dtype=torch.long, device=self.target_engine._device
        )
        pos_tensor = torch.tensor(positions, dtype=torch.long, device=self.target_engine._device)
        slots = self.target_engine.block_manager.get_slot_mapping(seq, positions)
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=self.target_engine._device)

        table = self.target_engine.block_manager.get_block_table(seq)
        block_tables_list = [table.get_all_physical_blocks() if table else []]
        seq_lens_list = [start_pos + len(verify_tokens)]
        seq_token_ranges = [(0, len(verify_tokens))]

        with torch.no_grad():
            logits, _ = self.target_engine.model.forward(
                input_ids=input_ids,
                positions=pos_tensor,
                kv_caches=self.target_engine._kv_caches,
                slot_mapping=slot_tensor,
                block_tables=block_tables_list,
                seq_lens=seq_lens_list,
                seq_token_ranges=seq_token_ranges,
            )

        # Roll back candidate tokens from sequence structure
        seq.output_token_ids = seq.output_token_ids[: orig_total_tokens - len(seq.prompt_token_ids)]

        # Verify candidate logits against proposed draft tokens
        cand_logits = logits[0, : num_draft + 1, :]
        sampling_params = sg.sampling_params

        if sampling_params.temperature == 0.0:
            accepted, bonus, num_accepted = self.verifier.verify_greedy(
                draft_candidates, cand_logits
            )
        else:
            accepted, bonus, num_accepted = self.verifier.verify_sampling(
                draft_candidates, cand_logits, temperature=sampling_params.temperature
            )

        self.total_accepted_tokens += num_accepted

        # Append all accepted tokens + corrective/bonus token to sequence
        emitted_this_step = [*accepted, bonus]
        self.total_emitted_tokens += len(emitted_this_step)

        spec_outputs: list[RequestOutput] = []
        for tok in emitted_this_step:
            seq.append_token(tok)
            token_text = self.target_engine.tokenizer.decode_token(tok)
            self.target_engine._generated_text[seq.seq_id] = (
                self.target_engine._generated_text.get(seq.seq_id, "") + token_text
            )
            finish_reason = self.target_engine._check_finish(seq, sg, tok)
            spec_outputs.append(
                self.target_engine._make_output(seq, sg, tok, token_text, finish_reason)
            )
            if finish_reason is not None:
                # Clean up finished sequence from running queue
                self.target_engine.scheduler._running = [
                    s
                    for s in self.target_engine.scheduler._running
                    if s.first_seq.seq_id != seq.seq_id
                ]
                break

        return spec_outputs
