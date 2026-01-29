"""
Baby EAGLE worker for SGLang speculative decoding.

Baby EAGLE is a tiny (~19M param) draft model that:
- Fits in L2 cache (35.6 MB)
- Uses KV cross-attention from target model
- Trained on hidden states for speculative decoding
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn

from sglang.srt.managers.schedule_batch import ScheduleBatch, alloc_for_decode
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs

# Import the proper loader from integration module
from sglang.srt.speculative.baby_eagle_integration import (
    load_baby_eagle_checkpoint,
    BabyEagleKVCudaGraphRunner,
)

logger = logging.getLogger(__name__)


class BabyEagleWorker:
    """
    Baby EAGLE speculative decoding worker for SGLang.

    Uses the trained BabyEagle model with KV cross-attention.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        # Get target model config
        target_config = target_worker.model_runner.model_config
        self.hidden_size = target_config.hidden_size
        self.model_config = target_config  # Required by scheduler

        # Speculative decoding params
        self.num_draft_tokens = server_args.speculative_num_steps or 3

        # Load the trained Baby EAGLE model
        checkpoint_path = server_args.speculative_draft_model_path
        if not checkpoint_path:
            raise ValueError("Baby EAGLE requires --speculative-draft-model-path")

        self.model = load_baby_eagle_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device,
            dtype=torch.float16,
        )

        # Create CUDA graph runner for efficient inference
        self.graph_runner = BabyEagleKVCudaGraphRunner(
            model=self.model,
            hidden_dim=self.hidden_size,
            k=self.num_draft_tokens,
            max_kv_len=2048,  # From model config
            device=self.device,
        )

        # Stats
        self.total_drafted = 0
        self.total_accepted = 0

        logger.info(f"Baby EAGLE worker initialized with {self.num_draft_tokens} draft tokens")

    def clear_cache_pool(self):
        """Clear any cached state."""
        pass

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """
        Main entry point for speculative decoding (v1 interface like NGRAM).

        Note: Full integration requires hidden state capture from target model.
        Currently operates in pass-through mode (no speculation).
        """
        if batch.forward_mode.is_extend():
            # Prefill: just run target model
            model_worker_batch = batch.get_model_worker_batch()
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            return GenerationBatchResult(
                logits_output=batch_result.logits_output,
                next_token_ids=batch_result.next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
                accept_lens=None,
            )

        # For decode: pass-through mode (no speculation yet)
        # The scheduler skips prepare_for_decode() when spec_algorithm is set,
        # so we must complete the preparation manually for pass-through mode.
        self._prepare_decode_batch(batch)

        model_worker_batch = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch
        )
        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
            accept_lens=None,
        )

    def _prepare_decode_batch(self, batch: ScheduleBatch):
        """Prepare batch for vanilla decode (no speculation).

        This completes the preparation that prepare_for_decode() skips
        when spec_algorithm is set.
        """
        # Handle penalizer if needed
        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                batch.output_ids.to(torch.int64)
            )

        # Set input_ids to last output tokens
        batch.input_ids = batch.output_ids
        batch.output_ids = None

        # Allocate memory for decode
        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)

        # Update req-level fields
        for req in batch.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        # Update seq_lens
        batch.seq_lens += 1
        batch.seq_lens_cpu += 1
        batch.orig_seq_lens += 1
        batch.seq_lens_sum = batch.seq_lens.sum().item()

    def draft(
        self,
        hidden_states: torch.Tensor,
        target_k: torch.Tensor,
        target_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate draft tokens using Baby EAGLE with KV cross-attention.

        Args:
            hidden_states: [batch, hidden_dim] from target model
            target_k: [batch, num_kv_heads, seq_len, head_dim] from layer 16
            target_v: [batch, num_kv_heads, seq_len, head_dim] from layer 16

        Returns:
            draft_tokens: [batch, k] draft token IDs
        """
        draft_tokens = self.graph_runner.run(hidden_states, target_k, target_v)
        self.total_drafted += draft_tokens.numel()
        return draft_tokens

    def get_stats(self) -> dict:
        """Get speculation statistics."""
        acceptance_rate = self.total_accepted / max(self.total_drafted, 1)
        return {
            "total_drafted": self.total_drafted,
            "total_accepted": self.total_accepted,
            "acceptance_rate": acceptance_rate,
        }
