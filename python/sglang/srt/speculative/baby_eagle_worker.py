"""
Baby EAGLE worker for SGLang speculative decoding.

Baby EAGLE is a tiny (~19M param) draft model that:
- Fits in L2 cache (35.6 MB)
- Uses KV cross-attention from target model
- Trained on hidden states for speculative decoding
"""

import logging
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from sglang.srt.managers.schedule_batch import ScheduleBatch, alloc_for_decode
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs

# Import the proper loader from integration module
from sglang.srt.speculative.baby_eagle_integration import (
    load_baby_eagle_checkpoint,
    BabyEagleKVCudaGraphRunner,
    extract_kv_from_pool,
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

        # Per-request hidden state cache (for speculation)
        # Maps request ID -> hidden_states tensor [1, hidden_dim]
        self.hidden_state_cache: Dict[str, torch.Tensor] = {}

        # Layer to extract KV from (configurable via env)
        import os
        self.kv_layer_idx = int(os.environ.get("BABY_EAGLE_KV_LAYER", "16"))

        # Configure target model to capture PRE-norm hidden states from the KV layer
        # This is critical - Baby EAGLE was trained on PRE-norm states
        if hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
            self.model_runner.model.set_eagle3_layers_to_capture([self.kv_layer_idx])
            logger.info(f"Configured target model to capture PRE-norm hidden states from layer {self.kv_layer_idx}")
        else:
            logger.warning(f"Target model does not support set_eagle3_layers_to_capture - hidden states may be POST-norm!")

        logger.info(f"Baby EAGLE worker initialized with {self.num_draft_tokens} draft tokens, KV layer={self.kv_layer_idx}")

    def clear_cache_pool(self):
        """Clear any cached state."""
        pass

    def _extract_prenorm_hidden_states(self, batch_result, batch_size: int):
        """
        Extract PRE-norm hidden states from aux_hidden_states.

        Returns tensor of shape [batch_size, hidden_dim] with PRE-norm states from kv_layer_idx.
        Falls back to POST-norm final hidden states if aux states not available.
        """
        hidden_states_output = batch_result.logits_output.hidden_states

        # Check if we got aux_hidden_states (tuple return)
        if isinstance(hidden_states_output, tuple) and len(hidden_states_output) == 2:
            final_hidden_states, aux_hidden_states = hidden_states_output

            # aux_hidden_states is a list, one entry per captured layer
            # We configured to capture self.kv_layer_idx, so it should be the first (and only) entry
            if len(aux_hidden_states) > 0:
                # Take the last token's hidden state from the captured layer
                prenorm_states = aux_hidden_states[0]  # [batch, seq_len, hidden_dim]
                # Get last token for each request
                if prenorm_states.dim() == 3:
                    prenorm_states = prenorm_states[:, -1, :]  # [batch, hidden_dim]

                logger.debug(f"Extracted PRE-norm hidden states from layer {self.kv_layer_idx}, shape={prenorm_states.shape}")
                return prenorm_states
            else:
                logger.warning("aux_hidden_states is empty! Falling back to POST-norm states")

        # Fallback: use final POST-norm hidden states
        if hidden_states_output is not None:
            if isinstance(hidden_states_output, tuple):
                hidden_states_output = hidden_states_output[0]  # Extract final states

            logger.debug(f"Using POST-norm final hidden states (shape={hidden_states_output.shape})")
            return hidden_states_output

        return None

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """
        Main entry point for speculative decoding (v1 interface like NGRAM).

        Note: Full integration requires hidden state capture from target model.
        Currently operates in pass-through mode (no speculation).
        """
        if batch.forward_mode.is_extend():
            # Prefill: run target model and capture hidden states
            model_worker_batch = batch.get_model_worker_batch()
            # Request last token's hidden state for speculation
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Store PRE-norm hidden states for each request (for next decode step)
            hidden_states = self._extract_prenorm_hidden_states(batch_result, len(batch.reqs))
            if hidden_states is not None:
                for i, req in enumerate(batch.reqs):
                    if i < hidden_states.shape[0]:
                        self.hidden_state_cache[req.rid] = hidden_states[i:i+1].clone()
                logger.debug(f"Prefill: captured PRE-norm hidden states for {len(batch.reqs)} reqs, shape={hidden_states.shape}")

            return GenerationBatchResult(
                logits_output=batch_result.logits_output,
                next_token_ids=batch_result.next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
                accept_lens=None,
            )

        # For decode: prepare batch first
        # The scheduler skips prepare_for_decode() when spec_algorithm is set,
        # so we must complete the preparation manually.
        self._prepare_decode_batch(batch)

        # Attempt to generate drafts using Baby EAGLE
        draft_tokens = self._generate_drafts(batch)

        if draft_tokens is not None:
            # Draft generation succeeded - run verify with draft tokens
            return self._verify_drafts(batch, draft_tokens)

        # Fall back to pass-through mode
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch
        )

        # Update PRE-norm hidden state cache
        hidden_states = self._extract_prenorm_hidden_states(batch_result, len(batch.reqs))
        if hidden_states is not None:
            for i, req in enumerate(batch.reqs):
                if i < hidden_states.shape[0]:
                    self.hidden_state_cache[req.rid] = hidden_states[i:i+1].clone()

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

    def _generate_drafts(self, batch: ScheduleBatch) -> Optional[torch.Tensor]:
        """Generate draft tokens using Baby EAGLE.

        Returns None if drafts cannot be generated (missing hidden states or KV).
        """
        # Only support batch size 1 for now (Baby EAGLE CUDA graphs are for bs=1)
        if batch.batch_size() != 1:
            return None

        req = batch.reqs[0]

        # Get cached hidden states from previous forward
        hidden_states = self.hidden_state_cache.get(req.rid)
        if hidden_states is None:
            logger.debug(f"No cached hidden states for req {req.rid}")
            return None

        # Extract KV cache from target model layer
        try:
            token_to_kv_pool = self.target_worker.model_runner.token_to_kv_pool
            target_k, target_v = extract_kv_from_pool(
                token_to_kv_pool=token_to_kv_pool,
                req_to_token=batch.req_to_token_pool.req_to_token,
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens,
                layer_idx=self.kv_layer_idx,
            )
        except Exception as e:
            logger.debug(f"Failed to extract KV cache: {e}")
            return None

        # Generate draft tokens
        draft_tokens = self.draft(hidden_states, target_k, target_v)
        logger.debug(f"Generated drafts: {draft_tokens.tolist()}")
        return draft_tokens

    def _verify_drafts(
        self, batch: ScheduleBatch, draft_tokens: torch.Tensor
    ) -> GenerationBatchResult:
        """Verify draft tokens with target model.

        For Move 5: Simple greedy verification without tree-structured attention.
        """
        batch_size = batch.batch_size()
        num_drafts = draft_tokens.shape[1]

        # For now, run target model on just the first draft token as a test
        # Full verification loop will be implemented in Move 6
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch
        )

        # Update PRE-norm hidden state cache
        hidden_states = self._extract_prenorm_hidden_states(batch_result, len(batch.reqs))
        if hidden_states is not None:
            for i, req in enumerate(batch.reqs):
                if i < hidden_states.shape[0]:
                    self.hidden_state_cache[req.rid] = hidden_states[i:i+1].clone()

        # Check if first draft token matches target output
        target_tokens = batch_result.next_token_ids
        if draft_tokens.shape[0] == target_tokens.shape[0]:
            first_draft = draft_tokens[:, 0]
            matches = (first_draft == target_tokens).sum().item()
            if matches > 0:
                self.total_accepted += matches
                logger.debug(f"Draft match rate: {matches}/{batch_size}")

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,  # Will be updated in Move 6
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
            accept_lens=None,
        )

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
