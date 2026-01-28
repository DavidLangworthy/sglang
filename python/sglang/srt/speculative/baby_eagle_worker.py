"""
Baby EAGLE worker for SGLang speculative decoding.

Baby EAGLE is a tiny (~20M param) draft model designed to fit in L2 cache.
This version uses a simplified architecture compatible with SGLang's EAGLE infrastructure.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2, EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient

logger = logging.getLogger(__name__)


class TinyEagleModel(nn.Module):
    """
    Tiny EAGLE model that fits in L2 cache (~40MB).

    Architecture:
    - Input: hidden_size * 2 (current + previous hidden states)
    - 2 transformer layers at 512 dim
    - Output: vocab_size logits

    Total params: ~20M (fits in 50MB L2 cache at FP16)
    """

    def __init__(
        self,
        hidden_size: int = 4096,  # Target model hidden size
        internal_dim: int = 512,  # Internal dimension (small for L2 fit)
        num_layers: int = 2,
        num_heads: int = 8,
        vocab_size: int = 128256,  # Full vocab for Llama-3
        draft_vocab_size: int = 8000,  # Small vocab to fit in L2 cache (~40MB total)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.internal_dim = internal_dim
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size

        # Input: project from hidden_size*2 to internal_dim
        # (concatenation of current hidden and embedding of previous token)
        self.input_proj = nn.Linear(hidden_size * 2, internal_dim, bias=False)

        # Embedding for draft tokens (small vocab)
        self.embed_tokens = nn.Embedding(draft_vocab_size, internal_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TinyTransformerLayer(internal_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output
        self.norm = nn.RMSNorm(internal_dim, eps=1e-5)
        self.lm_head = nn.Linear(internal_dim, draft_vocab_size, bias=False)

        # Token mapping: draft vocab -> full vocab
        self.register_buffer(
            "draft_to_target",
            torch.arange(draft_vocab_size, dtype=torch.long)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, hidden_size] from target
        input_ids: Optional[torch.Tensor] = None,  # [batch, seq] draft tokens
        prev_hidden: Optional[torch.Tensor] = None,  # [batch, hidden_size] previous hidden
    ) -> torch.Tensor:
        """
        Forward pass for draft token generation.

        Args:
            hidden_states: Current hidden state from target model
            input_ids: Previously drafted tokens (for continuation)
            prev_hidden: Previous hidden state (for concatenation)

        Returns:
            logits: [batch, seq, draft_vocab_size]
        """
        batch_size = hidden_states.shape[0]

        # Concatenate current and previous hidden states
        if prev_hidden is None:
            prev_hidden = hidden_states

        # Project concatenated hidden states
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
            prev_hidden = prev_hidden.unsqueeze(1)

        concat_hidden = torch.cat([hidden_states, prev_hidden], dim=-1)
        x = self.input_proj(concat_hidden)  # [batch, 1, internal_dim]

        # Add embeddings if we have draft tokens
        if input_ids is not None:
            # Clamp to draft vocab size
            input_ids = torch.clamp(input_ids, 0, self.draft_vocab_size - 1)
            tok_emb = self.embed_tokens(input_ids)  # [batch, seq, internal_dim]
            x = torch.cat([x, tok_emb], dim=1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    def map_to_target_vocab(self, draft_tokens: torch.Tensor) -> torch.Tensor:
        """Map draft token IDs to target vocab IDs."""
        return self.draft_to_target[draft_tokens]


class TinyTransformerLayer(nn.Module):
    """Simple transformer layer for Tiny EAGLE."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.gate_proj = nn.Linear(dim, dim * 4, bias=False)
        self.up_proj = nn.Linear(dim, dim * 4, bias=False)
        self.down_proj = nn.Linear(dim * 4, dim, bias=False)

        self.norm1 = nn.RMSNorm(dim, eps=1e-5)
        self.norm2 = nn.RMSNorm(dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Self-attention
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Causal attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        x = residual + self.o_proj(out)

        # MLP
        residual = x
        x = self.norm2(x)
        x = residual + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        return x


class BabyEagleDraftWorker(EagleDraftWorker):
    """
    Baby EAGLE draft worker that uses a tiny model instead of full EAGLE.

    Key differences from EAGLE:
    - Uses TinyEagleModel (~20M params) instead of full EAGLE (~700M)
    - Model fits in L2 cache for low latency
    - Uses smaller draft vocab (32K) mapped to full vocab
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Store args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker

        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk or 4
        self.speculative_num_steps = server_args.speculative_num_steps or 5
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens or 64

        # Get target model config
        target_config = target_worker.model_runner.model_config
        hidden_size = target_config.hidden_size
        vocab_size = target_config.vocab_size

        # Create or load Baby EAGLE model
        if server_args.speculative_draft_model_path:
            self.model = self._load_checkpoint(server_args.speculative_draft_model_path)
        else:
            # Create fresh model with random weights (for testing)
            self.model = TinyEagleModel(
                hidden_size=hidden_size,
                internal_dim=512,
                num_layers=2,
                num_heads=8,
                vocab_size=vocab_size,
                draft_vocab_size=8000,
            )
            self.model = self.model.to(self.device).half()

        self.model.eval()

        # Log model info
        num_params = sum(p.numel() for p in self.model.parameters())
        mem_mb = num_params * 2 / (1024 * 1024)
        logger.info(f"Baby EAGLE model: {num_params:,} params, {mem_mb:.1f} MB")

        # Set up token mapping
        self.draft_vocab_size = self.model.draft_vocab_size

    def _load_checkpoint(self, path: str) -> TinyEagleModel:
        """Load Baby EAGLE from checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if "model_config" in checkpoint:
                config = checkpoint["model_config"]
                model = TinyEagleModel(
                    hidden_size=getattr(config, 'target_hidden_dim', 4096),
                    internal_dim=getattr(config, 'internal_dim', 512),
                    num_layers=getattr(config, 'num_layers', 2),
                    num_heads=getattr(config, 'num_heads', 8),
                    vocab_size=128256,
                    draft_vocab_size=getattr(config, 'output_vocab_size', 8000),
                )
            else:
                model = TinyEagleModel()

            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model = model.to(self.device).half()
            logger.info(f"Loaded Baby EAGLE from {path}")
            return model

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, using random weights")
            return TinyEagleModel().to(self.device).half()

    @torch.no_grad()
    def draft_forward(
        self,
        forward_batch: ForwardBatch,
    ):
        """
        Run Baby EAGLE forward pass for drafting.

        This overrides EAGLE's draft_forward to use our tiny model.
        """
        spec_info = forward_batch.spec_info
        hidden_states = spec_info.hidden_states

        # Initialize storage for tree building
        parent_list = []
        top_scores_list = []
        draft_tokens_list = []

        # First step: predict from hidden states
        logits = self.model(hidden_states)  # [batch, 1, vocab]
        logits = logits[:, -1, :]  # [batch, vocab]

        probs = F.softmax(logits, dim=-1)
        top_scores, top_indices = torch.topk(probs, self.topk, dim=-1)

        # Store first level
        parent_list.append(torch.full_like(top_indices, -1))
        top_scores_list.append(top_scores)
        draft_tokens_list.append(top_indices)

        # Subsequent steps
        prev_hidden = hidden_states
        for step in range(1, self.speculative_num_steps):
            # Get previous level's tokens
            prev_tokens = draft_tokens_list[-1]  # [batch, topk]

            # Flatten for batch processing
            batch_size = prev_tokens.shape[0]
            flat_tokens = prev_tokens.reshape(-1)  # [batch * topk]

            # Expand hidden states for each token
            expanded_hidden = hidden_states.repeat_interleave(self.topk, dim=0)
            expanded_prev = prev_hidden.repeat_interleave(self.topk, dim=0)

            # Forward with previous tokens
            logits = self.model(
                expanded_hidden,
                flat_tokens.unsqueeze(1),
                expanded_prev
            )
            logits = logits[:, -1, :]  # [batch*topk, vocab]

            probs = F.softmax(logits, dim=-1)
            top_scores, top_indices = torch.topk(probs, self.topk, dim=-1)

            # Reshape back: [batch, topk*topk] for this level
            top_scores = top_scores.reshape(batch_size, -1)
            top_indices = top_indices.reshape(batch_size, -1)

            # Parent indices: each group of topk shares same parent
            parent_idx = torch.arange(self.topk, device=self.device)
            parent_idx = parent_idx.repeat_interleave(self.topk)
            parent_idx = parent_idx.unsqueeze(0).expand(batch_size, -1)

            parent_list.append(parent_idx)
            top_scores_list.append(top_scores)
            draft_tokens_list.append(top_indices)

        # Concatenate all levels
        all_parents = torch.cat(parent_list, dim=1)
        all_scores = torch.cat(top_scores_list, dim=1)
        all_tokens = torch.cat(draft_tokens_list, dim=1)

        # Map draft tokens to target vocab
        all_tokens = self.model.map_to_target_vocab(all_tokens)

        return all_parents, all_scores, all_tokens


class BabyEagleWorker(EAGLEWorkerV2):
    """
    Baby EAGLE speculative decoding worker.

    Uses the same verification logic as EAGLE but with a tiny draft model.
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
        # Store basic args
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk or 4
        self.speculative_num_steps = server_args.speculative_num_steps or 5
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens or 64
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.BABY_EAGLE

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Create Baby EAGLE draft worker
        self._draft_worker = BabyEagleDraftWorker(
            server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, nccl_port, target_worker
        )

        # Dummy tensors (required by parent class)
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self.plan_stream = None
        self.plan_stream_ctx = None

        logger.info("Baby EAGLE worker initialized")

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> BabyEagleDraftWorker:
        return self._draft_worker
