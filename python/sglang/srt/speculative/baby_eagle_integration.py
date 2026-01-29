"""
Baby EAGLE Integration for SGLang.

This module provides proper BabyEagle integration with KV cross-attention.
It loads the trained BabyEagle model directly from the original model.py.

Key architecture:
- Uses KV from ONE target layer (layer 16), replicated for all BabyEagle layers
- Truncates KV to 2048 tokens max
- 38% acceptance rate (measured via eval_speculative.py)
"""

import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Add Baby EAGLE project path for model imports
_BABY_EAGLE_PATH = "/workspace/lager/projects/baby_eagle"
if _BABY_EAGLE_PATH not in sys.path:
    sys.path.insert(0, _BABY_EAGLE_PATH)

# Configuration via environment variables
BABY_EAGLE_KV_LAYER = int(os.environ.get("BABY_EAGLE_KV_LAYER", "16"))
BABY_EAGLE_KV_MAX_LEN = int(os.environ.get("BABY_EAGLE_KV_MAX_LEN", "2048"))


class BabyEagleKVCudaGraphRunner:
    """
    CUDA Graph runner for Baby EAGLE with KV cross-attention.

    Uses the original trained BabyEagle model and captures CUDA graphs
    for efficient inference.
    """

    def __init__(
        self,
        model: nn.Module,  # Original BabyEagle model
        hidden_dim: int,
        k: int,
        max_kv_len: int,
        device: str,
    ):
        self.model = model
        self.hidden_dim = hidden_dim
        self.k = k  # Number of draft tokens
        self.max_kv_len = max_kv_len
        self.device = device

        # Model dimensions from config
        config = model.config
        self.num_kv_heads = config.target_num_kv_heads
        self.head_dim = config.target_head_dim
        self.draft_vocab_size = config.output_vocab_size

        # Static buffers
        self.static_hidden = torch.zeros(1, hidden_dim, dtype=torch.float16, device=device)
        self.static_k = torch.zeros(1, self.num_kv_heads, max_kv_len, self.head_dim,
                                     dtype=torch.float16, device=device)
        self.static_v = torch.zeros(1, self.num_kv_heads, max_kv_len, self.head_dim,
                                     dtype=torch.float16, device=device)
        self.static_output_tokens = torch.zeros(1, k, dtype=torch.long, device=device)

        # KV length tracking
        self.actual_kv_len = max_kv_len

        # Capture graph
        self._capture_graph()

    def _capture_graph(self):
        """Capture CUDA graph for draft generation with KV cross-attention.

        Note: We capture k separate graphs, one for each draft step, to avoid
        the dynamic tensor concatenation issue in CUDA graphs.
        """
        # Create KV list format expected by model (single layer repeated)
        kv_list = [(self.static_k, self.static_v)]

        # Static buffer for draft tokens (used as input for autoregressive steps)
        self.static_draft_input = torch.zeros(1, self.k, dtype=torch.long, device=self.device)

        # Warmup all configurations
        for step in range(self.k):
            with torch.no_grad():
                if step == 0:
                    logits = self.model(self.static_hidden, kv_list, None)
                else:
                    draft_slice = self.static_draft_input[:, :step]
                    logits = self.model(self.static_hidden, kv_list, draft_slice)
        torch.cuda.synchronize()

        # Capture separate graphs for each step
        self.graphs = []
        self.static_logits = []

        for step in range(self.k):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                if step == 0:
                    logits = self.model(self.static_hidden, kv_list, None)
                else:
                    draft_slice = self.static_draft_input[:, :step]
                    logits = self.model(self.static_hidden, kv_list, draft_slice)

                # Store the logits tensor for later access
                self.static_logits.append(logits)

            self.graphs.append(graph)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Baby EAGLE CUDA graph captured: {num_params:,} params, K={self.k}, max_kv_len={self.max_kv_len}")

    def run(
        self,
        hidden_states: torch.Tensor,
        target_k: torch.Tensor,
        target_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run CUDA graphs to generate draft tokens with KV cross-attention.

        Args:
            hidden_states: [batch, hidden_dim]
            target_k: [batch, num_kv_heads, seq_len, head_dim] from target layer 16
            target_v: [batch, num_kv_heads, seq_len, head_dim] from target layer 16

        Returns:
            draft_tokens: [batch, k]
        """
        kv_len = target_k.shape[2]

        # Copy inputs to static buffers
        self.static_hidden.copy_(hidden_states)

        # Copy KV (right-aligned if shorter than max_len)
        if kv_len < self.max_kv_len:
            self.static_k.zero_()
            self.static_v.zero_()
            self.static_k[:, :, -kv_len:, :].copy_(target_k)
            self.static_v[:, :, -kv_len:, :].copy_(target_v)
        else:
            self.static_k.copy_(target_k[:, :, -self.max_kv_len:, :])
            self.static_v.copy_(target_v[:, :, -self.max_kv_len:, :])

        self.actual_kv_len = min(kv_len, self.max_kv_len)

        # Clear draft input buffer
        self.static_draft_input.zero_()

        # Generate k tokens using separate graphs
        for step in range(self.k):
            # Replay the graph for this step
            self.graphs[step].replay()

            # Get the token from logits
            logits = self.static_logits[step]
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.clamp(next_token, 0, self.draft_vocab_size - 1)

            # Store in output and input buffers
            self.static_output_tokens[:, step] = next_token
            if step < self.k - 1:
                self.static_draft_input[:, step] = next_token

        return self.static_output_tokens.clone()


def extract_kv_from_pool(
    token_to_kv_pool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    layer_idx: int = BABY_EAGLE_KV_LAYER,
    max_kv_len: int = BABY_EAGLE_KV_MAX_LEN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract KV cache from SGLang's memory pool for a specific layer.

    This converts from SGLang's flat format [total_slots, num_heads, head_dim]
    to HuggingFace format [batch, num_heads, seq_len, head_dim].

    Args:
        token_to_kv_pool: SGLang's KV pool with k_buffer and v_buffer
        req_to_token: Token mapping table [num_reqs, max_tokens_per_req]
        req_pool_indices: Which requests to extract [batch]
        seq_lens: Sequence lengths [batch]
        layer_idx: Which layer to extract (default: 16)
        max_kv_len: Maximum KV length to extract

    Returns:
        (K, V) tuple with shape [batch, num_heads, kv_len, head_dim]
    """
    batch_size = req_pool_indices.shape[0]
    device = req_pool_indices.device

    # Get the KV buffers for the target layer
    k_buffer = token_to_kv_pool.k_buffer[layer_idx]  # [total_slots, num_heads, head_dim]
    v_buffer = token_to_kv_pool.v_buffer[layer_idx]

    num_heads = k_buffer.shape[1]
    head_dim = k_buffer.shape[2]

    # Find the maximum sequence length (capped at max_kv_len)
    max_seq_len = min(seq_lens.max().item(), max_kv_len)

    # Allocate output tensors
    k_out = torch.zeros(batch_size, num_heads, max_seq_len, head_dim,
                        dtype=k_buffer.dtype, device=device)
    v_out = torch.zeros(batch_size, num_heads, max_seq_len, head_dim,
                        dtype=v_buffer.dtype, device=device)

    # Extract KV for each request
    for b in range(batch_size):
        req_idx = req_pool_indices[b].item()
        seq_len = min(seq_lens[b].item(), max_kv_len)

        if seq_len > 0:
            # Get token slot indices for this request (last seq_len tokens)
            start_idx = max(0, seq_lens[b].item() - seq_len)
            slot_indices = req_to_token[req_idx, start_idx:seq_lens[b].item()].long()

            # Extract KV from buffers
            k_tokens = k_buffer[slot_indices]  # [seq_len, num_heads, head_dim]
            v_tokens = v_buffer[slot_indices]

            # Place in output (right-aligned)
            out_start = max_seq_len - seq_len
            k_out[b, :, out_start:, :] = k_tokens.transpose(0, 1)  # [num_heads, seq_len, head_dim]
            v_out[b, :, out_start:, :] = v_tokens.transpose(0, 1)

    return k_out, v_out


def load_baby_eagle_checkpoint(
    checkpoint_path: str,
    device: str,
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Load Baby EAGLE model from checkpoint.

    Uses the original BabyEagle model class from model.py to ensure
    architecture compatibility with trained weights.
    """
    # Import original model
    try:
        from model import BabyEagle, BabyEagleConfig
        logger.info("Successfully imported BabyEagle from model.py")
    except ImportError as e:
        logger.error(f"Failed to import BabyEagle model: {e}")
        raise

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    if "model_config" in checkpoint:
        cfg_data = checkpoint["model_config"]
        # Handle both dict and dataclass formats
        if isinstance(cfg_data, dict):
            config = BabyEagleConfig(**cfg_data)
        else:
            config = cfg_data
        logger.info(f"Baby EAGLE config: internal_dim={config.internal_dim}, "
                   f"layers={config.num_layers}, heads={config.num_heads}, "
                   f"vocab={config.output_vocab_size}")
    else:
        logger.info("No model_config in checkpoint, using defaults")
        config = BabyEagleConfig()

    # Create model with config
    model = BabyEagle(config)

    # Load state dict
    state_dict = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys: {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}...")

    model = model.to(device).to(dtype).eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded Baby EAGLE: {num_params:,} params ({num_params * 2 / 1024 / 1024:.1f} MB)")

    return model
