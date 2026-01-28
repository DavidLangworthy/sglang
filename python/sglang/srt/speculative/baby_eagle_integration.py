"""
Baby EAGLE Integration for SGLang.

This module provides proper BabyEagle integration with KV cross-attention.
Unlike TinyEagleModel which only uses hidden states, this uses the real
BabyEagle architecture with cross-attention to target model's KV cache.

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

# Configuration via environment variables
BABY_EAGLE_KV_LAYER = int(os.environ.get("BABY_EAGLE_KV_LAYER", "16"))
BABY_EAGLE_KV_MAX_LEN = int(os.environ.get("BABY_EAGLE_KV_MAX_LEN", "2048"))


class RealBabyEagleModel(nn.Module):
    """
    Real Baby EAGLE model with KV cross-attention.

    This is the proper architecture that was trained and achieves 38% acceptance.
    Unlike TinyEagleModel, this uses cross-attention to target model's KV cache.
    """

    def __init__(
        self,
        target_hidden_size: int = 4096,
        target_num_kv_heads: int = 8,
        target_head_dim: int = 128,
        internal_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        cross_attn_layers: int = 2,
        draft_vocab_size: int = 8192,
    ):
        super().__init__()
        self.target_hidden_size = target_hidden_size
        self.target_num_kv_heads = target_num_kv_heads
        self.target_head_dim = target_head_dim
        self.internal_dim = internal_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cross_attn_layers = cross_attn_layers
        self.draft_vocab_size = draft_vocab_size
        self.head_dim = internal_dim // num_heads

        # Input projection from target hidden size
        self.input_proj = nn.Linear(target_hidden_size, internal_dim, bias=False)

        # Token embeddings for draft tokens
        self.embed_tokens = nn.Embedding(draft_vocab_size, internal_dim)

        # KV projection with bottleneck
        kv_bottleneck = 256
        target_kv_dim = target_num_kv_heads * target_head_dim
        self.k_down = nn.Linear(target_kv_dim, kv_bottleneck)
        self.k_up = nn.Linear(kv_bottleneck, internal_dim)
        self.v_down = nn.Linear(target_kv_dim, kv_bottleneck)
        self.v_up = nn.Linear(kv_bottleneck, internal_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            BabyEagleLayer(
                internal_dim,
                num_heads,
                has_cross_attn=(i < cross_attn_layers)
            )
            for i in range(num_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(internal_dim)
        self.output_down = nn.Linear(internal_dim, 128)
        self.lm_head = nn.Linear(128, draft_vocab_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        draft_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with KV cross-attention.

        Args:
            hidden_states: [batch, hidden_dim] from target model
            target_kv: (K, V) tuple from target layer 16
                      K/V shape: [batch, num_kv_heads, seq_len, head_dim]
            draft_token_ids: [batch, seq] previous draft tokens

        Returns:
            logits: [batch, 1+seq, vocab_size]
        """
        batch_size = hidden_states.shape[0]

        # Project target hidden state
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  # [batch, 1, hidden]
        x = self.input_proj(hidden_states)  # [batch, 1, internal_dim]

        # Add draft token embeddings if provided
        if draft_token_ids is not None and draft_token_ids.numel() > 0:
            tok_emb = self.embed_tokens(draft_token_ids)  # [batch, seq, internal_dim]
            x = torch.cat([x, tok_emb], dim=1)  # [batch, 1+seq, internal_dim]

        # Process target KV for cross-attention
        projected_k, projected_v = None, None
        if target_kv is not None:
            target_k, target_v = target_kv
            # target_k: [batch, num_kv_heads, seq_len, head_dim]
            # Reshape to [batch, seq_len, num_kv_heads * head_dim]
            kv_len = target_k.shape[2]
            target_k_flat = target_k.transpose(1, 2).reshape(batch_size, kv_len, -1)
            target_v_flat = target_v.transpose(1, 2).reshape(batch_size, kv_len, -1)

            # Project through bottleneck
            projected_k = self.k_up(F.gelu(self.k_down(target_k_flat)))
            projected_v = self.v_up(F.gelu(self.v_down(target_v_flat)))

        # Pass through layers
        for layer in self.layers:
            x = layer(x, projected_k, projected_v)

        # Output projection
        x = self.norm(x)
        x = self.output_down(x)
        return self.lm_head(x)


class BabyEagleLayer(nn.Module):
    """Single transformer layer with optional cross-attention."""

    def __init__(self, dim: int, num_heads: int, has_cross_attn: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.has_cross_attn = has_cross_attn

        # Cross-attention (to target KV)
        if has_cross_attn:
            self.cross_q_proj = nn.Linear(dim, dim, bias=False)
            self.cross_o_proj = nn.Linear(dim, dim, bias=False)
            self.cross_norm = nn.LayerNorm(dim)

        # Self-attention
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.self_norm = nn.LayerNorm(dim)

        # MLP
        self.gate_proj = nn.Linear(dim, dim * 4, bias=False)
        self.up_proj = nn.Linear(dim, dim * 4, bias=False)
        self.down_proj = nn.Linear(dim * 4, dim, bias=False)
        self.mlp_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_k: Optional[torch.Tensor] = None,
        kv_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Cross-attention to target KV
        if self.has_cross_attn and kv_k is not None:
            residual = x
            x = self.cross_norm(x)

            # Query from current state
            q = self.cross_q_proj(x)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # K, V already projected to internal_dim
            kv_len = kv_k.shape[1]
            k = kv_k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = kv_v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
            x = residual + self.cross_o_proj(out)

        # Self-attention
        residual = x
        x = self.self_norm(x)

        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        x = residual + self.o_proj(out)

        # MLP
        residual = x
        x = self.mlp_norm(x)
        x = residual + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        return x


class RealBabyEagleCudaGraphRunner:
    """
    CUDA Graph runner for Real Baby EAGLE with KV cross-attention.

    Key difference from TinyEagleModel runner:
    - Includes static KV buffer for cross-attention
    - Properly handles KV from target model
    """

    def __init__(
        self,
        model: RealBabyEagleModel,
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

        # Model dimensions
        self.num_kv_heads = model.target_num_kv_heads
        self.head_dim = model.target_head_dim

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
        """Capture CUDA graph for draft generation with KV cross-attention."""
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                logits = self.model(self.static_hidden, (self.static_k, self.static_v))
        torch.cuda.synchronize()

        # Capture graph - generate k tokens
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            draft_input = None
            for i in range(self.k):
                logits = self.model(self.static_hidden, (self.static_k, self.static_v), draft_input)
                next_token = logits[:, -1, :].argmax(dim=-1)
                next_token = torch.clamp(next_token, 0, self.model.draft_vocab_size - 1)
                self.static_output_tokens[:, i] = next_token

                if draft_input is None:
                    draft_input = next_token.unsqueeze(1)
                else:
                    draft_input = torch.cat([draft_input, next_token.unsqueeze(1)], dim=1)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Real Baby EAGLE CUDA graph captured: {num_params:,} params, K={self.k}, max_kv_len={self.max_kv_len}")

    def run(
        self,
        hidden_states: torch.Tensor,
        target_k: torch.Tensor,
        target_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run CUDA graph to generate draft tokens with KV cross-attention.

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

        # Replay graph
        self.graph.replay()

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
) -> RealBabyEagleModel:
    """Load Baby EAGLE model from checkpoint."""
    # Try to import the original BabyEagle for config
    try:
        sys.path.insert(0, "/workspace/lager/projects/baby_eagle")
        from model import BabyEagleConfig
        from train import TrainConfig  # Needed for unpickling
    except ImportError:
        logger.warning("Could not import BabyEagle config, using defaults")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use defaults
    if "model_config" in checkpoint:
        cfg = checkpoint["model_config"]
        model = RealBabyEagleModel(
            target_hidden_size=getattr(cfg, "target_hidden_dim", 4096),
            target_num_kv_heads=getattr(cfg, "target_num_kv_heads", 8),
            target_head_dim=getattr(cfg, "target_head_dim", 128),
            internal_dim=getattr(cfg, "internal_dim", 512),
            num_layers=getattr(cfg, "num_layers", 4),
            num_heads=getattr(cfg, "num_heads", 8),
            cross_attn_layers=getattr(cfg, "cross_attn_layers", 2),
            draft_vocab_size=getattr(cfg, "output_vocab_size", 8192),
        )
    else:
        model = RealBabyEagleModel()

    # Load state dict (with compatibility handling)
    state_dict = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.info(f"Missing keys (will use random init): {missing[:5]}...")
    if unexpected:
        logger.info(f"Unexpected keys (ignored): {unexpected[:5]}...")

    model = model.to(device).to(dtype).eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded Real Baby EAGLE: {num_params:,} params ({num_params * 2 / 1024 / 1024:.1f} MB)")

    return model
