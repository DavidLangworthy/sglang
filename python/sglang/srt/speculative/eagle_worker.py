import logging
import time
from typing import List, Optional, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.distributed import get_tp_group
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    detect_nan,
    draft_tp_context,
    fast_topk,
    generate_token_bitmask,
    get_last_loc_large_page_size_large_top_k,
    load_token_map,
    select_top_k_tokens,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    empty_context,
    get_available_gpu_memory,
    is_cuda,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


# ============================================================================
# L0 INTEGRATION - Ultra-fast draft predictor
# ============================================================================

# Enable/disable L0 via environment variable
L0_ENABLED = os.environ.get("L0_ENABLED", "1") == "1"
L0_K = int(os.environ.get("L0_K", "4"))  # Number of L0 draft tokens
L0_CHECKPOINT = os.environ.get("L0_CHECKPOINT", "/workspace/lager/projects/parlay/three_tier/checkpoints_diverse/l0_best.pt")
L0_VOCAB_SIZE = int(os.environ.get("L0_VOCAB_SIZE", "8192"))  # Reduced vocab for L0
L0_CONFIDENCE_THRESHOLD = float(os.environ.get("L0_CONFIDENCE_THRESHOLD", "0.7"))  # Use L0 when all probs >= this
L0_SKIP_EAGLE = os.environ.get("L0_SKIP_EAGLE", "1") == "1"  # Skip EAGLE entirely when L0 confident

# ============================================================================
# BABY EAGLE INTEGRATION - Tiny draft model that fits in L2 cache
# ============================================================================
BABY_EAGLE_ENABLED = os.environ.get("BABY_EAGLE_ENABLED", "0") == "1"
BABY_EAGLE_CHECKPOINT = os.environ.get("BABY_EAGLE_CHECKPOINT", "/workspace/lager/projects/baby_eagle/checkpoints_v2/best.pt")
BABY_EAGLE_VOCAB_SIZE = int(os.environ.get("BABY_EAGLE_VOCAB_SIZE", "8000"))


def create_linear_verify_input(
    l0_tokens: torch.Tensor,
    verified_id: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    num_draft_tokens: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create linear (non-tree) verification structures from L0 tokens.

    For L0-Skip-EAGLE mode: when L0 is confident, bypass EAGLE and
    create a simple linear verification path.

    IMPORTANT: Must pad to num_draft_tokens to match CUDA graph expectations.

    Returns same structure as build_tree_kernel_efficient:
    (tree_mask, position, retrive_index, retrive_next_token, retrive_next_sibling, draft_tokens)
    """
    batch_size = seq_lens.shape[0]
    k = l0_tokens.shape[1]  # Number of L0 tokens per batch
    actual_tokens_per_batch = k + 1  # verified_id + k L0 tokens

    # Combine verified_id with L0 tokens to create draft sequence
    # verified_id: [batch] - the token that was just verified
    # l0_tokens: [batch, k] - L0's predictions
    # draft sequence: [verified_id, l0_tok0, l0_tok1, ..., l0_tok(k-1)] padded to num_draft_tokens

    # Create draft tokens: flatten to [batch * num_draft_tokens]
    # Pad each batch's tokens to num_draft_tokens with 0 (padding token)
    draft_tokens = torch.cat([verified_id.unsqueeze(1), l0_tokens], dim=1)  # [batch, k+1]

    # Pad to num_draft_tokens per batch
    if actual_tokens_per_batch < num_draft_tokens:
        padding = torch.zeros(
            batch_size, num_draft_tokens - actual_tokens_per_batch,
            dtype=draft_tokens.dtype, device=device
        )
        draft_tokens = torch.cat([draft_tokens, padding], dim=1)  # [batch, num_draft_tokens]

    draft_tokens_flat = draft_tokens.view(-1)  # [batch * num_draft_tokens]

    # Positions: for each batch, positions are [seq_len, seq_len+1, ..., seq_len+num_draft_tokens-1]
    positions = []
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        positions.extend([seq_len + i for i in range(num_draft_tokens)])
    positions = torch.tensor(positions, dtype=torch.int64, device=device)

    # Tree mask: all True (causal masking is handled elsewhere)
    total_tokens = batch_size * num_draft_tokens
    tree_mask = torch.ones(total_tokens, dtype=torch.bool, device=device)

    # Retrieval indices for greedy verification
    # retrive_index: for each batch, which token indices are candidates
    # Shape: [batch, num_draft_tokens]
    # For linear, indices are 0, 1, 2, ..., k for real tokens, -1 for padding
    retrive_index = torch.full((batch_size, num_draft_tokens), -1, dtype=torch.long, device=device)
    for b in range(batch_size):
        for i in range(actual_tokens_per_batch):
            retrive_index[b, i] = i  # Local index within batch

    # retrive_next_token: linear chain - token i points to token i+1
    retrive_next_token = torch.full((batch_size, num_draft_tokens), -1, dtype=torch.long, device=device)
    for b in range(batch_size):
        for i in range(actual_tokens_per_batch - 1):
            retrive_next_token[b, i] = i + 1

    # retrive_next_sibling: no siblings in linear path
    retrive_next_sibling = torch.full((batch_size, num_draft_tokens), -1, dtype=torch.long, device=device)

    return tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling, draft_tokens_flat


class L0DraftPredictor(nn.Module):
    """
    L0: Lightweight predictor that drafts K0 tokens before EAGLE.

    Architecture: hidden_state -> fc1 -> ReLU -> fc2 -> logits
    Uses reduced vocab (8192) for speed - tokens map to first 8K of full vocab.
    """

    def __init__(self, hidden_dim: int = 4096, inner_dim: int = 1024, vocab_size: int = 8192):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.fc2 = nn.Linear(inner_dim, vocab_size, bias=False)
        self.embed = nn.Embedding(vocab_size, inner_dim)

    def draft_k_tokens(self, hidden: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draft K tokens autoregressively from hidden state."""
        tokens = []
        probs_list = []
        inner = F.relu(self.fc1(hidden))  # [batch, inner_dim]

        for i in range(k):
            logits = self.fc2(inner)  # [batch, vocab]
            probs = F.softmax(logits, dim=-1)
            conf, tok = probs.max(dim=-1)  # [batch], [batch]
            tokens.append(tok)
            probs_list.append(conf)
            if i < k - 1:
                # Clamp to valid vocab range
                tok_clamped = tok.clamp(0, self.vocab_size - 1)
                inner = F.relu(inner + self.embed(tok_clamped))

        # Return [batch, k], [batch, k]
        return torch.stack(tokens, dim=1), torch.stack(probs_list, dim=1)


# Global stats - updated during inference
class L0Stats:
    def __init__(self):
        self.calls = 0
        self.l0_time_us = 0
        self.l0_tokens = 0
        self.l0_matches = 0
        self.l0_used = 0  # Times L0 was confident enough to use
        self.l0_skipped_eagle = 0  # Eagle steps skipped due to L0

    def report(self):
        if self.calls > 0:
            avg_time = self.l0_time_us / self.calls
            match_rate = self.l0_matches / max(1, self.l0_tokens) * 100
            use_rate = self.l0_used / self.calls * 100
            logger.info(f"L0 Stats: {self.calls} calls, avg {avg_time:.1f}μs, match={match_rate:.1f}%, used={use_rate:.1f}%")
        return {
            "calls": self.calls,
            "l0_time_ms": self.l0_time_us / 1000,
            "match_rate": self.l0_matches / max(1, self.l0_tokens),
            "use_rate": self.l0_used / max(1, self.calls),
        }


_l0_global_stats = L0Stats()


class L0CudaGraphRunner:
    """CUDA Graph runner for L0 predictions - zero overhead version (no confidence check)."""

    def __init__(self, l0_model: L0DraftPredictor, hidden_dim: int, k: int, device: str):
        self.l0_model = l0_model
        self.hidden_dim = hidden_dim
        self.k = k
        self.device = device

        # Static buffers for CUDA graph
        self.static_hidden = torch.zeros(1, hidden_dim, dtype=torch.float16, device=device)
        self.static_tokens = None

        # Capture the graph
        self._capture_graph()

    def _capture_graph(self):
        """Capture CUDA graph for L0 - minimal version."""
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                self.l0_model.draft_k_tokens(self.static_hidden, self.k)
        torch.cuda.synchronize()

        # Capture graph - tokens only, no probs needed
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_tokens, _ = self.l0_model.draft_k_tokens(self.static_hidden, self.k)

        logger.info(f"L0 CUDA graph captured (no-confidence): hidden_dim={self.hidden_dim}, K={self.k}, vocab={self.l0_model.vocab_size}")

    def run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run L0 using CUDA graph. Returns tokens only (no sync)."""
        if hidden_states.shape[0] == 1:
            self.static_hidden.copy_(hidden_states)
            self.graph.replay()
            return self.static_tokens
        else:
            # Fallback to eager for batch > 1
            with torch.no_grad():
                tokens, _ = self.l0_model.draft_k_tokens(hidden_states, self.k)
            return tokens

# ============================================================================
# BABY EAGLE - Tiny draft model (~19M params, fits in L2 cache)
# ============================================================================

class TinyTransformerLayer(nn.Module):
    """Minimal transformer layer for Baby EAGLE."""

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
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        x = residual + self.o_proj(out)

        residual = x
        x = self.norm2(x)
        x = residual + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


class TinyEagleModel(nn.Module):
    """Tiny EAGLE model optimized for L2 cache residency (~19M params, 35.6 MB)."""

    def __init__(self, target_hidden_size: int = 4096, internal_dim: int = 512,
                 num_layers: int = 2, num_heads: int = 8, draft_vocab_size: int = 8000):
        super().__init__()
        self.target_hidden_size = target_hidden_size
        self.internal_dim = internal_dim
        self.draft_vocab_size = draft_vocab_size

        self.input_proj = nn.Linear(target_hidden_size, internal_dim, bias=False)
        self.embed_tokens = nn.Embedding(draft_vocab_size, internal_dim)
        self.layers = nn.ModuleList([
            TinyTransformerLayer(internal_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(internal_dim, eps=1e-5)
        self.lm_head = nn.Linear(internal_dim, draft_vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, draft_token_ids: torch.Tensor = None) -> torch.Tensor:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
        x = self.input_proj(hidden_states)
        if draft_token_ids is not None:
            tok_emb = self.embed_tokens(draft_token_ids)
            x = torch.cat([x, tok_emb], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class BabyEagleCudaGraphRunner:
    """CUDA Graph runner for Baby EAGLE - generates k draft tokens."""

    def __init__(self, model: TinyEagleModel, hidden_dim: int, k: int, device: str):
        self.model = model
        self.hidden_dim = hidden_dim
        self.k = k
        self.device = device
        self.vocab_size = model.draft_vocab_size

        # Static buffers
        self.static_hidden = torch.zeros(1, hidden_dim, dtype=torch.float16, device=device)
        self.static_tokens = None
        self.graphs = []
        self.graph_outputs = []

        self._capture_graphs()

    def _capture_graphs(self):
        """Capture CUDA graphs for tree-style drafting."""
        stream = torch.cuda.Stream()

        # For simplicity, generate k tokens autoregressively with graphs
        for i in range(self.k):
            # Warmup
            with torch.cuda.stream(stream):
                for _ in range(3):
                    with torch.no_grad():
                        if i == 0:
                            _ = self.model(self.static_hidden)
                        else:
                            tok = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                            _ = self.model(self.static_hidden, tok)
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                with torch.no_grad():
                    if i == 0:
                        out = self.model(self.static_hidden)
                    else:
                        tok = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                        out = self.model(self.static_hidden, tok)

            self.graphs.append(graph)
            self.graph_outputs.append(out)

        torch.cuda.synchronize()
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Baby EAGLE CUDA graphs captured: {num_params:,} params ({num_params * 2 / 1024 / 1024:.1f} MB), K={self.k}")

    def run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Generate k draft tokens using CUDA graphs."""
        if hidden_states.shape[0] != 1:
            # Fallback to eager for batch > 1
            return self._run_eager(hidden_states)

        self.static_hidden.copy_(hidden_states)
        tokens = []

        for i in range(self.k):
            self.graphs[i].replay()
            logits = self.graph_outputs[i][:, -1, :]
            tok = logits.argmax(dim=-1)
            tokens.append(tok)

        return torch.stack(tokens, dim=1)  # [1, k]

    def _run_eager(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback for batch > 1."""
        batch_size = hidden_states.shape[0]
        tokens = []
        with torch.no_grad():
            logits = self.model(hidden_states)
            tok = logits[:, -1, :].argmax(dim=-1)
            tokens.append(tok)
            for _ in range(self.k - 1):
                logits = self.model(hidden_states, tok.unsqueeze(1))
                tok = logits[:, -1, :].argmax(dim=-1)
                tokens.append(tok)
        return torch.stack(tokens, dim=1)

# ============================================================================


class EAGLEWorker(TpModelWorker):

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
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            super().__init__(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_model_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_model_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            self.eagle_use_aux_hidden_state = True
            eagle_config = getattr(
                self.draft_model_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

        # ============= L0 INTEGRATION =============
        self.l0_model = None
        self.l0_graph_runner = None
        self.l0_k = L0_K
        self._l0_capturing = False  # Flag to disable L0 during EAGLE CUDA graph capture

        if L0_ENABLED:
            try:
                hidden_dim = self.target_worker.model_runner.model_config.hidden_size
                # Use reduced vocab for L0 (maps to first 8K tokens)
                self.l0_model = L0DraftPredictor(
                    hidden_dim=hidden_dim,
                    inner_dim=1024,
                    vocab_size=L0_VOCAB_SIZE,
                ).to(self.device).half().eval()
                self._l0_hidden_dim = hidden_dim

                # Load trained weights if checkpoint exists
                if os.path.exists(L0_CHECKPOINT):
                    checkpoint = torch.load(L0_CHECKPOINT, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    # Load with strict=False to allow missing keys (embed.weight is initialized randomly)
                    missing, unexpected = self.l0_model.load_state_dict(state_dict, strict=False)
                    logger.info(f"L0 model loaded from {L0_CHECKPOINT}")
                    if missing:
                        logger.info(f"  Missing keys (initialized randomly): {missing}")
                else:
                    logger.warning(f"L0 checkpoint not found: {L0_CHECKPOINT} - using random weights")

                logger.info(f"L0 model: hidden_dim={hidden_dim}, vocab={L0_VOCAB_SIZE}, K={self.l0_k}, conf_thresh={L0_CONFIDENCE_THRESHOLD}")
            except Exception as e:
                logger.warning(f"L0 initialization failed: {e}")
                import traceback
                traceback.print_exc()
                self.l0_model = None
        # ==========================================

        # ============= BABY EAGLE INTEGRATION =============
        self.baby_eagle_model = None
        self.baby_eagle_graph_runner = None
        self.baby_eagle_k = L0_K  # Use same K as L0 for now
        self._baby_eagle_capturing = False

        if BABY_EAGLE_ENABLED:
            try:
                hidden_dim = self.target_worker.model_runner.model_config.hidden_size
                self.baby_eagle_model = TinyEagleModel(
                    target_hidden_size=hidden_dim,
                    internal_dim=512,
                    num_layers=2,
                    num_heads=8,
                    draft_vocab_size=BABY_EAGLE_VOCAB_SIZE,
                ).to(self.device).half().eval()
                self._baby_eagle_hidden_dim = hidden_dim

                # Load trained weights if checkpoint exists
                if os.path.exists(BABY_EAGLE_CHECKPOINT):
                    checkpoint = torch.load(BABY_EAGLE_CHECKPOINT, map_location=self.device, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    missing, unexpected = self.baby_eagle_model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Baby EAGLE model loaded from {BABY_EAGLE_CHECKPOINT}")
                    if missing:
                        logger.info(f"  Missing keys (initialized randomly): {missing}")
                else:
                    logger.warning(f"Baby EAGLE checkpoint not found: {BABY_EAGLE_CHECKPOINT} - using random weights")

                num_params = sum(p.numel() for p in self.baby_eagle_model.parameters())
                logger.info(f"Baby EAGLE model: {num_params:,} params ({num_params * 2 / 1024 / 1024:.1f} MB), hidden_dim={hidden_dim}, vocab={BABY_EAGLE_VOCAB_SIZE}, K={self.baby_eagle_k}")
            except Exception as e:
                logger.warning(f"Baby EAGLE initialization failed: {e}")
                import traceback
                traceback.print_exc()
                self.baby_eagle_model = None
        # ==========================================

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        # Disable L0 during CUDA graph capture
        self._l0_capturing = True

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        # Capture extend
        if self.draft_extend_attn_backend and not _is_npu:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(
                self
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        # Re-enable L0 and capture L0's CUDA graph
        self._l0_capturing = False
        if self.l0_model is not None:
            try:
                self.l0_graph_runner = L0CudaGraphRunner(
                    self.l0_model,
                    self._l0_hidden_dim,
                    self.l0_k,
                    self.device,
                )
                logger.info("L0 CUDA graph captured - L0 enabled for inference")
            except Exception as e:
                logger.warning(f"L0 CUDA graph capture failed: {e}")
                self.l0_graph_runner = None

        # Capture Baby EAGLE CUDA graphs
        self._baby_eagle_capturing = False
        if self.baby_eagle_model is not None:
            try:
                self.baby_eagle_graph_runner = BabyEagleCudaGraphRunner(
                    self.baby_eagle_model,
                    self._baby_eagle_hidden_dim,
                    self.baby_eagle_k,
                    self.device,
                )
                logger.info("Baby EAGLE CUDA graph captured - Baby EAGLE enabled for inference")
            except Exception as e:
                logger.warning(f"Baby EAGLE CUDA graph capture failed: {e}")
                self.baby_eagle_graph_runner = None

    @property
    def draft_model_runner(self):
        return self.model_runner

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(
                batch
            )
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.forward_draft_extend(
                    batch, logits_output.hidden_states, next_token_ids, seq_lens_cpu
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                spec_info = self.draft(batch)
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )

            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                if (
                    self.server_args.enable_dp_attention
                    or batch.spec_info.verified_id.shape[0] > 0
                ):
                    # decode is not finished
                    self.forward_draft_extend_after_decode(batch)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def check_forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        local_need_forward = batch.spec_info.verified_id.shape[0] > 0
        if not self.server_args.enable_dp_attention:
            return local_need_forward

        global_need_forward = torch.tensor(
            [
                (local_need_forward),
            ],
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(
            global_need_forward, group=get_tp_group().cpu_group
        )
        global_need_forward_cnt = global_need_forward[0].item()
        need_forward = global_need_forward_cnt > 0
        return need_forward

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int, Optional[torch.Tensor]]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        return (
            logits_output,
            next_token_ids,
            model_worker_batch.seq_lens_cpu,
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1

        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        if self.page_size == 1:
            alloc_len_per_decode = self.speculative_num_steps * self.topk
            # TODO: We only need self.speculative_num_steps - 1 * topk cache loc
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len_per_decode,
                backup_state=True,
            )
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens

                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size
                num_new_pages_per_topk = (
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        if self.page_size > 1 and self.topk > 1:
            last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
            duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (self.topk - 1)
            target_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
            source_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
        else:
            # When source_cache_loc is not needed, simply skip
            duplicate_cache_len = 0
            source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps + self.page_size),
        )

        if self.page_size > 1 and self.topk > 1:
            if duplicate_cache_len > 0:
                self.draft_model_runner.token_to_kv_pool.move_kv_cache(
                    target_cache_loc, source_cache_loc
                )
            # Remove padded slots
            # TODO: We only need self.speculative_num_steps - 1 cache loc
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self.model_config.hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def draft(self, batch: ScheduleBatch):
        # Parse args
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
        else:
            self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_batch = self.topk
        spec_info.num_tokens_for_logprob_per_batch = self.topk
        batch.return_hidden_states = False

        # Get forward batch
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )

        # ============= BABY EAGLE INTEGRATION (zero overhead - always use Baby EAGLE) =============
        if hasattr(self, 'baby_eagle_graph_runner') and self.baby_eagle_graph_runner is not None and BABY_EAGLE_ENABLED:
            spec_info_inner = forward_batch.spec_info
            hidden_states = getattr(spec_info_inner, 'hidden_states', None)
            if hidden_states is not None and not self._baby_eagle_capturing and not batch.forward_mode.is_idle():
                # Run Baby EAGLE - CUDA graph accelerated
                baby_eagle_tokens = self.baby_eagle_graph_runner.run(hidden_states)

                # Clamp to draft vocab
                baby_eagle_tokens = torch.clamp(baby_eagle_tokens, 0, BABY_EAGLE_VOCAB_SIZE - 1)

                (
                    tree_mask,
                    position,
                    retrive_index,
                    retrive_next_token,
                    retrive_next_sibling,
                    draft_tokens,
                ) = create_linear_verify_input(
                    baby_eagle_tokens,
                    spec_info.verified_id,
                    batch.seq_lens,
                    batch.seq_lens_sum,
                    self.speculative_num_draft_tokens,
                    self.device,
                )

                return EagleVerifyInput(
                    draft_token=draft_tokens,
                    custom_mask=tree_mask,
                    positions=position,
                    retrive_index=retrive_index,
                    retrive_next_token=retrive_next_token,
                    retrive_next_sibling=retrive_next_sibling,
                    retrive_cum_len=None,
                    spec_steps=self.speculative_num_steps,
                    topk=self.topk,
                    draft_token_num=self.speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=forward_batch.seq_lens_sum,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                )
        # ===========================================================

        # ============= L0 INTEGRATION (zero overhead - always use L0) =============
        if hasattr(self, 'l0_graph_runner') and self.l0_graph_runner is not None and L0_SKIP_EAGLE:
            spec_info_inner = forward_batch.spec_info
            hidden_states = getattr(spec_info_inner, 'hidden_states', None)
            if hidden_states is not None and not self._l0_capturing and not batch.forward_mode.is_idle():
                # Run L0 - no confidence check, no CPU sync
                l0_tokens = self.l0_graph_runner.run(hidden_states)

                # Always use L0 tokens - skip EAGLE entirely
                _l0_global_stats.calls += 1
                _l0_global_stats.l0_skipped_eagle += 1

                (
                    tree_mask,
                    position,
                    retrive_index,
                    retrive_next_token,
                    retrive_next_sibling,
                    draft_tokens,
                ) = create_linear_verify_input(
                    l0_tokens,
                    spec_info.verified_id,
                    batch.seq_lens,
                    batch.seq_lens_sum,
                    self.speculative_num_draft_tokens,
                    self.device,
                )

                return EagleVerifyInput(
                    draft_token=draft_tokens,
                    custom_mask=tree_mask,
                    positions=position,
                    retrive_index=retrive_index,
                    retrive_next_token=retrive_next_token,
                    retrive_next_sibling=retrive_next_sibling,
                    retrive_cum_len=None,
                    spec_steps=self.speculative_num_steps,
                    topk=self.topk,
                    draft_token_num=self.speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=forward_batch.seq_lens_sum,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                )
        # ===========================================================

        # Normal EAGLE path (when L0 not confident or not enabled)
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            forward_batch.can_run_dp_cuda_graph = False
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for idle mode or 1-step draft
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            # Run forward steps
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        # L0 stats logging (periodic, minimal overhead)
        if _l0_global_stats.calls > 0 and _l0_global_stats.calls % 500 == 0:
            skip_rate = _l0_global_stats.l0_skipped_eagle / max(1, _l0_global_stats.calls) * 100
            print(f"[L0] calls={_l0_global_stats.calls}, skip_rate={skip_rate:.1f}%", flush=True)

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.server_args.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        global _l0_global_stats

        # Parse args
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        # Note: L0 integration is handled in eagle_draft() method, not here
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        # TODO: We only need self.speculative_num_steps - 1 cache loc
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            # This is a temporary fix for the case that the user is using standalone
            # speculative decoding and the draft model architecture is gpt-oss. gpt-oss
            # rope kernel needs cache_loc to be contiguous.
            if (
                self.server_args.speculative_algorithm == "STANDALONE"
                and self.model_config.hf_config.architectures[0] == "GptOssForCausalLM"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            if self.server_args.enable_nan_detection:
                detect_nan(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        # Note: L0 stats tracking is done in eagle_draft() method
        return parent_list, top_scores_index, draft_tokens

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker
        pass

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        seq_lens_pre_verify = batch.seq_lens.clone()
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_batch = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()

        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        if self.enable_nan_detection:
            detect_nan(logits_output)

        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
        ):
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        accepted_length = (
            torch.tensor(
                res.accept_length_per_req_cpu,
                device=logits_output.hidden_states.device,
                dtype=torch.int64,
            )
            + 1
        )
        cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
        # prepend 0 to the cumulative_accepted_lengths
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_accepted_lengths.dtype,
                    device=cumulative_accepted_lengths.device,
                ),
                cumulative_accepted_lengths[:-1],
            ]
        )
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        # res.accepted_indices.shape[0] > 0 skips DP attn idle batch
        if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
            # accepted_indices=[0,2,3,4,5,7,9,10,11], accepted_length=[4, 3, 2], cumulative_accepted_lengths=[4, 7, 9]
            # first_token_indices_per_req=prepend(0, accepted_indices[cumulative_accepted_lengths[:-1]]) = [0, 5, 10]
            # last_token_indices_per_req=accepted_indices[cumulative_accepted_lengths - 1] = [4, 9, 11] (last token ID of each req)
            # max_relative_indices_per_req = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
            # first_token_indices_per_req = res.accepted_indices[accepted_indices_start]
            accepted_steps = (
                res.accepted_indices[cumulative_accepted_lengths - 1]
                - accepted_indices_offset
            )
        else:
            accepted_steps = accepted_length - 1

        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accepted_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,
            )
        else:
            mamba_steps_to_track = None

        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
            num_tokens_per_batch=1,
            num_tokens_for_logprob_per_batch=1,
        )
        batch.return_hidden_states = False
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=seq_lens_cpu
        )
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False
        logits_output = self.draft_model_runner.forward(forward_batch).logits_output
        if self.enable_nan_detection:
            detect_nan(logits_output)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        self.capture_for_decode(logits_output, forward_batch.spec_info)

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        assert isinstance(batch.spec_info, EagleDraftInput)
        # Backup fields that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()
        req_pool_indices_backup = batch.req_pool_indices
        accept_length_backup = batch.spec_info.accept_length
        return_logprob_backup = batch.return_logprob

        input_is_idle = batch.forward_mode.is_idle()

        if not input_is_idle and batch.spec_info.verified_id.numel() == 0:
            batch = batch.copy()
            batch.prepare_for_idle()
            hidden_size = (
                self.model_config.hidden_size * 3
                if self.speculative_algorithm.is_eagle3()
                and self.eagle_use_aux_hidden_state
                else self.model_config.hidden_size
            )
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=hidden_size,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )

        batch.spec_info.num_tokens_per_batch = self.speculative_num_steps + 1
        batch.spec_info.num_tokens_for_logprob_per_batch = 1
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
        batch.forward_mode = (
            ForwardMode.DRAFT_EXTEND
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        batch.return_hidden_states = False
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()

        # Run
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
            forward_batch.spec_info.topk_p, forward_batch.spec_info.topk_index = (
                logits_output.topk_p,
                logits_output.topk_index,
            )
            forward_batch.spec_info.hidden_states = logits_output.hidden_states
        else:
            forward_batch.can_run_dp_cuda_graph = False
            if not forward_batch.forward_mode.is_idle():
                self.draft_model_runner.attn_backend.init_forward_metadata(
                    forward_batch
                )
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            self.capture_for_decode(logits_output, forward_batch.spec_info)

        if self.enable_nan_detection:
            detect_nan(logits_output)

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = (
            ForwardMode.DECODE if not input_is_idle else ForwardMode.IDLE
        )
        batch.seq_lens = seq_lens_backup
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.spec_info.accept_length = accept_length_backup
        batch.return_logprob = return_logprob_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message


@torch.compile(dynamic=True, disable=_is_npu)
def get_last_loc_large_page_size_top_k_1(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens,
    speculative_num_steps: int,
):
    prefix_lens = seq_lens
    seq_lens = prefix_lens + speculative_num_steps
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )
    return prefix_lens, seq_lens, last_loc
