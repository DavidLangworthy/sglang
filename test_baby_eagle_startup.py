#!/usr/bin/env python3
"""
Quick test to verify Baby EAGLE server starts without TypeError.
Tests the fix for spec_num_draft_tokens initialization.
"""
import sys
import os

# Add sglang to path
sys.path.insert(0, "/workspace/lager/priors/sglang-fork/python")

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

def test_baby_eagle_args():
    """Test that Baby EAGLE properly initializes spec_num_draft_tokens."""

    # Create server args with Baby EAGLE
    args = ServerArgs(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        speculative_algorithm="BABY_EAGLE",
        speculative_draft_model_path="/workspace/lager/projects/baby_eagle/checkpoints_v2/best_sglang.pt",
        speculative_num_steps=3,
        dtype="float16",
        device="cuda",
    )

    # Check that initialization happened correctly
    print(f"speculative_algorithm: {args.speculative_algorithm}")
    print(f"speculative_num_steps: {args.speculative_num_steps}")
    print(f"speculative_num_draft_tokens: {args.speculative_num_draft_tokens}")
    print(f"speculative_eagle_topk: {args.speculative_eagle_topk}")

    # These should all be set now
    assert args.speculative_num_draft_tokens is not None, "spec_num_draft_tokens should be set"
    assert args.speculative_num_draft_tokens == 3, f"Expected 3, got {args.speculative_num_draft_tokens}"
    assert args.speculative_eagle_topk == 1, f"Expected topk=1, got {args.speculative_eagle_topk}"

    print("\n✓ Baby EAGLE args initialized correctly!")
    print("✓ spec_num_draft_tokens = 3 (from spec_num_steps)")
    print("✓ speculative_eagle_topk = 1 (linear drafting)")

    return True

if __name__ == "__main__":
    try:
        test_baby_eagle_args()
        print("\nSUCCESS: TypeError fix verified!")
        sys.exit(0)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
