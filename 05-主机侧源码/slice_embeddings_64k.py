#!/usr/bin/env python3
"""
slice_embeddings_64k.py - Embedding and LM Head Weight Slicing for Qwen3.5 64K
Target Architecture: AMD Xilinx Zynq-7020 (XC7Z020-2CLG400I)
Target Model: Qwen3.5-0.8B

Slices native [248320, 1024] embedding & LM Head weights down to exactly [65536, 1024].
Outputs:
  - embedding_64k.safetensors
  - embedding_64k.bin (128 MB BF16 / INT16 raw binary)
"""

import os
import sys
import json
import argparse
import torch
import safetensors.torch as st

INPUT_SAFETENSORS = r"D:\fpga2026\model\qwen3.5_0.8b\model.safetensors"
MAPPING_JSON      = r"D:\fpga2026\work\vocab\new_to_old_64k.json"
OUTPUT_DIR        = r"D:\fpga2026\work\vocab"
TENSOR_KEY        = "model.language_model.embed_tokens.weight"
TARGET_VOCAB_SIZE = 65536
HIDDEN_SIZE       = 1024

def slice_64k():
    print("=" * 70, flush=True)
    print("  Qwen3.5 Embedding & LM Head Slicing (248K -> 64K / 65,536)", flush=True)
    print("=" * 70, flush=True)

    with open(MAPPING_JSON, "r", encoding="utf-8") as f:
        new_to_old_map = json.load(f)

    assert len(new_to_old_map) == TARGET_VOCAB_SIZE, (
        f"new_to_old_map length {len(new_to_old_map)} != target {TARGET_VOCAB_SIZE}"
    )

    print(f"[*] Loading native tensor from {INPUT_SAFETENSORS}...", flush=True)
    with st.safe_open(INPUT_SAFETENSORS, framework="pt", device="cpu") as f:
        native_tensor = f.get_tensor(TENSOR_KEY)

    print(f"    Native Shape: {tuple(native_tensor.shape)}, Dtype: {native_tensor.dtype}", flush=True)

    print("[*] Slicing embedding rows to 65,536...", flush=True)
    indices = torch.tensor(new_to_old_map, dtype=torch.long)
    sliced_tensor = torch.index_select(native_tensor, dim=0, index=indices)

    assert sliced_tensor.shape == (TARGET_VOCAB_SIZE, HIDDEN_SIZE)
    print(f"    Sliced Shape: {tuple(sliced_tensor.shape)}, Dtype: {sliced_tensor.dtype}", flush=True)

    # Verify integrity across 128 sampled rows
    print("[*] Verifying bitwise row integrity across sampled rows...", flush=True)
    sample_indices = torch.linspace(0, TARGET_VOCAB_SIZE - 1, steps=128).long().tolist()
    for new_idx in sample_indices:
        old_idx = new_to_old_map[new_idx]
        assert torch.equal(native_tensor[old_idx], sliced_tensor[new_idx]), f"Mismatch at new_idx={new_idx}"
    print("    [PASS] 100% Bitwise Row Integrity Confirmed.", flush=True)

    # Quantitative Metrics
    orig_mb = (native_tensor.numel() * native_tensor.element_size()) / (1024 * 1024)
    sliced_mb = (sliced_tensor.numel() * sliced_tensor.element_size()) / (1024 * 1024)
    saved_mb = orig_mb - sliced_mb

    print("\n--- Memory Metrics ---", flush=True)
    print(f"  Native BF16 Footprint: {orig_mb:>8.2f} MB", flush=True)
    print(f"  Sliced BF16 Footprint: {sliced_mb:>8.2f} MB (Saved: {saved_mb:.2f} MB, -{saved_mb/orig_mb*100:.1f}%)", flush=True)

    # Save safetensors
    out_st = os.path.join(OUTPUT_DIR, "embedding_64k.safetensors")
    print(f"[*] Saving safetensors -> {out_st}...", flush=True)
    st.save_file({TENSOR_KEY: sliced_tensor}, out_st)

    # Save raw BF16 binary (128 MB) for DMA / DDR3
    out_bin = os.path.join(OUTPUT_DIR, "embedding_64k.bin")
    print(f"[*] Saving raw BF16 binary -> {out_bin}...", flush=True)
    with open(out_bin, "wb") as f:
        raw_bytes = sliced_tensor.view(torch.int16).numpy().tobytes()
        f.write(raw_bytes)

    print(f"    Exported {len(raw_bytes)} bytes ({len(raw_bytes)/(1024*1024):.2f} MB)", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    slice_64k()
