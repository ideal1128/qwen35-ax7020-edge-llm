#!/usr/bin/env python3
r"""
export_qwen35_weights_bfp8_64k.py - Offline 24-Layer BFP8 / BF16 Weight Packer for Qwen3.5-0.8B
Target: Zynq-7020 (ALINX AX7020, 1GB DDR3)
DDR Base Address: 0x1000_0000 (safely above ARM kernel / UG585 mapping)

Output:
  - D:\fpga2026\work\qwen35_weights_64k.bin (Contiguous DDR3 image)
  - D:\fpga2026\work\qwen35_weights_64k_manifest.json (DDR3 memory map offset table)
"""

import os
import sys
import json
import struct
import time
import numpy as np
import safetensors.torch as st
import torch

from bfp8_packer import pack_matrix_soa, dequantize_block64

MODEL_DIR = r"D:\fpga2026\model\qwen3.5_0.8b"
VOCAB_DIR = r"D:\fpga2026\work\vocab"
WORK_DIR  = r"D:\fpga2026\work"
OUT_BIN   = os.path.join(WORK_DIR, "qwen35_weights_64k.bin")
OUT_JSON  = os.path.join(WORK_DIR, "qwen35_weights_64k_manifest.json")

DDR3_BASE_ADDR = 0x10000000  # 256MB boundary, UG585 safe window

def align64(n: int) -> int:
    return (n + 63) & ~63

def export_weights():
    print("=" * 75)
    print("  Qwen3.5-0.8B 24-Layer BFP8 / BF16 Weight Exporter (64K Vocab)")
    print("=" * 75)

    # 1. Load 64K mapping
    map_path = os.path.join(VOCAB_DIR, "new_to_old_64k.json")
    print(f"[*] Loading 64K vocabulary map: {map_path}")
    with open(map_path, "r", encoding="utf-8") as f:
        new_to_old = json.load(f)
    assert len(new_to_old) == 65536

    # 2. Open model.safetensors
    st_path = os.path.join(MODEL_DIR, "model.safetensors")
    print(f"[*] Reading tensors from: {st_path}")
    st_file = st.safe_open(st_path, framework="pt", device="cpu")

    manifest = {
        "model_name": "Qwen3.5-0.8B",
        "vocab_size": 65536,
        "num_layers": 24,
        "hidden_size": 1024,
        "ddr3_base_addr": hex(DDR3_BASE_ADDR),
        "tensors": {}
    }

    # Open binary for streaming write
    out_f = open(OUT_BIN, "wb")
    current_offset = 0

    def write_tensor_data(name: str, tensor_bytes: bytes, tensor_meta: dict):
        nonlocal current_offset
        # Ensure 64-byte alignment
        pad = align64(current_offset) - current_offset
        if pad > 0:
            out_f.write(b"\x00" * pad)
            current_offset += pad

        file_offset = current_offset
        phys_addr = DDR3_BASE_ADDR + file_offset
        out_f.write(tensor_bytes)
        current_offset += len(tensor_bytes)

        tensor_meta["file_offset"] = file_offset
        tensor_meta["phys_addr"] = hex(phys_addr)
        tensor_meta["byte_size"] = len(tensor_bytes)
        manifest["tensors"][name] = tensor_meta

    t0 = time.time()

    # --- 1. Sliced Embedding Table (65536 x 1024, BF16, 128 MB) ---
    print("\n[*] Processing Sliced Embedding Table (65536 x 1024)...")
    orig_embed = st_file.get_tensor("model.language_model.embed_tokens.weight")
    sliced_embed = orig_embed[new_to_old].to(torch.bfloat16).contiguous()
    embed_bytes = sliced_embed.view(torch.uint8).numpy().tobytes()
    write_tensor_data(
        "model.language_model.embed_tokens.weight",
        embed_bytes,
        {"dtype": "bfloat16", "shape": [65536, 1024], "type": "raw_bf16"}
    )
    print(f"    -> Exported: {len(embed_bytes)/(1024*1024):.2f} MB at file offset 0x{manifest['tensors']['model.language_model.embed_tokens.weight']['file_offset']:08X}")

    # --- 2. Final LayerNorm (1024, FP32) ---
    print("[*] Processing Final LayerNorm...")
    norm_w = st_file.get_tensor("model.language_model.norm.weight").to(torch.float32).contiguous()
    norm_bytes = norm_w.numpy().tobytes()
    write_tensor_data(
        "model.language_model.norm.weight",
        norm_bytes,
        {"dtype": "float32", "shape": [1024], "type": "raw_fp32"}
    )

    # --- 3. Process All 24 Backbone Layers ---
    print("\n[*] Processing 24 Transformer / DeltaNet Backbone Layers...")
    for layer_idx in range(24):
        layer_prefix = f"model.language_model.layers.{layer_idx}."
        is_full_attn = (layer_idx % 4 == 3) # Layers 3, 7, 11, 15, 19, 23
        layer_type_str = "FullAttention (GQA)" if is_full_attn else "LinearAttention (Gated DeltaNet)"
        print(f"  Layer {layer_idx:2d}/23 [{layer_type_str}]...", end="", flush=True)

        # A. Input LayerNorm
        in_norm = st_file.get_tensor(f"{layer_prefix}input_layernorm.weight").to(torch.float32)
        write_tensor_data(
            f"{layer_prefix}input_layernorm.weight",
            in_norm.numpy().tobytes(),
            {"dtype": "float32", "shape": list(in_norm.shape), "type": "raw_fp32"}
        )

        # B. Post Attention LayerNorm
        post_norm = st_file.get_tensor(f"{layer_prefix}post_attention_layernorm.weight").to(torch.float32)
        write_tensor_data(
            f"{layer_prefix}post_attention_layernorm.weight",
            post_norm.numpy().tobytes(),
            {"dtype": "float32", "shape": list(post_norm.shape), "type": "raw_fp32"}
        )

        if not is_full_attn:
            # Linear Attention Tensors
            # Small vectors & conv1d
            conv1d = st_file.get_tensor(f"{layer_prefix}linear_attn.conv1d.weight").to(torch.float32)
            write_tensor_data(f"{layer_prefix}linear_attn.conv1d.weight", conv1d.numpy().tobytes(),
                              {"dtype": "float32", "shape": list(conv1d.shape), "type": "raw_fp32"})

            a_log = st_file.get_tensor(f"{layer_prefix}linear_attn.A_log").to(torch.float32)
            write_tensor_data(f"{layer_prefix}linear_attn.A_log", a_log.numpy().tobytes(),
                              {"dtype": "float32", "shape": list(a_log.shape), "type": "raw_fp32"})

            dt_bias = st_file.get_tensor(f"{layer_prefix}linear_attn.dt_bias").to(torch.float32)
            write_tensor_data(f"{layer_prefix}linear_attn.dt_bias", dt_bias.numpy().tobytes(),
                              {"dtype": "float32", "shape": list(dt_bias.shape), "type": "raw_fp32"})

            attn_norm = st_file.get_tensor(f"{layer_prefix}linear_attn.norm.weight").to(torch.float32)
            write_tensor_data(f"{layer_prefix}linear_attn.norm.weight", attn_norm.numpy().tobytes(),
                              {"dtype": "float32", "shape": list(attn_norm.shape), "type": "raw_fp32"})

            # in_proj_a & in_proj_b: [16, 1024]
            in_a = st_file.get_tensor(f"{layer_prefix}linear_attn.in_proj_a.weight").to(torch.float32)
            write_tensor_data(f"{layer_prefix}linear_attn.in_proj_a.weight", in_a.numpy().tobytes(),
                              {"dtype": "float32", "shape": list(in_a.shape), "type": "raw_fp32"})

            in_b = st_file.get_tensor(f"{layer_prefix}linear_attn.in_proj_b.weight").to(torch.float32)
            write_tensor_data(f"{layer_prefix}linear_attn.in_proj_b.weight", in_b.numpy().tobytes(),
                              {"dtype": "float32", "shape": list(in_b.shape), "type": "raw_fp32"})

            # Large matrices quantized to BFP8: in_proj_qkv [6144, 1024], in_proj_z [2048, 1024], out_proj [1024, 2048]
            for mat_name, t_name in [("in_proj_qkv", f"{layer_prefix}linear_attn.in_proj_qkv.weight"),
                                     ("in_proj_z",   f"{layer_prefix}linear_attn.in_proj_z.weight"),
                                     ("out_proj",    f"{layer_prefix}linear_attn.out_proj.weight")]:
                mat = st_file.get_tensor(t_name).to(torch.float32).numpy()
                m_bytes, e_bytes = pack_matrix_soa(mat)
                combined_bfp8 = m_bytes + e_bytes
                write_tensor_data(
                    t_name, combined_bfp8,
                    {
                        "dtype": "bfp8_soa",
                        "shape": list(mat.shape),
                        "type": "block64_bfp8",
                        "mantissa_bytes": len(m_bytes),
                        "exponent_bytes": len(e_bytes)
                    }
                )
        else:
            # Full Attention Tensors
            q_norm = st_file.get_tensor(f"{layer_prefix}self_attn.q_norm.weight").to(torch.float32)
            write_tensor_data(f"{layer_prefix}self_attn.q_norm.weight", q_norm.numpy().tobytes(),
                              {"dtype": "float32", "shape": list(q_norm.shape), "type": "raw_fp32"})

            k_norm = st_file.get_tensor(f"{layer_prefix}self_attn.k_norm.weight").to(torch.float32)
            write_tensor_data(f"{layer_prefix}self_attn.k_norm.weight", k_norm.numpy().tobytes(),
                              {"dtype": "float32", "shape": list(k_norm.shape), "type": "raw_fp32"})

            for mat_name, t_name in [("q_proj", f"{layer_prefix}self_attn.q_proj.weight"),
                                     ("k_proj", f"{layer_prefix}self_attn.k_proj.weight"),
                                     ("v_proj", f"{layer_prefix}self_attn.v_proj.weight"),
                                     ("o_proj", f"{layer_prefix}self_attn.o_proj.weight")]:
                mat = st_file.get_tensor(t_name).to(torch.float32).numpy()
                m_bytes, e_bytes = pack_matrix_soa(mat)
                combined_bfp8 = m_bytes + e_bytes
                write_tensor_data(
                    t_name, combined_bfp8,
                    {
                        "dtype": "bfp8_soa",
                        "shape": list(mat.shape),
                        "type": "block64_bfp8",
                        "mantissa_bytes": len(m_bytes),
                        "exponent_bytes": len(e_bytes)
                    }
                )

        # C. MLP Projections (Shared across all layers: gate_proj, up_proj, down_proj)
        for mlp_mat in ["gate_proj", "up_proj", "down_proj"]:
            t_name = f"{layer_prefix}mlp.{mlp_mat}.weight"
            mat = st_file.get_tensor(t_name).to(torch.float32).numpy()
            m_bytes, e_bytes = pack_matrix_soa(mat)
            combined_bfp8 = m_bytes + e_bytes
            write_tensor_data(
                t_name, combined_bfp8,
                {
                    "dtype": "bfp8_soa",
                    "shape": list(mat.shape),
                    "type": "block64_bfp8",
                    "mantissa_bytes": len(m_bytes),
                    "exponent_bytes": len(e_bytes)
                }
            )

        print(" Done!")

    # Final padding to align file size
    final_pad = align64(current_offset) - current_offset
    if final_pad > 0:
        out_f.write(b"\x00" * final_pad)
        current_offset += final_pad

    out_f.close()
    t1 = time.time()

    manifest["total_binary_bytes"] = current_offset
    manifest["total_binary_mb"] = round(current_offset / (1024 * 1024), 2)
    manifest["total_tensors"] = len(manifest["tensors"])

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 75)
    print(f"  [SUCCESS] Exported {len(manifest['tensors'])} tensors in {t1 - t0:.2f} seconds!")
    print(f"  Binary Image:  {OUT_BIN} ({manifest['total_binary_mb']} MB)")
    print(f"  Manifest Map:  {OUT_JSON}")
    print(f"  DDR3 Window:   {manifest['ddr3_base_addr']} ~ {hex(DDR3_BASE_ADDR + current_offset)}")
    print("=" * 75)

if __name__ == "__main__":
    export_weights()
