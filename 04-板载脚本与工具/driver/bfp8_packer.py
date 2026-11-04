#!/usr/bin/env python3
"""
bfp8_packer.py - Production-grade Block-64 BFP8 SoA Packer for Qwen3.5 on Zynq-7020
Conforms strictly to DeepSeek-V3 Infra Spec (SPEC-INFRA-2026-DEEPSEEK-V3-ZYNQ7020 §2.1):
  - Block size = 64 elements
  - Mantissa stream: strict 64-byte aligned (8 beats of 64-bit AXI_HP)
  - Exponent stream: 8 exponents packed into one 64-bit AXI word
  - Quantization formula: x_i = M_i * 2^(E - 127 - 7)
"""

from __future__ import annotations
import numpy as np
import struct
from typing import Tuple, Dict, Any
from pathlib import Path


def quantize_block64(vec64: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Quantizes a 64-element float vector to Block-64 BFP8 format.
    Returns:
        mantissas: np.ndarray of shape (64,), dtype np.int8
        exp: int in [0, 255]
    """
    assert len(vec64) == 64, f"Vector length must be 64, got {len(vec64)}"
    arr = np.asarray(vec64, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr)))

    if max_abs < 1e-12:
        return np.zeros(64, dtype=np.int8), 127

    # Find exponent: ceil(log2(max_abs)) so that 2^real_log2 >= max_abs
    log2_val = np.ceil(np.log2(max_abs))
    exp_biased = int(np.clip(log2_val + 127, 0, 255))
    real_log2 = exp_biased - 127

    # Scale: target mantissa peak in [-127, 127] (7 fractional/magnitude bits)
    # x = M * 2^(real_log2 - 7)  ==> M = x * 2^(7 - real_log2)
    scale = 2.0 ** (7 - real_log2)
    scaled = np.round(arr * scale)
    mantissas = np.clip(scaled, -128, 127).astype(np.int8)

    return mantissas, exp_biased


def dequantize_block64(mantissas: np.ndarray, exp_biased: int) -> np.ndarray:
    """Reconstructs float vector from Block-64 BFP8 representation."""
    real_log2 = exp_biased - 127
    scale = 2.0 ** (real_log2 - 7)
    return mantissas.astype(np.float32) * scale


def pack_matrix_soa(matrix: np.ndarray) -> Tuple[bytes, bytes]:
    """
    High-performance vectorized Block-64 BFP8 SoA packer.
    Bit-exact equivalent to elementwise quantize_block64 loop.
    Packs a 2D matrix (rows x cols, where rows*cols is multiple of 64)
    into SoA dual-stream format:
      1. Mantissa stream: contiguous 64-byte blocks
      2. Exponent stream: 8 exponents packed per 64-bit word
    """
    flat = matrix.flatten().astype(np.float32)
    assert len(flat) % 64 == 0, f"Matrix total elements must be multiple of 64, got {len(flat)}"
    blocks = flat.reshape(-1, 64)

    max_abs = np.max(np.abs(blocks), axis=1)
    safe_max = np.maximum(max_abs, 1e-12)
    log2_val = np.ceil(np.log2(safe_max))
    exp_biased = np.clip(log2_val + 127, 0, 255).astype(np.uint8)

    zero_mask = max_abs < 1e-12
    exp_biased[zero_mask] = 127
    real_log2 = exp_biased.astype(np.float32) - 127
    scale = (2.0 ** (7.0 - real_log2))[:, None]

    scaled = np.round(blocks * scale)
    mantissas = np.clip(scaled, -128, 127).astype(np.int8)
    mantissas[zero_mask] = 0

    pad_exp = (8 - (len(exp_biased) % 8)) % 8
    if pad_exp > 0:
        exp_padded = np.pad(exp_biased, (0, pad_exp), constant_values=127)
    else:
        exp_padded = exp_biased

    return mantissas.tobytes(), exp_padded.tobytes()


if __name__ == "__main__":
    # Self-test: SNR of random normal and outlier distribution
    np.random.seed(42)
    test_vec = np.random.randn(64).astype(np.float32)
    test_vec[17] = 25.0

    m, e = quantize_block64(test_vec)
    rec = dequantize_block64(m, e)

    diff = np.abs(test_vec - rec)
    sig_pwr = np.sum(test_vec ** 2)
    noise_pwr = np.sum(diff ** 2)
    snr = 10.0 * np.log10(sig_pwr / noise_pwr)

    print(f"[*] Block-64 BFP8 Quantization Self-Test:")
    print(f"    Peak original: {np.max(np.abs(test_vec)):.2f}, Peak reconstructed: {np.max(np.abs(rec)):.2f}")
    print(f"    Max Absolute Error: {np.max(diff):.4f}")
    print(f"    Signal-to-Noise Ratio: {snr:.2f} dB (High Quality)")
    assert snr > 30.0, f"SNR too low: {snr}"

    # Test SoA Matrix Packing
    test_mat = np.random.randn(128, 64).astype(np.float32)
    m_bytes, e_bytes = pack_matrix_soa(test_mat)
    print(f"[*] Packed Matrix (128x64):")
    print(f"    Mantissa stream: {len(m_bytes)} bytes (Strictly {len(m_bytes)/64:.0f} blocks of 64B)")
    print(f"    Exponent stream: {len(e_bytes)} bytes ({len(e_bytes)//8} 64-bit words)")
    assert len(m_bytes) == 128 * 64
    assert len(e_bytes) % 8 == 0
    print("[SUCCESS] bfp8_packer.py Self-Test Passed 100%!")
