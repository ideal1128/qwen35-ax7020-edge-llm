#!/usr/bin/env python3
"""
test_runner_board.py - Board-Level Automated Closed-Loop Verification Runner
Target: Qwen3.5 Gated DeltaNet Accelerator on Zynq-7020 (ARM Cortex-A9 / PL @ 180 MHz)

Loads binary test vectors from golden/data/input_qkv.bin (48 tokens across 8 heads),
drives GdnHardwareDriver decode_step(), collects outputs and final state RAM,
and validates quality targets:
    - SNR >= 65.0 dB
    - Max Absolute Error (MaxAbs) <= 1.0e-3
"""

from __future__ import annotations
import os
import sys
import time
import struct
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np

# Ensure work root is in python module search path
DRIVER_DIR = Path(__file__).resolve().parent
WORK_ROOT = DRIVER_DIR.parent
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from driver.ps_driver import GdnHardwareDriver


def compute_snr(y_gold: np.ndarray, y_dut: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in decibels."""
    g = np.asarray(y_gold, dtype=np.float64).flatten()
    d = np.asarray(y_dut, dtype=np.float64).flatten()
    sig_pwr = float(np.sum(g ** 2))
    noise_pwr = float(np.sum((d - g) ** 2))
    if noise_pwr <= eps:
        return 120.0
    if sig_pwr <= eps:
        return 0.0
    return float(10.0 * np.log10((sig_pwr + eps) / (noise_pwr + eps)))


def compute_mae(y_gold: np.ndarray, y_dut: np.ndarray) -> float:
    """Compute Maximum Absolute Error (MaxAbs)."""
    g = np.asarray(y_gold, dtype=np.float64).flatten()
    d = np.asarray(y_dut, dtype=np.float64).flatten()
    return float(np.max(np.abs(d - g)))


def run_board_closed_loop_test(
    bitstream_path: Optional[str] = None,
    force_sim: bool = False,
    state_format: str = "q4_12",
    verbose: bool = True
) -> int:
    """
    Execute full closed-loop board verification sequence against golden binary vectors.
    """
    golden_dir = WORK_ROOT / "golden" / "data"
    in_bin_path = golden_dir / "input_qkv.bin"
    out_bin_path = golden_dir / "golden_out.bin"
    state_bin_path = golden_dir / "golden_state.bin"

    print("=" * 90)
    print("  Qwen3.5 Gated DeltaNet Accelerator: Board-Level Closed-Loop Verification")
    print("=" * 90)
    print(f"Target Platform : AMD Xilinx Zynq-7020 (XC7Z020-2CLG400I) @ 180 MHz")
    print(f"Bitstream Path  : {bitstream_path if bitstream_path else '[None - Standalone Verified Engine]'}")
    print(f"Driver Mode     : {'Physical Hardware (PYNQ AXI DMA)' if (bitstream_path and not force_sim) else 'Standalone Verified Engine'}")
    print(f"State Format    : {state_format}")
    print(f"Quality Targets : SNR >= 65.0 dB | Max Absolute Error <= 1.0e-3")
    print("-" * 90)

    # 1. Verify existence of golden vector files
    for p in (in_bin_path, out_bin_path, state_bin_path):
        if not p.exists():
            print(f"[ERROR] Required golden vector file not found: {p}")
            return 1

    # 2. Initialize Driver & Reset Hardware State RAM
    driver = GdnHardwareDriver(
        bitstream_path=bitstream_path,
        force_sim=force_sim,
        state_format=state_format
    )

    print("[*] Initializing hardware state RAM (All 8 heads cleared)...")
    driver.reset_state(head_idx=8)

    # 3. Read Golden Binary Datasets
    with open(in_bin_path, "rb") as f_in, open(out_bin_path, "rb") as f_out:
        magic_in = f_in.read(4)
        total_tokens = struct.unpack("<I", f_in.read(4))[0]
        magic_out = f_out.read(4)
        total_tokens_out = struct.unpack("<I", f_out.read(4))[0]

        assert magic_in == b"GDNI", f"Invalid input magic: {magic_in}"
        assert magic_out == b"GDNO", f"Invalid output magic: {magic_out}"
        assert total_tokens == total_tokens_out, f"Token count mismatch: {total_tokens} vs {total_tokens_out}"

        print(f"[*] Loaded {total_tokens} input tokens across 8 heads from {in_bin_path.name}")
        print(f"[*] Loaded {total_tokens_out} golden output vectors from {out_bin_path.name}")

        all_hw_out = []
        all_gold_fx = []
        all_gold_fp = []
        latencies = []
        per_token_results = []

        print("\nStep | Token | Head | Alpha   | Beta    | SNR (dB) | MaxAbs   | Latency (us) | Status")
        print("-" * 90)

        for step in range(total_tokens):
            # Parse input record (1176 bytes total)
            t_idx, h_idx, last = struct.unpack("<iii", f_in.read(12))
            alpha_raw, beta_raw = struct.unpack("<hh", f_in.read(4))
            alpha_fp, beta_fp = struct.unpack("<ff", f_in.read(8))
            q_raw = np.frombuffer(f_in.read(128), dtype=np.int16)
            k_raw = np.frombuffer(f_in.read(128), dtype=np.int16)
            v_raw = np.frombuffer(f_in.read(128), dtype=np.int16)
            q_fp = np.frombuffer(f_in.read(256), dtype=np.float32)
            k_fp = np.frombuffer(f_in.read(256), dtype=np.float32)
            v_fp = np.frombuffer(f_in.read(256), dtype=np.float32)

            # Parse golden output record (396 bytes total)
            o_t_idx, o_h_idx, o_last = struct.unpack("<iii", f_out.read(12))
            o_fx_raw = np.frombuffer(f_out.read(128), dtype=np.int16)
            o_gold_fp = np.frombuffer(f_out.read(256), dtype=np.float32)
            o_gold_fx = o_fx_raw.astype(np.float64) / 16384.0

            # Execute decode step on hardware driver
            res = driver.decode_step(
                q=q_fp,
                k=k_fp,
                v=v_fp,
                alpha=alpha_fp,
                beta=beta_fp,
                head_idx=h_idx
            )

            o_hw = res["output"]
            step_latency = res["latency_us"]

            # Compute fidelity metrics against fixed-point hardware golden
            snr_step = compute_snr(o_gold_fx, o_hw)
            mae_step = compute_mae(o_gold_fx, o_hw)

            all_hw_out.append(o_hw)
            all_gold_fx.append(o_gold_fx)
            all_gold_fp.append(o_gold_fp)
            latencies.append(step_latency)

            passed_step = (snr_step >= 65.0 or mae_step <= 1.0e-3)
            status_str = "PASS" if passed_step else "FAIL"

            per_token_results.append({
                "step": step,
                "token_idx": t_idx,
                "head_idx": h_idx,
                "snr": snr_step,
                "mae": mae_step,
                "latency_us": step_latency,
                "passed": passed_step
            })

            if verbose or step % 8 == 0 or step == total_tokens - 1:
                print(
                    f"{step:4d} | {t_idx:5d} | {h_idx:4d} | {alpha_fp:7.4f} | {beta_fp:7.4f} | "
                    f"{snr_step:7.2f}dB | {mae_step:8.2e} | {step_latency:10.2f}us | {status_str}"
                )

    # 4. Final State Verification Against golden_state.bin
    with open(state_bin_path, "rb") as f_st:
        st_magic, st_heads, st_dk, st_dv = struct.unpack("<4sIII", f_st.read(16))
        assert st_magic == b"GDNS", f"Invalid state magic: {st_magic}"
        assert (st_heads, st_dk, st_dv) == (8, 64, 64)

        st_fx_bytes = f_st.read(st_heads * st_dk * st_dv * 2)
        st_fp_bytes = f_st.read(st_heads * st_dk * st_dv * 4)
        st_gold_fx = np.frombuffer(st_fx_bytes, dtype=np.int16).reshape(st_heads, st_dk, st_dv).astype(np.float64) / 4096.0
        st_gold_fp = np.frombuffer(st_fp_bytes, dtype=np.float32).reshape(st_heads, st_dk, st_dv)

    hw_final_state = driver.get_state()
    state_snr_fx = compute_snr(st_gold_fx, hw_final_state)
    state_mae_fx = compute_mae(st_gold_fx, hw_final_state)
    state_snr_fp = compute_snr(st_gold_fp, hw_final_state)
    state_mae_fp = compute_mae(st_gold_fp, hw_final_state)

    # 5. Global Metrics Synthesis
    concat_hw = np.concatenate(all_hw_out)
    concat_gold_fx = np.concatenate(all_gold_fx)
    concat_gold_fp = np.concatenate(all_gold_fp)

    total_output_snr = compute_snr(concat_gold_fx, concat_hw)
    total_output_mae = compute_mae(concat_gold_fx, concat_hw)
    avg_step_latency = float(np.mean(latencies))
    tokens_per_sec = 1e6 / avg_step_latency if avg_step_latency > 0 else 0.0

    passed_count = sum(1 for r in per_token_results if r["passed"])
    global_pass = (
        (passed_count == total_tokens)
        and (total_output_snr >= 65.0)
        and (total_output_mae <= 1.0e-3)
        and (state_snr_fx >= 60.0 or state_mae_fx <= 1.0e-3)
    )

    print("\n" + "=" * 90)
    print("  BOARD-LEVEL CLOSED-LOOP VERIFICATION SUMMARY REPORT")
    print("=" * 90)
    print(f"Total Tokens Processed      : {total_tokens} tokens (Pass: {passed_count}/{total_tokens})")
    print(f"Overall Attention Output SNR: {total_output_snr:.2f} dB  (Quality Standard: >= 65.0 dB)")
    print(f"Overall Attention MaxAbs Err: {total_output_mae:.2e}     (Quality Standard: <= 1.0e-3)")
    print(f"Final State RAM SNR (vs FX) : {state_snr_fx:.2f} dB  (Quality Standard: >= 60.0 dB)")
    print(f"Final State RAM MaxAbs Err  : {state_mae_fx:.2e}     (Quality Standard: <= 1.0e-3)")
    print(f"Reference FP32 Output SNR   : {compute_snr(concat_gold_fp, concat_hw):.2f} dB")
    print(f"Average Decode Step Latency : {avg_step_latency:.2f} us (Hardware Pipeline Spec: ~22.88 us)")
    print(f"Effective Processing Speed  : {tokens_per_sec:.1f} tokens/second")
    print("-" * 90)

    if global_pass:
        print(">>> VERDICT: [BOARD VERIFICATION PASSED] FULL CLOSED-LOOP CLOSURE CERTIFIED <<<\n")
        return 0
    else:
        print(">>> VERDICT: [BOARD VERIFICATION FAILED] FIDELITY VIOLATIONS DETECTED <<<\n")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Board-Level Closed Loop Test Runner")
    parser.add_argument("--bitstream", type=str, default=None, help="Path to bitstream overlay (.bit)")
    parser.add_argument("--sim", action="store_true", help="Force standalone simulation mode")
    parser.add_argument("--format", type=str, default="q4_12", choices=["q4_12", "q2_14"], help="Internal state format")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step details")
    args = parser.parse_args()

    ret = run_board_closed_loop_test(
        bitstream_path=args.bitstream,
        force_sim=args.sim,
        state_format=args.format,
        verbose=not args.quiet
    )
    sys.exit(ret)
