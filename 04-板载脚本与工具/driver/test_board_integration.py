#!/usr/bin/env python3
"""
test_board_integration.py - Integration Test Runner between PS Hardware Driver & 4-Tier Test Suite
Target: Qwen3.5 Gated DeltaNet Accelerator on Zynq-7020 (ARM Cortex-A9 / PL @ 180 MHz)

Binds GdnHardwareDriver to the 4-Tier verification test harness (tests/run_all_tests.py),
executing all 16 test suites across 99 test cases directly through the hardware driver pipeline.
Validates:
    - 100% pass rate across all 99 test vectors (16 suites, Tiers 1-4)
    - Average SNR >= 65.0 dB
    - Max Absolute Error (MaxAbs) <= 1.0e-3
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Ensure work root is in python module search path
DRIVER_DIR = Path(__file__).resolve().parent
WORK_ROOT = DRIVER_DIR.parent
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from driver.ps_driver import GdnHardwareDriver
from tests.vector_types import GdnStepOutput, SuiteResult, TestCaseResult
from tests.test_cases.base_test import BaseGdnTestCase
from tests.run_all_tests import SUITE_CATALOG, run_suite, format_table


class GdnHardwareEngineAdapter:
    """
    Adapter bridging GdnHardwareDriver to the interface expected by BaseGdnTestCase
    and the 4-Tier verification test suites.
    """

    def __init__(self, driver: Optional[GdnHardwareDriver] = None, bitstream_path: Optional[str] = None):
        if driver is not None:
            self.driver = driver
        else:
            self.driver = GdnHardwareDriver(
                bitstream_path=bitstream_path,
                state_format="q2_14"  # Uses DSP48 wide accumulator precision matching 4-Tier test harness
            )
        self.d_k = self.driver.D_K
        self.d_v = self.driver.D_V
        self.num_heads = self.driver.NUM_HEADS

    def reset_state(self, head_idx: Optional[int] = None) -> None:
        """Reset state RAM for specific head or all heads."""
        self.driver.reset_state(head_idx)

    def set_state_float(self, S: np.ndarray, head_idx: int = 0) -> None:
        """Set state matrix for head_idx."""
        self.driver.set_state(S, head_idx=head_idx)

    def get_state_float(self, head_idx: int = 0) -> np.ndarray:
        """Retrieve copy of state matrix for head_idx."""
        return self.driver.get_state(head_idx=head_idx)

    def step(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        alpha: float,
        beta: float,
        head_idx: int = 0,
        mode: int = 0
    ) -> GdnStepOutput:
        """
        Execute single-step recurrence via hardware driver and wrap in GdnStepOutput.
        """
        if mode == 1:
            self.driver.reset_state(head_idx)

        res = self.driver.decode_step(
            q=q,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
            head_idx=head_idx
        )

        return GdnStepOutput(
            u=res["u"],
            e=res["e"],
            o=res["output"],
            S=res["state"]
        )

    def forward_sequence(
        self,
        q_seq: np.ndarray,
        k_seq: np.ndarray,
        v_seq: np.ndarray,
        alpha_seq: np.ndarray,
        beta_seq: np.ndarray,
        head_idx: int = 0
    ) -> List[GdnStepOutput]:
        """Process sequence of tokens sequentially through driver."""
        T = len(q_seq)
        outputs = []
        for t in range(T):
            out = self.step(
                q=q_seq[t],
                k=k_seq[t],
                v=v_seq[t],
                alpha=float(alpha_seq[t]),
                beta=float(beta_seq[t]),
                head_idx=head_idx,
                mode=0
            )
            outputs.append(out)
        return outputs


def run_board_integration_tests(
    bitstream_path: Optional[str] = None,
    tier_filter: Optional[int] = None,
    verbose: bool = False
) -> int:
    """
    Execute 4-Tier E2E test suite integrated with GdnHardwareDriver.
    """
    print("=" * 90)
    print("  Qwen3.5 Linear Attention Engine: Hardware Driver 4-Tier Integration Suite")
    print("=" * 90)
    print(f"Target Hardware : AMD Xilinx XC7Z020-2CLG400I (Zynq-7020 @ 180 MHz)")
    print(f"Driver Engine   : GdnHardwareDriver (AXI4-Lite + AXI DMA Interface)")
    print(f"Bitstream Path  : {bitstream_path if bitstream_path else '[None - Standalone Verified Engine]'}")
    print(f"Selected Tiers  : {f'Tier {tier_filter}' if tier_filter else 'ALL (Tiers 1-4)'}")
    print(f"Quality Targets : SNR >= 65.0 dB | Max Absolute Error <= 1.0e-3")
    print("-" * 90)

    # Patch BaseGdnTestCase.setUp to use GdnHardwareEngineAdapter
    orig_setUp = BaseGdnTestCase.setUp

    def patched_setUp(self_tc: BaseGdnTestCase) -> None:
        orig_setUp(self_tc)
        # Bind the test case's DUT engine directly to our hardware driver adapter
        self_tc.fixed_engine = GdnHardwareEngineAdapter(bitstream_path=bitstream_path)

    BaseGdnTestCase.setUp = patched_setUp

    # Filter suites if tier specified
    if tier_filter is not None:
        suites_to_run = [(t, name, cls) for t, name, cls in SUITE_CATALOG if t == tier_filter]
    else:
        suites_to_run = SUITE_CATALOG

    suite_results: List[SuiteResult] = []
    t_start = time.perf_counter()

    for tier_num, suite_name, test_class in suites_to_run:
        result = run_suite(tier_num, suite_name, test_class, verbose=verbose)
        suite_results.append(result)

    t_elapsed = time.perf_counter() - t_start

    # Format and print results
    print("\n" + format_table(suite_results))

    total_vectors = sum(s.total_count for s in suite_results)
    total_passed = sum(s.passed_count for s in suite_results)
    total_failed = sum(s.failed_count for s in suite_results)

    all_test_records: List[TestCaseResult] = []
    for s in suite_results:
        all_test_records.extend(s.test_results)

    snr_list = [r.snr_db for r in all_test_records if r.passed]
    avg_snr = float(np.mean(snr_list)) if snr_list else 0.0
    min_snr = float(np.min(snr_list)) if snr_list else 0.0
    max_mae = float(np.max([r.mae for r in all_test_records])) if all_test_records else 0.0

    print("\nIntegration Execution Summary (Elapsed: {:.2f}s):".format(t_elapsed))
    print(f"  Total Vectors Evaluated : {total_vectors}")
    print(f"  Passed                  : {total_passed} / {total_vectors} ({100.0 * total_passed / max(1, total_vectors):.1f}%)")
    print(f"  Failed                  : {total_failed}")
    print(f"  Average SNR             : {avg_snr:.2f} dB (Quality Standard: >= 65.0 dB)")
    print(f"  Minimum SNR (Worst-Case): {min_snr:.2f} dB")
    print(f"  Maximum Absolute Error  : {max_mae:.2e} (Quality Standard: <= 1.0e-3)")

    if total_failed == 0 and total_passed == total_vectors and avg_snr >= 65.0 and max_mae <= 1.0e-3:
        print("\n>>> OVERALL INTEGRATION RESULT: [PASSED] 100% HARDWARE SUITE CERTIFIED <<<\n")
        return 0
    else:
        print("\n>>> OVERALL INTEGRATION RESULT: [FAILED] DEFECTS DETECTED <<<\n")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PS Hardware Driver Integration Test Runner")
    parser.add_argument("--bitstream", type=str, default=None, help="Path to bitstream overlay (.bit)")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=None, help="Filter specific tier")
    parser.add_argument("--verbose", action="store_true", help="Print details for each test case")
    args = parser.parse_args()

    ret = run_board_integration_tests(
        bitstream_path=args.bitstream,
        tier_filter=args.tier,
        verbose=args.verbose
    )
    sys.exit(ret)
