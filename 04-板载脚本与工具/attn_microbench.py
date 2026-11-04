#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attn_microbench.py -- isolate the cost of the full-attention step vs context.

Why a micro-benchmark: driving the whole 24-layer model out to a 544-token
context takes about 8 minutes, because every prefill token costs ~0.9 s.  The
thing that changed is only the 6 full-attention layers, so measuring that one
function directly at several context lengths gives the same answer in seconds
and isolates the effect cleanly.

For each context length T it times:
  * the original numpy path  PagedAttentionPool.compute_attention()
  * the new C+NEON path      libqwen_attn.so full_attn_f32()
and reports both, plus the ratio.  Multiply the difference by 6 to get the
per-token effect on the real model.

ASCII only.  Python 3.4 / numpy 1.8.2 compatible.
"""
import ctypes
import sys
import time

sys.path.insert(0, "/mnt/qwen")

import numpy as np
from qwen_real_engine import PagedAttentionPool

N_LAYERS = 6
N_KV = 2
N_Q = 8
HEAD_DIM = 256
BLOCK = 16
SCALE = 0.0625

REPS = 5


def load_lib():
    lib = ctypes.CDLL("/mnt/qwen/libqwen_attn.so")
    fn = lib.full_attn_f32
    fn.restype = None
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ]
    return fn


def build_pool(total_tokens, rng):
    """Fill a pool so that seq_len + 1 == total_tokens."""
    pool = PagedAttentionPool()
    for i in range(total_tokens):
        pool.allocate_slot()
        k = rng.randn(N_KV, HEAD_DIM).astype(np.float32) * 0.5
        v = rng.randn(N_KV, HEAD_DIM).astype(np.float32) * 0.5
        pool.append_kv(0, k, v)
        pool.step_done()
    pool.seq_len = total_tokens - 1
    return pool


def main():
    print("=== full-attention micro-benchmark (one layer, %d reps) ===" % REPS)
    fn = load_lib()
    rng = np.random.RandomState(1234)

    bt = np.zeros(128, dtype=np.int32)
    out = np.zeros(N_Q * HEAD_DIM, dtype=np.float32)

    print("%6s %12s %12s %9s %10s" %
          ("T", "numpy ms", "C+NEON ms", "speedup", "saved/layer"))

    rows = []
    for T in (20, 100, 300, 544, 1000, 2000):
        pool = build_pool(T, rng)
        q = rng.randn(N_Q, HEAD_DIM).astype(np.float32)

        # ---- numpy path ----
        pool.compute_attention(0, q)          # warm
        t0 = time.time()
        for _ in range(REPS):
            pool.compute_attention(0, q)
        t_np = (time.time() - t0) * 1000.0 / REPS

        # ---- C path ----
        nbt = len(pool.block_table)
        for i in range(nbt):
            bt[i] = pool.block_table[i]
        args = (out.ctypes.data, q.ctypes.data,
                pool.k_pool.ctypes.data, pool.v_pool.ctypes.data, bt.ctypes.data,
                N_LAYERS, N_KV, N_Q, BLOCK, HEAD_DIM, 0, T, SCALE)
        fn(*args)                              # warm
        t0 = time.time()
        for _ in range(REPS):
            fn(*args)
        t_c = (time.time() - t0) * 1000.0 / REPS

        # ---- agreement check ----
        ref = pool.compute_attention(0, q)
        fn(*args)
        max_abs = float(np.max(np.abs(out.astype(np.float64) -
                                      ref.astype(np.float64))))

        saved = t_np - t_c
        print("%6d %12.3f %12.3f %8.2fx %9.3f  (max_abs=%.2e)" %
              (T, t_np, t_c, (t_np / t_c) if t_c > 0 else 0.0, saved, max_abs))
        rows.append((T, t_np, t_c))

    print("")
    print("=== extrapolated per-token effect on the 24-layer model ===")
    print("(the model has 6 full-attention layers, so multiply by 6)")
    print("%6s %14s %14s %14s" % ("T", "numpy total", "C total", "saved"))
    for T, t_np, t_c in rows:
        print("%6d %11.1f ms %11.1f ms %11.1f ms" %
              (T, t_np * 6, t_c * 6, (t_np - t_c) * 6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
