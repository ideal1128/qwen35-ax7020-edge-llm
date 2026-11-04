#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""attn_post_verify.py -- bit-exactness harness for the C full-attention glue.

Run ON THE BOARD, from /mnt/qwen:

    python3 -u attn_post_verify.py

What it checks
--------------
The full-attention callback of qwen_real_engine.py used to do its glue in
numpy (rmsnorm_heads x2, apply_rope x2, the q/gate slice, pool.append_kv and
the sigmoid gate).  That glue now lives in libqwen_attn.so as
attn_post_f32().  This harness feeds random inputs to both implementations and
requires *bit-identical* float32 output, not a tolerance: a 1-ulp drift in the
rmsnorm scale is enough to flip a greedy argmax and change the whole answer.

  stage 1  attn_rmsnorm_heads_f32  vs  qwen_real_engine.rmsnorm_heads()
  stage 2  attn_rope_f32           vs  qwen_real_engine.apply_rope()
  stage 3  attn_gate_f32           vs  a * (1/(1+exp(-gate)))
  full     attn_post_f32           vs  numpy glue + the SHIPPED C attention
                                       kernel full_attn_f32()

The last comparison is the one that matters: the shipped kernel is the
accepted baseline, so the new one-call path must reproduce "numpy glue +
full_attn_f32" exactly.  For information it also prints how far the C
attention kernel itself sits from the pure-numpy pool.compute_attention()
(that ~6e-08 was already accepted before this change and is not touched here).

numpy 1.8.2 sums mean(x*x) sequentially while numpy >= 1.9 sums pairwise, so
every mode of attn_set_sum_mode() is tried and the matching one is reported.
The engine calibrates this itself at start-up; this script exists to prove
that calibration has a solution on this board.

Python 3.4 / numpy 1.8.2 compatible (no f-strings, no pathlib).
"""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/qwen")
try:
    os.chdir("/mnt/qwen")
except OSError:
    pass

import qwen_real_engine as E     # rmsnorm_heads / apply_rope / precompute_rope_freqs

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "libqwen_attn.so")

NH = 8           # q heads
NKV = 2          # kv heads
HD = 256         # head_dim
BS = 16          # block_size
NL = 6           # full-attention layers in the KV pool
EPS = 1e-6
SCALE = 0.0625   # 1/sqrt(256)

FAILS = []


def note(ok, msg):
    print(("  [%s] %s" % ("PASS" if ok else "FAIL", msg)))
    if not ok:
        FAILS.append(msg)


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.int32)


def compare(name, got, ref):
    """Bit-level comparison; the acceptance criterion is max_abs == 0."""
    got = np.ascontiguousarray(got, dtype=np.float32).reshape(-1)
    ref = np.ascontiguousarray(ref, dtype=np.float32).reshape(-1)
    if got.shape != ref.shape:
        note(False, "%s: shape %r != %r" % (name, got.shape, ref.shape))
        return False
    diff = np.abs(got - ref)
    max_abs = float(diff.max()) if diff.size else 0.0
    nbit = int(np.sum(bits(got) != bits(ref)))
    ok = (max_abs == 0.0)
    note(ok, "%-28s max_abs=%.3e  bitdiff=%d/%d"
         % (name, max_abs, nbit, got.size))
    return ok


def load_lib():
    lib = ctypes.CDLL(LIB_PATH)

    f = lib.attn_post_f32
    f.restype = ctypes.c_int
    f.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
    ]

    f = lib.attn_rmsnorm_heads_f32
    f.restype = None
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                  ctypes.c_float]

    f = lib.attn_rope_f32
    f.restype = None
    f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                  ctypes.c_void_p, ctypes.c_void_p]

    f = lib.attn_gate_f32
    f.restype = None
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_int, ctypes.c_int]

    f = lib.attn_set_sum_mode
    f.restype = None
    f.argtypes = [ctypes.c_int]

    f = lib.attn_get_sum_mode
    f.restype = ctypes.c_int
    f.argtypes = []

    f = lib.full_attn_f32
    f.restype = None
    f.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
    ]
    return lib


# ---------------------------------------------------------------------------
# stage 1: rmsnorm_heads
# ---------------------------------------------------------------------------
def test_rmsnorm(lib):
    print("\n=== stage 1: attn_rmsnorm_heads_f32 vs numpy rmsnorm_heads ===")
    rs = np.random.RandomState(11)
    fn = lib.attn_rmsnorm_heads_f32
    setmode = lib.attn_set_sum_mode

    # first: does numpy 1.8.2's x**2 differ from x*x at all?
    pow_vs_mul = 0
    for _ in range(200):
        x = (rs.randn(8, HD) * (10.0 ** rs.randint(-3, 3))).astype(np.float32)
        if not np.array_equal(np.mean(x ** 2, axis=-1), np.mean(x * x, axis=-1)):
            pow_vs_mul += 1
    print("  numpy mean(x**2) != mean(x*x) on %d/200 random batches" % pow_vs_mul)

    worst = {}
    for mode in (0, 1, 2, 3):
        setmode(mode)
        worst[mode] = 0.0
        nbit = 0
        for trial in range(16):
            q_raw = (rs.randn(NH, 2, HD) * (10.0 ** rs.randint(-3, 3))).astype(np.float32)
            w = (rs.randn(HD) * 0.1).astype(np.float32)
            out = np.empty(NH * HD, dtype=np.float32)
            fn(out.ctypes.data, q_raw.ctypes.data, NH, HD, 2 * HD,
               w.ctypes.data, ctypes.c_float(EPS))
            ref = E.rmsnorm_heads(q_raw[:, 0, :], w).reshape(-1)
            d = float(np.abs(out - ref).max())
            if d > worst[mode]:
                worst[mode] = d
            nbit += int(np.sum(bits(out) != bits(ref)))
        print("  sum_mode=%d  max_abs=%.3e  bitdiff=%d  %s"
              % (mode, worst[mode], nbit, "EXACT" if worst[mode] == 0.0 else "differs"))

    exact = [m for m in (0, 1, 2, 3) if worst[m] == 0.0]
    note(bool(exact), "at least one summation mode is bit-exact: %r" % (exact,))
    if exact:
        # check the k-head shape too (stride = head_dim, 2 heads)
        setmode(exact[0])
        for trial in range(8):
            k_raw = (rs.randn(NKV, HD) * 4.0).astype(np.float32)
            w = (rs.randn(HD) * 0.1).astype(np.float32)
            out = np.empty(NKV * HD, dtype=np.float32)
            fn(out.ctypes.data, k_raw.ctypes.data, NKV, HD, HD,
               w.ctypes.data, ctypes.c_float(EPS))
            ref = E.rmsnorm_heads(k_raw, w).reshape(-1)
            if not compare("k-heads trial %d" % trial, out, ref):
                break
    return exact


# ---------------------------------------------------------------------------
# stage 2: apply_rope
# ---------------------------------------------------------------------------
def test_rope(lib, cos_tab, sin_tab):
    print("\n=== stage 2: attn_rope_f32 vs numpy apply_rope ===")
    rs = np.random.RandomState(22)
    fn = lib.attn_rope_f32
    ok = True
    for trial in range(16):
        x = (rs.randn(NH, HD) * 2.0).astype(np.float32)
        pos = int(rs.randint(0, 2048))
        ref = E.apply_rope(x.copy(), pos, cos_tab, sin_tab)
        buf = np.array(x, dtype=np.float32, copy=True)
        fn(buf.ctypes.data, NH, HD,
           cos_tab.ctypes.data + pos * 32 * 4,
           sin_tab.ctypes.data + pos * 32 * 4)
        ok = compare("rope pos=%4d" % pos, buf, ref) and ok
        if not ok:
            break
    # 2 kv heads as well
    if ok:
        x = (rs.randn(NKV, HD) * 2.0).astype(np.float32)
        pos = 33
        ref = E.apply_rope(x.copy(), pos, cos_tab, sin_tab)
        buf = np.array(x, dtype=np.float32, copy=True)
        fn(buf.ctypes.data, NKV, HD,
           cos_tab.ctypes.data + pos * 32 * 4,
           sin_tab.ctypes.data + pos * 32 * 4)
        ok = compare("rope kv-heads pos=33", buf, ref)
    return ok


# ---------------------------------------------------------------------------
# stage 3: sigmoid gate
# ---------------------------------------------------------------------------
def test_gate(lib):
    print("\n=== stage 3: attn_gate_f32 vs numpy a * sigmoid(gate) ===")
    rs = np.random.RandomState(33)
    fn = lib.attn_gate_f32
    ok = True
    for trial, scale in enumerate((0.1, 1.0, 4.0, 12.0)):
        a = (rs.randn(NH, HD) * 2.0).astype(np.float32)
        q_raw = (rs.randn(NH, 2, HD) * scale).astype(np.float32)
        gate = q_raw[:, 1, :]
        ref = (a * (1.0 / (1.0 + np.exp(-gate)))).reshape(-1)
        out = np.empty(NH * HD, dtype=np.float32)
        fn(out.ctypes.data, a.ctypes.data, q_raw.ctypes.data, NH, HD)
        ok = compare("gate scale=%.1f" % scale, out, ref) and ok
    return ok


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------
def numpy_attention(k_pool, v_pool, block_table, layer, q, total_tokens):
    """Pure-numpy equivalent of PagedAttentionPool.compute_attention()."""
    if len(block_table) == 1:
        k_all = k_pool[block_table[0], layer, :, :total_tokens, :]
        v_all = v_pool[block_table[0], layer, :, :total_tokens, :]
    else:
        k_all = np.concatenate([k_pool[b, layer] for b in block_table],
                               axis=1)[:, :total_tokens, :]
        v_all = np.concatenate([v_pool[b, layer] for b in block_table],
                               axis=1)[:, :total_tokens, :]
    k_exp = np.repeat(k_all, 4, axis=0)
    v_exp = np.repeat(v_all, 4, axis=0)
    logits = np.einsum('hd,htd->ht', q, k_exp) * 0.0625
    max_l = np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(logits - max_l)
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    return np.einsum('ht,htd->hd', probs, v_exp).reshape(-1)


def test_full(lib, cos_tab, sin_tab, mode):
    print("\n=== full pipeline: attn_post_f32 vs numpy glue + shipped kernel ===")
    rs = np.random.RandomState(44)
    setmode = lib.attn_set_sum_mode
    setmode(mode)

    # (num_blocks, total_tokens, block_table, layer)
    cases = [
        (1, 1, [0], 0),
        (1, 16, [0], 3),
        (2, 17, [0, 1], 5),
        (3, 33, [2, 0, 1], 1),
        (3, 40, [1, 2, 0], 4),
        (4, 48, [3, 1, 0, 2], 0),   # over-long table: bt[-1] != bt[pos/BS]
        (4, 49, [3, 1, 0, 2], 2),   # first token of a fresh block
        (4, 64, [2, 3, 1, 0], 5),   # last token of a full block
    ]
    ok = True
    for n_blocks, total_tokens, bt, layer in cases:
        pos = total_tokens - 1
        bt32 = np.array(bt, dtype=np.int32)
        pool_shape = (n_blocks, NL, NKV, BS, HD)
        k_pool = (rs.randn(*pool_shape) * 0.5).astype(np.float32)
        v_pool = (rs.randn(*pool_shape) * 0.5).astype(np.float32)
        q_raw = (rs.randn(NH, 2, HD)).astype(np.float32)
        k_raw = (rs.randn(NKV, HD)).astype(np.float32)
        v_raw = (rs.randn(NKV, HD)).astype(np.float32)
        qw = (rs.randn(HD) * 0.1).astype(np.float32)
        kw = (rs.randn(HD) * 0.1).astype(np.float32)

        # --- reference: numpy glue, then the SHIPPED C attention kernel ---
        # The token being generated is logical position pos, so its physical
        # block is block_table[pos / block_size] -- NOT block_table[-1], which
        # only coincides with it because the engine keeps the table exactly
        # sized.  One case below has a deliberately over-long table so this
        # indexing is actually exercised.
        k_ref = np.array(k_pool, dtype=np.float32, copy=True)
        v_ref = np.array(v_pool, dtype=np.float32, copy=True)
        q_rot = E.apply_rope(E.rmsnorm_heads(q_raw[:, 0, :], qw), pos,
                             cos_tab, sin_tab)
        k_rot = E.apply_rope(E.rmsnorm_heads(k_raw, kw), pos,
                             cos_tab, sin_tab)
        blk = bt[pos // BS]
        off = pos % BS
        k_ref[blk, layer, :, off, :] = k_rot
        v_ref[blk, layer, :, off, :] = v_raw
        a_ref = np.empty(NH * HD, dtype=np.float32)
        lib.full_attn_f32(a_ref.ctypes.data, q_rot.ctypes.data,
                          k_ref.ctypes.data, v_ref.ctypes.data, bt32.ctypes.data,
                          NL, NKV, NH, BS, HD, layer, total_tokens,
                          ctypes.c_float(SCALE))
        gate = q_raw[:, 1, :].reshape(-1)
        gated_ref = a_ref * (1.0 / (1.0 + np.exp(-gate)))

        # --- the new one-call C path ---
        k_new = np.array(k_pool, dtype=np.float32, copy=True)
        v_new = np.array(v_pool, dtype=np.float32, copy=True)
        gated_new = np.empty(NH * HD, dtype=np.float32)
        rc = lib.attn_post_f32(
            gated_new.ctypes.data, q_raw.ctypes.data, k_raw.ctypes.data,
            v_raw.ctypes.data, qw.ctypes.data, kw.ctypes.data,
            cos_tab.ctypes.data + pos * 32 * 4,
            sin_tab.ctypes.data + pos * 32 * 4,
            ctypes.c_float(EPS), NH, NKV, HD,
            k_new.ctypes.data, v_new.ctypes.data, bt32.ctypes.data,
            NL, BS, layer, total_tokens, ctypes.c_float(SCALE))
        tag = "T=%2d blocks=%d layer=%d" % (total_tokens, n_blocks, layer)
        if rc != 0:
            note(False, "%s: attn_post_f32 returned %d" % (tag, rc))
            ok = False
            continue
        ok = compare(tag + " gated", gated_new, gated_ref) and ok
        ok = compare(tag + " k_pool", k_new, k_ref) and ok
        ok = compare(tag + " v_pool", v_new, v_ref) and ok

        # informational: the shipped kernel vs pure numpy attention
        a_np = numpy_attention(k_ref, v_ref, bt, layer, q_rot, total_tokens)
        print("      (info) shipped kernel vs numpy attention: max_abs=%.3e"
              % (float(np.abs(a_ref - a_np).max()),))

    # argument validation must reject impossible shapes instead of crashing
    print("\n  negative tests (must return non-zero, not crash):")
    q_raw = np.zeros((NH, 2, HD), dtype=np.float32)
    bad = np.zeros((NH, HD), dtype=np.float32)
    k_pool = np.zeros((1, NL, NKV, BS, HD), dtype=np.float32)
    v_pool = np.zeros((1, NL, NKV, BS, HD), dtype=np.float32)
    bt32 = np.zeros(1, dtype=np.int32)
    g = np.empty(NH * HD, dtype=np.float32)
    rc = lib.attn_post_f32(g.ctypes.data, q_raw.ctypes.data, bad.ctypes.data,
                           bad.ctypes.data, bad.ctypes.data, bad.ctypes.data,
                           bad.ctypes.data, bad.ctypes.data, ctypes.c_float(EPS),
                           3, 2, HD, k_pool.ctypes.data, v_pool.ctypes.data,
                           bt32.ctypes.data, NL, BS, 0, 1, ctypes.c_float(SCALE))
    note(rc != 0, "nh=3 nkv=2 rejected (rc=%d)" % rc)
    rc = lib.attn_post_f32(g.ctypes.data, q_raw.ctypes.data, bad.ctypes.data,
                           bad.ctypes.data, bad.ctypes.data, bad.ctypes.data,
                           bad.ctypes.data, bad.ctypes.data, ctypes.c_float(EPS),
                           NH, NKV, 512, k_pool.ctypes.data, v_pool.ctypes.data,
                           bt32.ctypes.data, NL, BS, 0, 1, ctypes.c_float(SCALE))
    note(rc != 0, "head_dim=512 rejected (rc=%d)" % rc)
    return ok


def main():
    print("=" * 72)
    print("  attn_post bit-exactness harness (numpy %s)" % (np.__version__,))
    print("=" * 72)
    if not os.path.exists(LIB_PATH):
        print("FATAL: %s not found" % (LIB_PATH,))
        return 2
    lib = load_lib()
    cos_tab, sin_tab = E.precompute_rope_freqs(rotary_dim=64, max_seq=2048)

    exact_modes = test_rmsnorm(lib)
    test_rope(lib, cos_tab, sin_tab)
    test_gate(lib)

    if exact_modes:
        test_full(lib, cos_tab, sin_tab, exact_modes[0])
    else:
        print("\n!! no bit-exact rmsnorm summation mode -- full test skipped")
        FAILS.append("no bit-exact rmsnorm summation mode")

    lib.attn_set_sum_mode(0)
    print("\n" + "=" * 72)
    if FAILS:
        print("  VERDICT: FAIL (%d problem(s))" % (len(FAILS),))
        for m in FAILS:
            print("    - %s" % (m,))
        print("=" * 72)
        return 1
    print("  VERDICT: PASS -- C glue is bit-identical to numpy")
    print("  engine will calibrate sum_mode=%d at start-up" % (exact_modes[0],))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
