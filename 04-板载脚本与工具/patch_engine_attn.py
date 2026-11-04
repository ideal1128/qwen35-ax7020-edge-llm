#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_engine_attn.py -- route the full-attention layers through libqwen_attn.so

Run ON THE BOARD:

    cd /mnt/qwen
    python3 patch_engine_attn.py            # apply
    python3 patch_engine_attn.py --check    # dry run, report what would change
    python3 patch_engine_attn.py --revert   # restore the .pre_attn backup

What it changes in qwen_real_engine.py:

  1. Inserts a ctypes binding for libqwen_attn.so just before
     `class PagedAttentionPool`.
  2. Replaces the single line
         a_out = pool.compute_attention(pool_idx, q_rot)
     with
         a_out = _attn_dispatch(pool, pool_idx, q_rot)
     which uses the C kernel when the library is loaded and silently falls
     back to the numpy path otherwise.

Everything else is untouched.  A byte-for-byte backup is written next to the
original as qwen_real_engine.py.pre_attn.

ASCII only, Python 3.4 compatible (no f-strings).
"""
import os
import shutil
import sys

TARGET = "/mnt/qwen/qwen_real_engine.py"

MARKER_BEGIN = "# ==== BEGIN qwen_attn binding (patch_engine_attn.py) ===="
MARKER_END = "# ==== END qwen_attn binding ===="

OLD_CALL = "a_out = pool.compute_attention(pool_idx, q_rot)"
NEW_CALL = "a_out = _attn_dispatch(pool, pool_idx, q_rot)"

BINDING = '''# ==== BEGIN qwen_attn binding (patch_engine_attn.py) ====
# Full attention in C+NEON.  The numpy path below builds a [8,T,256] copy of
# K and V with np.repeat on every layer of every token; at T=544 that is a
# 4.5 MB allocation per layer and it is what makes long conversations slow
# (817 ms/tok at ctx 20 -> 1214 ms/tok at ctx 544).  The C kernel indexes the
# GQA 4:1 mapping instead of materialising it, and reads K/V once per layer.
_ATTN_LIB = None
_ATTN_FN = None
_ATTN_OUT = None
_ATTN_BT = None


def _attn_load():
    """Load libqwen_attn.so once.  Returns True if the C path is usable."""
    global _ATTN_LIB, _ATTN_FN, _ATTN_OUT, _ATTN_BT
    if _ATTN_FN is not None:
        return True
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "libqwen_attn.so")
    if not os.path.exists(path):
        return False
    try:
        import ctypes
        lib = ctypes.CDLL(path)
        fn = lib.full_attn_f32
        fn.restype = None
        fn.argtypes = [
            ctypes.c_void_p,   # out
            ctypes.c_void_p,   # q
            ctypes.c_void_p,   # k_pool
            ctypes.c_void_p,   # v_pool
            ctypes.c_void_p,   # block_table (int32)
            ctypes.c_int,      # n_layers
            ctypes.c_int,      # n_kv_heads
            ctypes.c_int,      # n_q_heads
            ctypes.c_int,      # block_size
            ctypes.c_int,      # head_dim
            ctypes.c_int,      # layer
            ctypes.c_int,      # total_tokens
            ctypes.c_float,    # scale
        ]
        import numpy as _np
        _ATTN_OUT = _np.zeros(8 * 256, dtype=_np.float32)
        _ATTN_BT = _np.zeros(128, dtype=_np.int32)
        _ATTN_LIB = lib
        _ATTN_FN = fn
        return True
    except Exception as exc:
        sys.stderr.write("[attn] libqwen_attn.so load failed: %s\\n" % (exc,))
        return False


def _attn_dispatch(pool, layer_idx, q):
    """C+NEON full attention, falling back to the numpy pool on any problem."""
    if not _attn_load():
        return pool.compute_attention(layer_idx, q)
    try:
        import ctypes
        import numpy as _np
        total_tokens = pool.seq_len + 1
        bt = pool.block_table
        n_bt = len(bt)
        if n_bt > _ATTN_BT.shape[0] or total_tokens > 2048:
            return pool.compute_attention(layer_idx, q)
        for i in range(n_bt):
            _ATTN_BT[i] = bt[i]
        qc = _np.ascontiguousarray(q, dtype=_np.float32)
        _ATTN_FN(
            _ATTN_OUT.ctypes.data,
            qc.ctypes.data,
            pool.k_pool.ctypes.data,
            pool.v_pool.ctypes.data,
            _ATTN_BT.ctypes.data,
            ctypes.c_int(pool.num_layers),
            ctypes.c_int(pool.num_kv_heads),
            ctypes.c_int(8),
            ctypes.c_int(pool.block_size),
            ctypes.c_int(pool.head_dim),
            ctypes.c_int(layer_idx),
            ctypes.c_int(total_tokens),
            ctypes.c_float(0.0625),
        )
        return _ATTN_OUT
    except Exception as exc:
        sys.stderr.write("[attn] C path failed, using numpy: %s\\n" % (exc,))
        return pool.compute_attention(layer_idx, q)
# ==== END qwen_attn binding ===='''


def read_text(path):
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8")


def write_text(path, text):
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))


def main():
    global TARGET
    args = sys.argv[1:]
    check = "--check" in args
    revert = "--revert" in args
    if "--target" in args:
        i = args.index("--target")
        if i + 1 < len(args):
            TARGET = args[i + 1]
    BACKUP = TARGET + ".pre_attn"

    if not os.path.exists(TARGET):
        print("ERROR: %s not found" % TARGET)
        return 1

    if revert:
        if not os.path.exists(BACKUP):
            print("ERROR: no backup at %s" % BACKUP)
            return 1
        shutil.copyfile(BACKUP, TARGET)
        print("reverted %s from %s" % (TARGET, BACKUP))
        return 0

    src = read_text(TARGET)

    if MARKER_BEGIN in src:
        print("already patched (marker present), nothing to do")
        return 0

    n_sites = src.count(OLD_CALL)
    if n_sites == 0:
        print("ERROR: could not find the call site:")
        print("       %s" % OLD_CALL)
        print("The engine file may be a different version.  Aborting.")
        return 1

    # There are two call sites in the shipped engine:
    #   _on_full_attn_cb  (the C-callback path actually used by forward_token_c)
    #   _layer_fwd        (the pure-Python fallback path)
    # Both must go through the kernel or the two paths would disagree.
    if n_sites > 4:
        print("ERROR: call site appears %d times, refusing to patch blindly"
              % n_sites)
        return 1

    if src.count("class PagedAttentionPool") != 1:
        print("ERROR: PagedAttentionPool anchor not found exactly once")
        return 1

    # 1. insert the binding before the pool class
    src = src.replace("class PagedAttentionPool",
                      BINDING + "\n\n\nclass PagedAttentionPool", 1)
    # 2. swap every call site
    src = src.replace(OLD_CALL, NEW_CALL)

    if check:
        print("[check] would patch %s" % TARGET)
        print("[check] binding inserted before class PagedAttentionPool")
        print("[check] replacing %d call site(s):" % n_sites)
        print("[check]   %s" % OLD_CALL)
        print("[check]   -> %s" % NEW_CALL)
        return 0

    if not os.path.exists(BACKUP):
        shutil.copyfile(TARGET, BACKUP)
        print("backup written: %s" % BACKUP)

    write_text(TARGET, src)
    print("patched %s" % TARGET)
    print("  binding inserted, call site swapped")
    print("  revert with: python3 patch_engine_attn.py --revert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
