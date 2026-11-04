#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# prof_attn_post.py -- per-step wall-clock breakdown of the PATCHED
#                      QwenRealEngine._on_full_attn_cb
#
# Same idea as prof_attn.py, but it mirrors the new callback, i.e. the whole
# numpy glue (rmsnorm_heads + rope + append_kv + attention + sigmoid gate) is
# one attn_post_f32() call.  Comparing the two tables shows exactly what the
# C version bought us:
#
#   before: 03_slice_qgate 06_rmsnorm_heads 07_rope 08_append_kv
#           09_attn_kernel 10_sigmoid_gate   (six numpy steps, 17.8 ms/token)
#   after : 06_attn_post_C                 (one C call)
#
# Run:  python3 -u prof_attn_post.py --cps 20320,22909 --greedy --max 8
# ---------------------------------------------------------------------------
import sys
import time
import numpy as np
import qwen_real_engine as E

T = {}
CALLS = [0]


def _add(k, t):
    T[k] = T.get(k, 0.0) + t


def timed_cb(self, l, h_ptr, ctx):
    CALLS[0] += 1
    p = self._plan[l]
    b = self._b
    normed_ptr = b["normed"][1]

    t0 = time.time()
    self._k_rmsnorm(normed_ptr, h_ptr, p['in_ln'], 1024, 1e-6)
    _add('01_rmsnorm_in', time.time() - t0)

    pool = self.paged_attn
    if not pool.block_table:
        pool.allocate_slot()
    pos = pool.seq_len
    pool_idx = p['pool_idx']

    mp, ep, M, Kd, yqg, yqgp = p['q']
    t0 = time.time()
    self._k_gemv(yqgp, normed_ptr, mp, ep, M, Kd)
    _add('02_gemv_q', time.time() - t0)

    mp, ep, M, Kd, yk, ykp = p['k']
    t0 = time.time()
    self._k_gemv(ykp, normed_ptr, mp, ep, M, Kd)
    _add('04_gemv_k', time.time() - t0)

    mp, ep, M, Kd, yv, yvp = p['v']
    t0 = time.time()
    self._k_gemv(yvp, normed_ptr, mp, ep, M, Kd)
    _add('05_gemv_v', time.time() - t0)

    t0 = time.time()
    gated_ptr = self._attn_post(pool, pool_idx, pos, p, yqgp, ykp, yvp)
    _add('06_attn_post_C', time.time() - t0)

    mp, ep, M, Kd, yo, yop = p['o']
    t0 = time.time()
    self._k_gemv(yop, gated_ptr, mp, ep, M, Kd)
    _add('11_gemv_o', time.time() - t0)

    t0 = time.time()
    np.add(self._h_arr, yo, out=self._h_arr)
    _add('12_residual', time.time() - t0)


E.QwenRealEngine._on_full_attn_cb = timed_cb


def dump():
    tot = sum(T.values())
    if tot <= 0.0:
        return
    print("")
    print("=== _on_full_attn_cb breakdown (C attn_post path) ===")
    print("callback calls: %d" % CALLS[0])
    print("%-20s %10s %8s %10s" % ("step", "total ms", "share", "ms/call"))
    for k in sorted(T, key=lambda x: -T[x]):
        print("%-20s %10.1f %7.1f%% %10.3f"
              % (k, T[k] * 1000.0, 100.0 * T[k] / tot, T[k] * 1000.0 / CALLS[0]))
    print("%-20s %10.1f %7.1f%% %10.3f"
          % ("TOTAL", tot * 1000.0, 100.0, tot * 1000.0 / CALLS[0]))


sys.argv = ['chat_v3.py'] + sys.argv[1:]
try:
    exec(compile(open('/mnt/qwen/chat_v3.py').read(), 'chat_v3.py', 'exec'))
finally:
    dump()
