#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# prof_attn.py -- per-step wall-clock breakdown of
#                  QwenRealEngine._on_full_attn_cb
#
# The C profiler attributes ~67 ms/token to "py_full_attn", but that slot only
# wraps the Python callback.  At a 20-token context the actual attention kernel
# is ~0.4 ms x 6 layers = 2.4 ms and the four gemvs are ~16.6 ms, so roughly
# 48 ms is unaccounted for.  This script replaces the callback with an
# instrumented copy and reports where that time goes.
#
# Run:  python3 -u prof_attn.py --cps 20320,22909 --greedy --max 8
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

    t0 = time.time()
    q_and_gate = yqg.reshape(8, 2, 256)
    q_raw = q_and_gate[:, 0, :]
    gate = q_and_gate[:, 1, :].reshape(2048)
    _add('03_slice_qgate', time.time() - t0)

    mp, ep, M, Kd, yk, ykp = p['k']
    t0 = time.time()
    self._k_gemv(ykp, normed_ptr, mp, ep, M, Kd)
    _add('04_gemv_k', time.time() - t0)
    k_raw = yk.reshape(2, 256)

    mp, ep, M, Kd, yv, yvp = p['v']
    t0 = time.time()
    self._k_gemv(yvp, normed_ptr, mp, ep, M, Kd)
    _add('05_gemv_v', time.time() - t0)
    v_raw = yv.reshape(2, 256)

    t0 = time.time()
    q_normed = E.rmsnorm_heads(q_raw, p['q_norm'])
    k_normed = E.rmsnorm_heads(k_raw, p['k_norm'])
    _add('06_rmsnorm_heads', time.time() - t0)

    t0 = time.time()
    q_rot = E.apply_rope(q_normed, pos, self.cos_table, self.sin_table)
    k_rot = E.apply_rope(k_normed, pos, self.cos_table, self.sin_table)
    _add('07_rope', time.time() - t0)

    t0 = time.time()
    pool.append_kv(pool_idx, k_rot, v_raw)
    _add('08_append_kv', time.time() - t0)

    t0 = time.time()
    a_out = E._attn_dispatch(pool, pool_idx, q_rot)
    _add('09_attn_kernel', time.time() - t0)

    t0 = time.time()
    gated_attn = a_out * (1.0 / (1.0 + np.exp(-gate)))
    _add('10_sigmoid_gate', time.time() - t0)

    mp, ep, M, Kd, yo, yop = p['o']
    t0 = time.time()
    self._k_gemv(yop, gated_attn.ctypes.data, mp, ep, M, Kd)
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
    print("=== _on_full_attn_cb breakdown ===")
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
