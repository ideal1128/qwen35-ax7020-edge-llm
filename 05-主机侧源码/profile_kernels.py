#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""profile_kernels.py - per-kernel time accounting for one token (Python 3.4 safe)."""
import os
import sys
import time
import collections

sys.path.insert(0, "/mnt/qwen")
os.chdir("/mnt/qwen")

from qwen_real_engine import QwenRealEngine

_ENGINE = os.environ.get("ENGINE", "")
if _ENGINE:
    QwenRealEngine = __import__(_ENGINE).QwenRealEngine

KERNELS = ["bfp8_gemv", "gdn_recurrent_step", "rmsnorm", "swiglu", "add_residual",
           "causal_conv1d_update_silu", "qk_l2norm", "rmsnorm_gated", "lm_head_argmax"]

total = collections.OrderedDict()
calls = collections.OrderedDict()


def wrap(eng):
    for name in KERNELS:
        if not hasattr(eng.kernels, name):
            continue
        f = getattr(eng.kernels, name)
        total[name] = 0.0
        calls[name] = 0

        def make(n, fn):
            def g(*a):
                t = time.time()
                r = fn(*a)
                total[n] += time.time() - t
                calls[n] += 1
                return r
            return g
        setattr(eng.kernels, name, make(name, f))


def main():
    eng = QwenRealEngine()
    wrap(eng)

    tok = 9419
    eng.forward_token(tok)          # warm

    for d in total:
        total[d] = 0.0
        calls[d] = 0

    t0 = time.time()
    tok = eng.forward_token(tok)
    wall = time.time() - t0

    print("[PROF] wall = {:.1f} ms".format(wall * 1000.0))
    acc = 0.0
    for name in total:
        if calls[name]:
            print("[PROF] {:28s} {:9.1f} ms  {:5d} calls  {:7.2f} ms/call".format(
                name, total[name] * 1000.0, calls[name],
                total[name] * 1000.0 / calls[name]))
            acc += total[name]
    print("[PROF] kernels total = {:.1f} ms, python/other = {:.1f} ms".format(
        acc * 1000.0, (wall - acc) * 1000.0))


if __name__ == "__main__":
    main()
