#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# attn_overhead_bench.py
#
# Two questions left open by prof_attn.py:
#   (A) rmsnorm_heads costs 0.60 ms per call on an 8x256 float32 array.
#       Five numpy ops on 8 KB should be ~25 us.  Which sub-op is slow, and
#       does calling it right after an OpenMP gemv change anything?
#   (B) the o_proj gemv moves 1.06 MB in 2.383 ms (447 MB/s) while k_proj
#       moves 0.53 MB in 0.595 ms (892 MB/s).  Fixed cost per call, or a
#       real bandwidth difference?
#
# ASCII only.  Run: python3 -u attn_overhead_bench.py
# ---------------------------------------------------------------------------
import ctypes
import time
import numpy as np

K = ctypes.CDLL('/mnt/qwen/libqwen_kernels.so')
K.bfp8_gemv.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
K.rmsnorm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_int, ctypes.c_float]


def bench(fn, n, warm=20):
    for _ in range(warm):
        fn()
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) * 1000.0 / n      # ms per call


print("=== (A) tiny numpy ops, exact same shapes as the attn callback ===")

q_raw = np.random.randn(8, 256).astype(np.float32)
k_raw = np.random.randn(2, 256).astype(np.float32)
w = np.random.randn(256).astype(np.float32)
gate = np.random.randn(2048).astype(np.float32)
a_out = np.random.randn(8, 256).astype(np.float32)
cos_t = np.random.randn(2048, 32).astype(np.float32)
sin_t = np.random.randn(2048, 32).astype(np.float32)


def rmsnorm_heads(x, weight, eps=1e-6):
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * (1.0 + weight)


def apply_rope(x, pos, cos_table, sin_table):
    cos = cos_table[pos]
    sin = sin_table[pos]
    x_rot = x[:, :64]
    x_pass = x[:, 64:]
    x1 = x_rot[:, :32]
    x2 = x_rot[:, 32:]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x2 * cos + x1 * sin
    return np.concatenate([rot_x1, rot_x2, x_pass], axis=-1)


gate2 = gate.reshape(8, 256)


def sigmoid_gate(a, g):
    return a * (1.0 / (1.0 + np.exp(-g)))


rows = [
    ("python noop", lambda: None),
    ("np.empty(2048)", lambda: np.empty(2048, dtype=np.float32)),
    ("np.empty(8,256)", lambda: np.empty((8, 256), dtype=np.float32)),
    ("np.add out= 2048", lambda: np.add(gate, gate, out=gate)),
    ("np.exp 2048", lambda: np.exp(gate)),
    ("rmsnorm_heads (8,256)", lambda: rmsnorm_heads(q_raw, w)),
    ("rmsnorm_heads (2,256)", lambda: rmsnorm_heads(k_raw, w)),
    ("apply_rope (8,256)", lambda: apply_rope(q_raw, 100, cos_t, sin_t)),
    ("apply_rope (2,256)", lambda: apply_rope(k_raw, 100, cos_t, sin_t)),
    ("sigmoid_gate (8,256)", lambda: sigmoid_gate(a_out, gate2)),
]
for name, fn in rows:
    print("  %-24s %9.4f ms" % (name, bench(fn, 2000)))

print("")
print("  sub-ops of rmsnorm_heads (8,256):")
sub = [
    ("x**2", lambda: q_raw ** 2),
    ("np.mean(...,-1,keep)", lambda: np.mean(q_raw ** 2, axis=-1, keepdims=True)),
    ("np.sqrt(v+eps)", lambda: np.sqrt(np.mean(q_raw ** 2, axis=-1, keepdims=True) + 1e-6)),
    ("x / sqrt(...)", lambda: q_raw / np.sqrt(
        np.mean(q_raw ** 2, axis=-1, keepdims=True) + 1e-6)),
    ("1.0 + weight", lambda: 1.0 + w),
]
for name, fn in sub:
    print("    %-24s %9.4f ms" % (name, bench(fn, 2000)))

print("")
print("  strided view (the real q_raw is yqg.reshape(8,2,256)[:,0,:]):")
yqg = np.random.randn(8, 2, 256).astype(np.float32)
q_strided = yqg[:, 0, :]
print("    %-24s %9.4f ms" % ("rmsnorm_heads strided",
                              bench(lambda: rmsnorm_heads(q_strided, w), 2000)))
print("    %-24s %9.4f ms" % ("apply_rope strided",
                              bench(lambda: apply_rope(q_strided, 100, cos_t, sin_t), 2000)))


print("")
print("=== (B) bfp8_gemv bandwidth vs M  (K=1024) ===")
print("  %6s %10s %10s %12s %12s" % ("M", "bytes", "ms/call", "MB/s", "MB/s(2nd)"))


def make_gemv(M, Kd):
    n = M * Kd
    # numpy 1.8.2 has no dtype= on randint
    raw = np.random.randint(-127, 128, size=n + n // 64).astype(np.int8)
    y = np.empty(M, dtype=np.float32)
    x = np.random.randn(Kd).astype(np.float32) * 0.05
    mp = raw.ctypes.data
    ep = raw.ctypes.data + n
    xp = x.ctypes.data
    yp = y.ctypes.data
    return (lambda: K.bfp8_gemv(mp, xp, ep, yp, M, Kd)), raw, n * 8.125 / 8.0


for M in (256, 512, 1024, 2048, 4096):
    fn, raw, nbytes = make_gemv(M, 1024)
    ms = bench(fn, 60, warm=10)
    # second-call measurement: subtract the cost of a hot call with no
    # page-fault exposure by timing a long run after many warmups
    ms_hot = bench(fn, 200, warm=50)
    print("  %6d %10d %10.4f %12.1f %12.1f"
          % (M, int(nbytes), ms, nbytes / ms / 1000.0, nbytes / ms_hot / 1000.0))

print("")
print("=== (C) does an OpenMP gemv perturb the following numpy op? ===")
fn1024, raw1024, nb = make_gemv(1024, 1024)
plain = bench(lambda: rmsnorm_heads(q_raw, w), 2000)


def interleaved():
    fn1024()
    return rmsnorm_heads(q_raw, w)


both = bench(interleaved, 500)
print("  rmsnorm alone        %9.4f ms" % plain)
print("  gemv(1024x1024) alone%9.4f ms" % bench(fn1024, 500))
print("  gemv+rmsnorm pair    %9.4f ms  (sum of parts %.4f)"
      % (both, plain + bench(fn1024, 500)))
