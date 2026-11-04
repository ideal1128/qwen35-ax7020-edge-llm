#!/usr/bin/env python3
# gdn_rec_verify.py -- bit-exact A/B test for gdn_recurrent_step.
#
#   python3 gdn_rec_verify.py [old_so] [new_so] [layout]
#
# layout = memory layout the NEW kernel uses for the state:
#   't'  transposed   state[h*D*D + j*D + i]
#   'b'  4x4 blocked  state[h*D*D + (jg*D + 4*ig)*4 + 4*l + t] = S[4*ig+t][4*jg+l]
#   'r'  row-major (identical to the old kernel; not used any more)
# The old kernel (backup .so) always keeps the state row-major:
#   S = state[h*D*D + i*D + j]
#
# The test feeds identical q/k/v/alpha/beta to both kernels over several
# recurrent steps, starting once from an all-zero state and once from a random
# non-zero state, and requires
#   * attn_out  bit-identical (memcmp)
#   * state     bit-identical after mapping the new layout back (memcmp)
# It also micro-benchmarks both kernels (18 calls == one token).
#
# Pure ASCII, python 3.4 compatible (no f-strings).

import os
import sys
import time
import ctypes
import numpy as np

H = 16
D = 128
OLD_SO = sys.argv[1] if len(sys.argv) > 1 else '/mnt/qwen/libqwen_kernels.so.bak_task2'
NEW_SO = sys.argv[2] if len(sys.argv) > 2 else '/mnt/qwen/libqwen_kernels.so'
LAYOUT = sys.argv[3] if len(sys.argv) > 3 else 'b'
NG = D // 4


def load(path):
    lib = ctypes.CDLL(os.path.abspath(path))
    fn = lib.gdn_recurrent_step
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int]
    return lib, fn


def call(fn, out, state, q, k, v, alpha, beta):
    fn(out.ctypes.data, state.ctypes.data,
       q.ctypes.data, k.ctypes.data, v.ctypes.data,
       alpha.ctypes.data, beta.ctypes.data, H, D)


def bits(a):
    return np.ascontiguousarray(a).view(np.uint32).ravel()


def diff_report(name, a, b):
    aa = bits(a)
    bb = bits(b)
    idx = np.nonzero(aa != bb)[0]
    if idx.size == 0:
        return True, ''
    i = int(idx[0])
    return False, ('%s: %d/%d elements differ, first at %d: '
                   'old=0x%08x new=0x%08x' % (name, idx.size, aa.size, i,
                                              int(aa[i]), int(bb[i])))


def to_new(S):
    """logical S[h][i][j] -> memory image the new kernel expects"""
    S = np.ascontiguousarray(S, dtype=np.float32)
    if LAYOUT == 'r':
        return np.array(S, dtype=np.float32, order='C')
    if LAYOUT == 't':
        return np.ascontiguousarray(S.transpose(0, 2, 1))
    x = S.reshape(H, NG, 4, NG, 4)          # h, ig, t, jg, l
    x = x.transpose(0, 3, 1, 4, 2)          # h, jg, ig, l, t
    return np.ascontiguousarray(x)


def from_new(B):
    """memory image of the new kernel -> logical S[h][i][j]"""
    B = np.ascontiguousarray(B, dtype=np.float32)
    if LAYOUT == 'r':
        return np.array(B, dtype=np.float32, order='C')
    if LAYOUT == 't':
        return np.ascontiguousarray(B.transpose(0, 2, 1))
    x = B.reshape(H, NG, NG, 4, 4)          # h, jg, ig, l, t
    x = x.transpose(0, 2, 4, 1, 3)          # h, ig, t, jg, l
    return np.ascontiguousarray(x.reshape(H, D, D))


def main():
    print('old .so: %s' % OLD_SO)
    print('new .so: %s  (layout %s)' % (NEW_SO, LAYOUT))
    old_lib, old_fn = load(OLD_SO)
    new_lib, new_fn = load(NEW_SO)

    # layout round-trip self-check
    rs0 = np.random.RandomState(1234)
    Sx = rs0.randn(H, D, D).astype(np.float32)
    assert np.array_equal(from_new(to_new(Sx)), Sx), 'layout round-trip broken'

    total_fail = 0
    nsteps = 6

    for trial in range(2):
        mode = 'zero' if trial == 0 else 'random'
        rs = np.random.RandomState(20261104 + trial)
        if mode == 'zero':
            S = np.zeros((H, D, D), dtype=np.float32)
        else:
            S = (rs.randn(H, D, D) * 0.5).astype(np.float32)
        S_old = np.array(S, dtype=np.float32, order='C')
        S_new = to_new(S)

        print('')
        print('=== trial %d: initial state = %s ===' % (trial, mode))
        for step in range(nsteps):
            q = rs.randn(H, D).astype(np.float32)
            k = rs.randn(H, D).astype(np.float32)
            v = rs.randn(H, D).astype(np.float32)
            alpha = rs.uniform(0.50, 0.999, H).astype(np.float32)
            beta = rs.uniform(0.00, 1.00, H).astype(np.float32)

            out_old = np.zeros((H, D), dtype=np.float32)
            out_new = np.zeros((H, D), dtype=np.float32)
            call(old_fn, out_old, S_old, q, k, v, alpha, beta)
            call(new_fn, out_new, S_new, q, k, v, alpha, beta)

            ok_out, msg_out = diff_report('out', out_old, out_new)
            ok_st, msg_st = diff_report('state', S_old, from_new(S_new))

            dmax = 0.0
            if not ok_out:
                dmax = float(np.max(np.abs(out_old - out_new)))
            print('  step %d: out %s   state %s%s'
                  % (step, 'BITEXACT' if ok_out else 'MISMATCH',
                     'BITEXACT' if ok_st else 'MISMATCH',
                     '' if ok_out else ('   max|dout|=%.3g' % dmax)))
            if not ok_out:
                print('    ' + msg_out)
            if not ok_st:
                print('    ' + msg_st)
            if not (ok_out and ok_st):
                total_fail += 1
                break

    # ---- micro-benchmark: one call == one GDN layer, 18 layers per token ----
    print('')
    print('=== micro-benchmark (one call = one GDN layer, 18 layers/token) ===')
    rs = np.random.RandomState(7)
    q = rs.randn(H, D).astype(np.float32)
    k = rs.randn(H, D).astype(np.float32)
    v = rs.randn(H, D).astype(np.float32)
    alpha = rs.uniform(0.5, 0.999, H).astype(np.float32)
    beta = rs.uniform(0.0, 1.0, H).astype(np.float32)
    out = np.zeros((H, D), dtype=np.float32)
    n = 200

    for name, fn in (('old', old_fn), ('new', new_fn)):
        st = np.ascontiguousarray(np.random.RandomState(9).randn(H, D, D).astype(np.float32))
        call(fn, out, st, q, k, v, alpha, beta)   # warm up
        t0 = time.time()
        for _ in range(n):
            call(fn, out, st, q, k, v, alpha, beta)
        ms = (time.time() - t0) * 1000.0 / n
        print('  %s: %.3f ms/call  ->  %.2f ms/token (x18)'
              % (name, ms, ms * 18.0))

    print('')
    if total_fail == 0:
        print('RESULT: PASS  (out and state bit-identical, %d steps x 2 trials)'
              % nsteps)
        return 0
    print('RESULT: FAIL  (%d mismatching steps)' % total_fail)
    return 1


if __name__ == '__main__':
    sys.exit(main())
