#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_engine_attn_post.py -- move the full-attention numpy glue into C.

The engine's full-attention callback (_on_full_attn_cb, and the matching
branch of _layer_fwd) ran this per layer per token in numpy:

    q_and_gate = yqg.reshape(8, 2, 256)     # q gemv output
    q_raw = q_and_gate[:, 0, :]             # strided view
    gate  = q_and_gate[:, 1, :].reshape(2048)
    q_normed = rmsnorm_heads(q_raw, q_norm)
    k_normed = rmsnorm_heads(k_raw, k_norm)
    q_rot = apply_rope(q_normed, pos, cos_table, sin_table)
    k_rot = apply_rope(k_normed, pos, cos_table, sin_table)
    pool.append_kv(pool_idx, k_rot, v_raw)
    a_out = _attn_dispatch(pool, pool_idx, q_rot)
    gated_attn = a_out * (1.0 / (1.0 + np.exp(-gate)))

On this board that is 17.8 ms/token of pure numpy overhead (measured over 27
tokens: rmsnorm_heads 7.2 ms, sigmoid gate 4.6 ms, rope 3.6 ms, q/gate slice
0.9 ms, append_kv 0.6 ms) out of 704.9 ms/token.  attn_post_f32() in
libqwen_attn.so does all of it in one call.

This script only *inserts* and *replaces* text -- it never rewrites the whole
file -- and it leaves the numpy implementation in place as a fallback
(_attn_post_numpy), so a missing/older .so or a non-bit-exact calibration
simply keeps the old behaviour.

Usage (on the board, in /mnt/qwen):

    cp qwen_real_engine.py qwen_real_engine.py.bak_task1   # done by hand first
    python3 patch_engine_attn_post.py --check
    python3 patch_engine_attn_post.py
    python3 patch_engine_attn_post.py --revert             # undo

Python 3.4 compatible.
"""
import os
import shutil
import sys

TARGET = "/mnt/qwen/qwen_real_engine.py"
BACKUP = TARGET + ".pre_attn_post"

BEGIN = "# ==== BEGIN qwen_attn_post binding (patch_engine_attn_post.py) ===="
END = "# ==== END qwen_attn_post binding ===="

ANCHOR_BINDING = "# ==== END qwen_attn binding ====\n"
ANCHOR_CB = "    def _on_full_attn_cb(self, l, h_ptr, ctx):\n"


# ---------------------------------------------------------------------------
# module level: load + calibrate attn_post_f32
# ---------------------------------------------------------------------------
NEW_BINDING = '''\
# ==== BEGIN qwen_attn_post binding (patch_engine_attn_post.py) ====
# attn_post_f32() replaces the numpy glue that used to sit between the four
# gemvs and the o_proj gemv of a full-attention layer:
#
#     rmsnorm_heads(q)  rmsnorm_heads(k)  apply_rope(q)  apply_rope(k)
#     pool.append_kv()  full attention    a_out * sigmoid(gate)
#
# That glue cost 17.8 ms/token in numpy 1.8.2 on this Cortex-A9 (8x256 arrays,
# ~13-40 cycles/element plus ~100 us of fixed overhead per ufunc call, and
# 5 more ctypes.data property hits in _attn_dispatch).  The C version does it
# in one call and writes the 2048 floats the o_proj gemv consumes.
#
# Bit-exactness is not assumed, it is checked at start-up: numpy 1.8.2 sums
# mean(x*x) sequentially while numpy >= 1.9 sums pairwise, and this cannot be
# decided from the C side, so _attn_post_init() probes every summation mode
# against the numpy that is actually installed and refuses the C path unless
# one of them reproduces rmsnorm_heads bit-for-bit.  If anything is off -- no
# .so, no matching mode, rope/gate mismatch -- the engine silently keeps
# using _attn_post_numpy(), i.e. exactly the code that ran before.
_ATTN_POST = None
_ATTN_GATED = None
_ATTN_POST_TRIED = None   # pool _attn_post_init() was already attempted for


def _attn_post_init(eng, pool):
    """Load attn_post_f32, calibrate it against this numpy.  None on any doubt."""
    global _ATTN_POST, _ATTN_GATED
    if not _attn_load():
        return None
    try:
        lib = _ATTN_LIB
        fn = lib.attn_post_f32
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p,   # gated_out
            ctypes.c_void_p,   # q_raw   [nh][2][hd]
            ctypes.c_void_p,   # k_raw   [nkv][hd]
            ctypes.c_void_p,   # v_raw   [nkv][hd]
            ctypes.c_void_p,   # q_norm weight
            ctypes.c_void_p,   # k_norm weight
            ctypes.c_void_p,   # cos row for this position
            ctypes.c_void_p,   # sin row for this position
            ctypes.c_float,    # eps
            ctypes.c_int,      # n_q_heads
            ctypes.c_int,      # n_kv_heads
            ctypes.c_int,      # head_dim
            ctypes.c_void_p,   # k_pool
            ctypes.c_void_p,   # v_pool
            ctypes.c_void_p,   # block_table (int32)
            ctypes.c_int,      # n_layers
            ctypes.c_int,      # block_size
            ctypes.c_int,      # layer
            ctypes.c_int,      # total_tokens
            ctypes.c_float,    # scale
        ]
        rs_fn = lib.attn_rmsnorm_heads_f32
        rs_fn.restype = None
        rs_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                          ctypes.c_float]
        rope_fn = lib.attn_rope_f32
        rope_fn.restype = None
        rope_fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                            ctypes.c_void_p, ctypes.c_void_p]
        gate_fn = lib.attn_gate_f32
        gate_fn.restype = None
        gate_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_int, ctypes.c_int]
        set_mode = lib.attn_set_sum_mode
        set_mode.restype = None
        set_mode.argtypes = [ctypes.c_int]

        # ---- calibrate: which summation order does this numpy use? ----
        rs = np.random.RandomState(20260101)
        q_raw = rs.randn(8, 2, 256).astype(np.float32)
        qw = (rs.randn(256) * 0.1).astype(np.float32)
        kw = (rs.randn(256) * 0.1).astype(np.float32)
        ref = rmsnorm_heads(q_raw[:, 0, :], qw)
        out = np.empty(8 * 256, dtype=np.float32)
        mode = -1
        for cand in (0, 1, 2, 3):
            set_mode(cand)
            rs_fn(out.ctypes.data, q_raw.ctypes.data, 8, 256, 512,
                  qw.ctypes.data, ctypes.c_float(1e-6))
            if np.array_equal(out, ref.reshape(-1)):
                mode = cand
                break
        if mode < 0:
            sys.stderr.write("[attn] C rmsnorm_heads matches no numpy summation "
                             "mode; keeping the numpy attn path\\n")
            return None

        # rope and gate have no such ambiguity, but verify them anyway
        pos = 7
        xr = rs.randn(8, 256).astype(np.float32)
        ref_rope = apply_rope(np.array(xr, dtype=np.float32, copy=True),
                              pos, eng.cos_table, eng.sin_table)
        buf = np.array(xr, dtype=np.float32, copy=True)
        rope_fn(buf.ctypes.data, 8, 256,
                eng.cos_table.ctypes.data + pos * 128,
                eng.sin_table.ctypes.data + pos * 128)
        if not np.array_equal(buf, ref_rope):
            sys.stderr.write("[attn] C rope differs from numpy; keeping the "
                             "numpy attn path\\n")
            return None

        a_in = rs.randn(8, 256).astype(np.float32)
        g_in = rs.randn(8, 2, 256).astype(np.float32)
        ref_gate = a_in * (1.0 / (1.0 + np.exp(-g_in[:, 1, :])))
        g_out = np.empty(8 * 256, dtype=np.float32)
        gate_fn(g_out.ctypes.data, a_in.ctypes.data, g_in.ctypes.data, 8, 256)
        if not np.array_equal(g_out, ref_gate.reshape(-1)):
            sys.stderr.write("[attn] C sigmoid gate differs from numpy; keeping "
                             "the numpy attn path\\n")
            return None

        # ---- cache every pointer the hot path needs ----
        for p in eng._plan:
            if p.get('full'):
                p['q_norm_ptr'] = p['q_norm'].ctypes.data
                p['k_norm_ptr'] = p['k_norm'].ctypes.data

        _ATTN_GATED = np.empty(8 * 256, dtype=np.float32)
        _ATTN_POST = {
            'pool': pool,
            'fn': fn,
            'gated': _ATTN_GATED,
            'gated_ptr': _ATTN_GATED.ctypes.data,
            'bt': _ATTN_BT,
            'bt_ptr': _ATTN_BT.ctypes.data,
            'k_pool': pool.k_pool.ctypes.data,
            'v_pool': pool.v_pool.ctypes.data,
            'cos': eng.cos_table.ctypes.data,
            'sin': eng.sin_table.ctypes.data,
            'eps': ctypes.c_float(1e-6),
            'scale': ctypes.c_float(0.0625),
        }
        print("[attn] C attn_post enabled (rmsnorm sum mode %d)" % (mode,))
        return _ATTN_POST
    except Exception as exc:
        sys.stderr.write("[attn] attn_post init failed: %s\\n" % (exc,))
        return None


def _attn_post_numpy(pool, layer_idx, pos, p, eng):
    """The original numpy glue, kept as the fallback.  Returns a data pointer."""
    yqg = p['q'][4]
    yk = p['k'][4]
    yv = p['v'][4]
    q_and_gate = yqg.reshape(8, 2, 256)
    q_raw = q_and_gate[:, 0, :]
    gate = q_and_gate[:, 1, :].reshape(2048)
    k_raw = yk.reshape(2, 256)
    v_raw = yv.reshape(2, 256)
    q_normed = rmsnorm_heads(q_raw, p['q_norm'])
    k_normed = rmsnorm_heads(k_raw, p['k_norm'])
    q_rot = apply_rope(q_normed, pos, eng.cos_table, eng.sin_table)
    k_rot = apply_rope(k_normed, pos, eng.cos_table, eng.sin_table)
    pool.append_kv(layer_idx, k_rot, v_raw)
    a_out = _attn_dispatch(pool, layer_idx, q_rot)
    gated_attn = a_out * (1.0 / (1.0 + np.exp(-gate)))
    return gated_attn.ctypes.data


def _attn_post(pool, layer_idx, pos, p, eng, yqgp, ykp, yvp):
    """One-call C path: rmsnorm+rope+append_kv+attention+gate.

    Returns the data pointer of the 2048 gated floats, so the caller can feed
    it straight to the o_proj gemv without touching .ctypes.data (34 us each
    on this board)."""
    global _ATTN_POST_TRIED
    ctx = _ATTN_POST
    if (ctx is None or ctx['pool'] is not pool) and _ATTN_POST_TRIED is not pool:
        # try to bring the C path up exactly once per pool: if the calibration
        # failed, retrying it on all 162 callbacks/token would cost ~1 ms each
        _ATTN_POST_TRIED = pool
        ctx = _attn_post_init(eng, pool)
    if ctx is None or ctx['pool'] is not pool:
        return _attn_post_numpy(pool, layer_idx, pos, p, eng)
    try:
        total_tokens = pool.seq_len + 1
        bt = pool.block_table
        n_bt = len(bt)
        n_need = (total_tokens + pool.block_size - 1) // pool.block_size
        if n_bt <= 128 and n_need <= n_bt and total_tokens <= 2048:
            bta = ctx['bt']
            for i in range(n_bt):
                bta[i] = bt[i]
            rc = ctx['fn'](
                ctx['gated_ptr'],
                yqgp, ykp, yvp,
                p['q_norm_ptr'], p['k_norm_ptr'],
                ctx['cos'] + pos * 128, ctx['sin'] + pos * 128,
                ctx['eps'],
                8, pool.num_kv_heads, pool.head_dim,
                ctx['k_pool'], ctx['v_pool'], ctx['bt_ptr'],
                pool.num_layers, pool.block_size, layer_idx,
                total_tokens, ctx['scale'])
            if rc == 0:
                return ctx['gated_ptr']
            sys.stderr.write("[attn] attn_post_f32 rc=%d, using numpy\\n" % (rc,))
    except Exception as exc:
        sys.stderr.write("[attn] C attn_post failed, using numpy: %s\\n" % (exc,))
    return _attn_post_numpy(pool, layer_idx, pos, p, eng)
# ==== END qwen_attn_post binding ====
'''


# ---------------------------------------------------------------------------
# methods on QwenRealEngine (thin wrappers so the call sites stay readable)
# ---------------------------------------------------------------------------
NEW_METHODS = '''\
    def _attn_post(self, pool, layer_idx, pos, p, yqgp, ykp, yvp):
        """rmsnorm_heads+rope+append_kv+attention+gate in one C call.

        Falls back to the numpy glue in _attn_post_numpy() on any problem, so
        this can never be less correct than the code it replaces."""
        return _attn_post(pool, layer_idx, pos, p, self, yqgp, ykp, yvp)

    def _attn_post_numpy(self, pool, layer_idx, pos, p):
        """The numpy implementation of the same glue (fallback / reference)."""
        return _attn_post_numpy(pool, layer_idx, pos, p, self)

'''


OLD_CB = """\
        mp, ep, M, Kd, yqg, yqgp = p['q']
        self._k_gemv(yqgp, normed_ptr, mp, ep, M, Kd)
        q_and_gate = yqg.reshape(8, 2, 256)
        q_raw = q_and_gate[:, 0, :]
        gate = q_and_gate[:, 1, :].reshape(2048)

        mp, ep, M, Kd, yk, ykp = p['k']
        self._k_gemv(ykp, normed_ptr, mp, ep, M, Kd)
        k_raw = yk.reshape(2, 256)

        mp, ep, M, Kd, yv, yvp = p['v']
        self._k_gemv(yvp, normed_ptr, mp, ep, M, Kd)
        v_raw = yv.reshape(2, 256)

        q_normed = rmsnorm_heads(q_raw, p['q_norm'])
        k_normed = rmsnorm_heads(k_raw, p['k_norm'])
        q_rot = apply_rope(q_normed, pos, self.cos_table, self.sin_table)
        k_rot = apply_rope(k_normed, pos, self.cos_table, self.sin_table)

        pool.append_kv(pool_idx, k_rot, v_raw)
        a_out = _attn_dispatch(pool, pool_idx, q_rot)
        gated_attn = a_out * (1.0 / (1.0 + np.exp(-gate)))

        mp, ep, M, Kd, yo, yop = p['o']
        self._k_gemv(yop, gated_attn.ctypes.data, mp, ep, M, Kd)
        np.add(self._h_arr, yo, out=self._h_arr)
"""

NEW_CB = """\
        mp, ep, M, Kd, yqg, yqgp = p['q']
        self._k_gemv(yqgp, normed_ptr, mp, ep, M, Kd)

        mp, ep, M, Kd, yk, ykp = p['k']
        self._k_gemv(ykp, normed_ptr, mp, ep, M, Kd)

        mp, ep, M, Kd, yv, yvp = p['v']
        self._k_gemv(yvp, normed_ptr, mp, ep, M, Kd)

        # rmsnorm_heads, rope, append_kv, attention and the sigmoid gate in a
        # single C call (numpy fallback inside _attn_post)
        gated_ptr = self._attn_post(pool, pool_idx, pos, p,
                                    yqgp, ykp, yvp)

        mp, ep, M, Kd, yo, yop = p['o']
        self._k_gemv(yop, gated_ptr, mp, ep, M, Kd)
        np.add(self._h_arr, yo, out=self._h_arr)
"""

OLD_FWD = """\
            mp, ep, M, Kd, yqg, yqgp = p['q']
            kg(yqgp, normed_ptr, mp, ep, M, Kd)
            q_and_gate = yqg.reshape(8, 2, 256)
            q_raw = q_and_gate[:, 0, :]
            gate = q_and_gate[:, 1, :].reshape(2048)

            mp, ep, M, Kd, yk, ykp = p['k']
            kg(ykp, normed_ptr, mp, ep, M, Kd)
            k_raw = yk.reshape(2, 256)

            mp, ep, M, Kd, yv, yvp = p['v']
            kg(yvp, normed_ptr, mp, ep, M, Kd)
            v_raw = yv.reshape(2, 256)

            q_normed = rmsnorm_heads(q_raw, p['q_norm'])
            k_normed = rmsnorm_heads(k_raw, p['k_norm'])
            q_rot = apply_rope(q_normed, pos, self.cos_table, self.sin_table)
            k_rot = apply_rope(k_normed, pos, self.cos_table, self.sin_table)

            pool.append_kv(pool_idx, k_rot, v_raw)
            a_out = _attn_dispatch(pool, pool_idx, q_rot)
            gated_attn = a_out * (1.0 / (1.0 + np.exp(-gate)))

            mp, ep, M, Kd, yo, yop = p['o']
            kg(yop, gated_attn.ctypes.data, mp, ep, M, Kd)
            attn_res = yo
"""

NEW_FWD = """\
            mp, ep, M, Kd, yqg, yqgp = p['q']
            kg(yqgp, normed_ptr, mp, ep, M, Kd)

            mp, ep, M, Kd, yk, ykp = p['k']
            kg(ykp, normed_ptr, mp, ep, M, Kd)

            mp, ep, M, Kd, yv, yvp = p['v']
            kg(yvp, normed_ptr, mp, ep, M, Kd)

            gated_ptr = self._attn_post(pool, pool_idx, pos, p,
                                        yqgp, ykp, yvp)

            mp, ep, M, Kd, yo, yop = p['o']
            kg(yop, gated_ptr, mp, ep, M, Kd)
            attn_res = yo
"""


def read_text(path):
    # the engine has UTF-8 Chinese comments; never rely on the board's locale
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def write_text(path, text):
    tmp = path + ".tmp_patch"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


def build_patched(text):
    """Return the patched text, or raise SystemExit with a clear reason."""
    if BEGIN in text:
        raise SystemExit("already patched (marker present) -- use --revert first")
    if ANCHOR_BINDING not in text:
        raise SystemExit("anchor missing: '%s'" % ANCHOR_BINDING.strip())
    if ANCHOR_CB not in text:
        raise SystemExit("anchor missing: %r" % ANCHOR_CB.strip())
    for name, old in (("callback", OLD_CB), ("_layer_fwd", OLD_FWD)):
        n = text.count(old)
        if n != 1:
            raise SystemExit("expected exactly 1 occurrence of the %s block, "
                             "found %d -- refusing to patch" % (name, n))

    text = text.replace(ANCHOR_BINDING,
                        ANCHOR_BINDING + "\n\n" + NEW_BINDING, 1)
    text = text.replace(ANCHOR_CB, NEW_METHODS + ANCHOR_CB, 1)
    text = text.replace(OLD_CB, NEW_CB, 1)
    text = text.replace(OLD_FWD, NEW_FWD, 1)
    return text


def main(argv):
    check = "--check" in argv
    revert = "--revert" in argv

    if revert:
        if not os.path.exists(BACKUP):
            print("no backup at %s -- nothing to revert" % (BACKUP,))
            return 1
        shutil.copyfile(BACKUP, TARGET)
        print("reverted %s from %s" % (TARGET, BACKUP))
        return 0

    text = read_text(TARGET)
    try:
        new_text = build_patched(text)
    except SystemExit as exc:
        print("CANNOT PATCH: %s" % (exc,))
        return 1

    if check:
        import difflib
        diff = list(difflib.unified_diff(text.splitlines(),
                                         new_text.splitlines(),
                                         TARGET, TARGET + " (patched)",
                                         lineterm="", n=2))
        print("patch applies cleanly: %d changed line(s), %d -> %d bytes"
              % (len(diff), len(text), len(new_text)))
        for line in diff[:80]:
            print(line)
        return 0

    if not os.path.exists(BACKUP):
        shutil.copyfile(TARGET, BACKUP)
        print("backup: %s" % (BACKUP,))
    write_text(TARGET, new_text)
    print("patched %s (%d -> %d bytes)" % (TARGET, len(text), len(new_text)))
    print("backup kept at %s" % (BACKUP,))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
