import os
import sys
import time
import atexit
import json
import mmap
import struct
import ctypes
import numpy as np

# Physical Hardware Constants on Zynq-7020
CORE_A_BASE = 0x43C00000
DMA0_BASE   = 0x40400000
TX_BUF_ADDR = 0x10001000
RX_BUF_ADDR = 0x10020000
DEV_MEM     = "/dev/mem"
PAGE_SIZE   = 4096


class LinearLayerPlanC(ctypes.Structure):
    _fields_ = [
        ('in_ln_w', ctypes.c_void_p),
        ('mant_qkv', ctypes.c_void_p),
        ('exp_qkv', ctypes.c_void_p),
        ('M_qkv', ctypes.c_int),
        ('K_qkv', ctypes.c_int),
        ('conv_w', ctypes.c_void_p),
        ('conv_s', ctypes.c_void_p),
        ('wa', ctypes.c_void_p),
        ('wb', ctypes.c_void_p),
        ('nea', ctypes.c_void_p),
        ('dt', ctypes.c_void_p),
        ('state', ctypes.c_void_p),
        ('mant_z', ctypes.c_void_p),
        ('exp_z', ctypes.c_void_p),
        ('M_z', ctypes.c_int),
        ('K_z', ctypes.c_int),
        ('norm_w', ctypes.c_void_p),
        ('mant_o', ctypes.c_void_p),
        ('exp_o', ctypes.c_void_p),
        ('M_o', ctypes.c_int),
        ('K_o', ctypes.c_int),
        ('post_ln_w', ctypes.c_void_p),
        ('mant_gate', ctypes.c_void_p),
        ('exp_gate', ctypes.c_void_p),
        ('M_gate', ctypes.c_int),
        ('K_gate', ctypes.c_int),
        ('mant_up', ctypes.c_void_p),
        ('exp_up', ctypes.c_void_p),
        ('M_up', ctypes.c_int),
        ('K_up', ctypes.c_int),
        ('mant_down', ctypes.c_void_p),
        ('exp_down', ctypes.c_void_p),
        ('M_down', ctypes.c_int),
        ('K_down', ctypes.c_int),
    ]


class MlpPlanC(ctypes.Structure):
    _fields_ = [
        ('post_ln_w', ctypes.c_void_p),
        ('mant_gate', ctypes.c_void_p),
        ('exp_gate', ctypes.c_void_p),
        ('M_gate', ctypes.c_int),
        ('K_gate', ctypes.c_int),
        ('mant_up', ctypes.c_void_p),
        ('exp_up', ctypes.c_void_p),
        ('M_up', ctypes.c_int),
        ('K_up', ctypes.c_int),
        ('mant_down', ctypes.c_void_p),
        ('exp_down', ctypes.c_void_p),
        ('M_down', ctypes.c_int),
        ('K_down', ctypes.c_int),
    ]


class FastBoardHw:
    def __init__(self):
        if not os.path.exists(DEV_MEM) or os.environ.get("QWEN_CPU_ONLY", "1") == "1":
            self.mock = True
            return
        self.mock = False
        try:
            self.f = open(DEV_MEM, "r+b")
            self.mm_core = mmap.mmap(self.f.fileno(), PAGE_SIZE, offset=CORE_A_BASE)
            self.mm_dma  = mmap.mmap(self.f.fileno(), PAGE_SIZE, offset=DMA0_BASE)
            self.mm_tx   = mmap.mmap(self.f.fileno(), PAGE_SIZE, offset=TX_BUF_ADDR)
            self.mm_rx   = mmap.mmap(self.f.fileno(), PAGE_SIZE, offset=RX_BUF_ADDR)
        except Exception:
            self.mock = True
            return

    def reset_state(self):
        if getattr(self, 'mock', False):
            return
        self.mm_core[0x10:0x14] = struct.pack("<I", 1) # MODE_RESET
        self.mm_core[0x18:0x1C] = struct.pack("<I", 8) # ALL HEADS
        self.mm_core[0x00:0x04] = struct.pack("<I", 1) # START
        time.sleep(0.001)

    def linear_attn_step(self, q, k, v, alpha=0.5, beta=0.5, head_idx=0):
        if getattr(self, 'mock', False):
            return np.zeros(64, dtype=np.float32)
        raw_tx = bytearray(512)
        q_i16 = (np.clip(q * 16384.0, -32768, 32767).astype(np.int16)).view(np.uint16)
        k_i16 = (np.clip(k * 16384.0, -32768, 32767).astype(np.int16)).view(np.uint16)
        v_i16 = (np.clip(v * 16384.0, -32768, 32767).astype(np.int16)).view(np.uint16)
        for i in range(64):
            struct.pack_into("<HHHxx", raw_tx, i * 8, int(q_i16[i]), int(k_i16[i]), int(v_i16[i]))
        self.mm_tx[:512] = raw_tx

        self.mm_core[0x10:0x14] = struct.pack("<I", 0) # MODE_DECODE
        self.mm_core[0x18:0x1C] = struct.pack("<I", head_idx)
        self.mm_core[0x20:0x24] = struct.pack("<I", int(np.clip(alpha, 0.0, 1.0) * 65535))
        self.mm_core[0x28:0x2C] = struct.pack("<I", int(np.clip(beta, 0.0, 1.0) * 65535))
        self.mm_core[0x00:0x04] = struct.pack("<I", 1) # START

        self.mm_dma[0x30:0x34] = struct.pack("<I", 1)
        self.mm_dma[0x48:0x4C] = struct.pack("<I", RX_BUF_ADDR)
        self.mm_dma[0x58:0x5C] = struct.pack("<I", 256)

        self.mm_dma[0x00:0x04] = struct.pack("<I", 1)
        self.mm_dma[0x18:0x1C] = struct.pack("<I", TX_BUF_ADDR)
        self.mm_dma[0x28:0x2C] = struct.pack("<I", 512)

        for _ in range(50):
            s2mm = struct.unpack("<I", self.mm_dma[0x34:0x38])[0]
            mm2s = struct.unpack("<I", self.mm_dma[0x04:0x08])[0]
            if (s2mm & 0x02) and (mm2s & 0x02):
                break
            time.sleep(0.0005)

        rx_raw = self.mm_rx[:256]
        rx_u32 = struct.unpack("<64I", rx_raw)
        out_int16 = [(w & 0xFFFF) if (w & 0x8000) == 0 else ((w & 0xFFFF) - 65536) for w in rx_u32]
        return np.array(out_int16, dtype=np.float32) / 16384.0

    def close(self):
        if getattr(self, 'mock', False):
            return
        self.mm_core.close()
        self.mm_dma.close()
        self.mm_tx.close()
        self.mm_rx.close()
        self.f.close()


# ---------------------------------------------------------------------------
# CPU-only harness switch.
#
# The engine constructor touches PL registers at 0x43C00000 / 0x40400000.  With
# the CPU-side bitstream those addresses are not mapped, so /dev/mem mmap gives
# SIGBUS and the process dies with no traceback.  The CPU inference path never
# uses self.hw, so a no-op stub is numerically identical.
#
# Set QWEN_CPU_ONLY=1 to enable.  Default behaviour is unchanged.
# ---------------------------------------------------------------------------
if os.environ.get("QWEN_CPU_ONLY", "") == "1":
    class FastBoardHw(object):
        def __init__(self, *a, **k):
            self.mock = True

        def reset_state(self, *a, **k):
            pass

        def linear_attn_step(self, *a, **k):
            return None

        def close(self, *a, **k):
            pass


class CTokenizer:
    def __init__(self, bin_path="/mnt/qwen/tokenizer_64k.bin"):
        if not os.path.exists(bin_path):
            alt_path = r"D:\fpga2026\work\vocab\tokenizer_64k.bin"
            if os.path.exists(alt_path):
                bin_path = alt_path

        lib_candidates = [
            "/mnt/qwen/libtokenizer.so",
            os.path.join(os.path.dirname(__file__), "libtokenizer.dll"),
            os.path.join(os.path.dirname(__file__), "libtokenizer.so"),
            r"D:\fpga2026\work\driver\libtokenizer.dll"
        ]
        self.lib = None
        for cand in lib_candidates:
            if os.path.exists(cand):
                try:
                    self.lib = ctypes.CDLL(cand)
                    break
                except Exception:
                    pass

        self.tok_ptr = None
        if self.lib:
            self.lib.tokenizer_create.restype = ctypes.c_void_p
            self.lib.tokenizer_create.argtypes = [ctypes.c_char_p]
            self.lib.tokenizer_destroy.argtypes = [ctypes.c_void_p]
            self.lib.tokenizer_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int]
            self.lib.tokenizer_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            if os.path.exists(bin_path):
                self.tok_ptr = self.lib.tokenizer_create(bin_path.encode('utf-8'))

    def encode(self, text):
        if not self.tok_ptr:
            return []
        tokens_arr = (ctypes.c_int * 1024)()
        n = self.lib.tokenizer_encode(self.tok_ptr, text.encode('utf-8'), tokens_arr, 1024)
        return [tokens_arr[i] for i in range(n)]

    def decode(self, token_id):
        if not self.tok_ptr:
            return ""
        buf = ctypes.create_string_buffer(256)
        self.lib.tokenizer_decode(self.tok_ptr, token_id, buf, 256)
        return buf.value.decode('utf-8', errors='replace')

    def __del__(self):
        if getattr(self, 'tok_ptr', None) and getattr(self, 'lib', None):
            self.lib.tokenizer_destroy(self.tok_ptr)
            self.tok_ptr = None


# ==== BEGIN qwen_attn binding (patch_engine_attn.py) ====
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
        sys.stderr.write("[attn] libqwen_attn.so load failed: %s\n" % (exc,))
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
        sys.stderr.write("[attn] C path failed, using numpy: %s\n" % (exc,))
        return pool.compute_attention(layer_idx, q)
# ==== END qwen_attn binding ====


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
                             "mode; keeping the numpy attn path\n")
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
                             "numpy attn path\n")
            return None

        a_in = rs.randn(8, 256).astype(np.float32)
        g_in = rs.randn(8, 2, 256).astype(np.float32)
        ref_gate = a_in * (1.0 / (1.0 + np.exp(-g_in[:, 1, :])))
        g_out = np.empty(8 * 256, dtype=np.float32)
        gate_fn(g_out.ctypes.data, a_in.ctypes.data, g_in.ctypes.data, 8, 256)
        if not np.array_equal(g_out, ref_gate.reshape(-1)):
            sys.stderr.write("[attn] C sigmoid gate differs from numpy; keeping "
                             "the numpy attn path\n")
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
        sys.stderr.write("[attn] attn_post init failed: %s\n" % (exc,))
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
            sys.stderr.write("[attn] attn_post_f32 rc=%d, using numpy\n" % (rc,))
    except Exception as exc:
        sys.stderr.write("[attn] C attn_post failed, using numpy: %s\n" % (exc,))
    return _attn_post_numpy(pool, layer_idx, pos, p, eng)
# ==== END qwen_attn_post binding ====


# ---------------------------------------------------------------------------
# KV-pool and RoPE sizing.
#
# These are module-level so chat_v3.py can resize them (--ctx N) BEFORE
# QwenRealEngine() is constructed.  They must move together: a position past
# the end of cos_table/sin_table is an out-of-bounds read in the C attention
# path (which does `cos_table + pos * 128` with no bounds check) and an
# IndexError in the numpy path.
#
# Cost of the pool: one token of KV is
#     6 full-attn layers * 2 kv heads * 256 head_dim * 4 B * (k and v)
#   = 24,576 B
# so 4096 tokens = 100.7 MB, 8192 tokens = 201.3 MB.  The board has 1006 MB
# of which ~850 MB is free, and the 614 MB weight file lives in page cache,
# so a bigger pool directly steals from the page cache and makes the next
# cold prefill slower.  4096 is the balance point.
# ---------------------------------------------------------------------------
PAGED_BLOCK_SIZE = 16
PAGED_NUM_BLOCKS = 256          # 256 * 16 = 4096 tokens of KV (~101 MB)
ROPE_MAX_SEQ = 4096             # must be >= PAGED_NUM_BLOCKS * PAGED_BLOCK_SIZE


class PagedAttentionPool:
    """
    vLLM-style Dynamic PagedAttention Memory Pool on Zynq DDR3.
    Pre-allocates contiguous block pool to prevent memory fragmentation and OOM.
    """
    def __init__(self, num_layers=6, num_kv_heads=2, head_dim=256,
                 block_size=PAGED_BLOCK_SIZE, num_blocks=PAGED_NUM_BLOCKS):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks

        # num_blocks * block_size tokens; 24,576 B per token (see note above)
        self.k_pool = np.zeros((num_blocks, num_layers, num_kv_heads, block_size, head_dim), dtype=np.float32)
        self.v_pool = np.zeros((num_blocks, num_layers, num_kv_heads, block_size, head_dim), dtype=np.float32)
        self.reset()

    def reset(self):
        self.free_blocks = list(range(self.num_blocks - 1, -1, -1))
        self.block_table = []
        self.seq_len = 0

    def allocate_slot(self):
        offset = self.seq_len % self.block_size
        if offset == 0:
            if not self.free_blocks:
                raise RuntimeError("PagedAttentionPool: Out of physical memory blocks!")
            self.block_table.append(self.free_blocks.pop())
        return self.block_table[-1], offset

    def append_kv(self, layer_idx, k, v):
        curr_block = self.block_table[-1]
        offset = self.seq_len % self.block_size
        self.k_pool[curr_block, layer_idx, :, offset, :] = k
        self.v_pool[curr_block, layer_idx, :, offset, :] = v

    def step_done(self):
        self.seq_len += 1

    def compute_attention(self, layer_idx, q):
        # q: [8, 256] -> 8 query heads, 2 KV heads (4:1 GQA)
        total_tokens = self.seq_len + 1
        if len(self.block_table) == 1:
            b = self.block_table[0]
            k_all = self.k_pool[b, layer_idx, :, :total_tokens, :]
            v_all = self.v_pool[b, layer_idx, :, :total_tokens, :]
        else:
            k_blocks = [self.k_pool[b, layer_idx] for b in self.block_table]
            v_blocks = [self.v_pool[b, layer_idx] for b in self.block_table]
            k_all = np.concatenate(k_blocks, axis=1)[:, :total_tokens, :]
            v_all = np.concatenate(v_blocks, axis=1)[:, :total_tokens, :]

        # Broadcast 2 KV heads to 8 query heads (GQA 4:1)
        k_exp = np.repeat(k_all, 4, axis=0) # [8, T, 256]
        v_exp = np.repeat(v_all, 4, axis=0) # [8, T, 256]

        # Scaled Dot-Product Attention: 1/sqrt(256) = 0.0625
        logits = np.einsum('hd,htd->ht', q, k_exp) * 0.0625
        max_l = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True) # [8, T]

        # Context output: [8, 256] -> flattened to [2048]
        out = np.einsum('ht,htd->hd', probs, v_exp).reshape(-1)
        return out


def precompute_rope_freqs(rotary_dim=64, max_seq=2048, theta=10000000.0):
    freqs = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    t = np.arange(max_seq, dtype=np.float32)
    freqs_table = np.outer(t, freqs)
    return np.cos(freqs_table), np.sin(freqs_table)


def apply_rope(x, pos, cos_table, sin_table):
    # x: [num_heads, head_dim] (partial rotary on first 64 dims)
    cos = cos_table[pos] # [32]
    sin = sin_table[pos] # [32]
    x_rot = x[:, :64]
    x_pass = x[:, 64:]
    x1 = x_rot[:, :32]
    x2 = x_rot[:, 32:]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x2 * cos + x1 * sin
    return np.concatenate([rot_x1, rot_x2, x_pass], axis=-1)


def rmsnorm_heads(x, weight, eps=1e-6):
    # x: [num_heads, head_dim], weight: [head_dim]
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * (1.0 + weight)


class LayerStateDict(dict):
    """Dictionary supporting both layer id (e.g. 0, 1, 2, 4...) and 0-based linear index (0..17)."""
    def __init__(self, linear_layers, shape, dtype=np.float32):
        super().__init__()
        self.linear_layers = list(linear_layers)
        for l in self.linear_layers:
            self[l] = np.zeros(shape, dtype=dtype)

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        if isinstance(key, int) and 0 <= key < len(self.linear_layers):
            return super().__getitem__(self.linear_layers[key])
        raise KeyError(key)


class QwenRealEngine:
    def __init__(self):
        print("[*] Initializing Qwen3.5-0.8B Real Hardware Engine...")
        self.hw = FastBoardHw()
        self.hw.reset_state()

        print("[*] Loading NEON SIMD Accelerated Kernel Library...")
        kernel_candidates = [
            "/mnt/qwen/libqwen_kernels.so",
            os.path.join(os.path.dirname(__file__), "libqwen_kernels.dll"),
            os.path.join(os.path.dirname(__file__), "libqwen_kernels.so"),
            r"D:\fpga2026\work\driver\libqwen_kernels.dll"
        ]
        self.kernels = None
        for cand in kernel_candidates:
            if os.path.exists(cand):
                try:
                    self.kernels = ctypes.CDLL(cand)
                    if hasattr(self.kernels, 'init_kernels'):
                        self.kernels.init_kernels()
                    break
                except Exception:
                    pass

        # Explicitly declare ctypes argtypes for C ABI safety
        if self.kernels:
            if hasattr(self.kernels, 'rmsnorm'):
                self.kernels.rmsnorm.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_float
                ]
            if hasattr(self.kernels, 'bfp8_gemv'):
                self.kernels.bfp8_gemv.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int
                ]
            if hasattr(self.kernels, 'swiglu'):
                self.kernels.swiglu.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_int
                ]
            if hasattr(self.kernels, 'lm_head_argmax'):
                self.kernels.lm_head_argmax.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int
                ]
                self.kernels.lm_head_argmax.restype = ctypes.c_int

            if hasattr(self.kernels, 'causal_conv1d_update_silu'):
                self.kernels.causal_conv1d_update_silu.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int
                ]
            if hasattr(self.kernels, 'qk_l2norm'):
                self.kernels.qk_l2norm.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int
                ]
            if hasattr(self.kernels, 'gdn_recurrent_step'):
                self.kernels.gdn_recurrent_step.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int
                ]
            if hasattr(self.kernels, 'rmsnorm_gated'):
                self.kernels.rmsnorm_gated.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_float
                ]
            if hasattr(self.kernels, 'forward_linear_layer'):
                self.kernels.forward_linear_layer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            if hasattr(self.kernels, 'forward_mlp'):
                self.kernels.forward_mlp.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        def resolve_file(board_path, host_rel_or_abs):
            if os.path.exists(board_path):
                return board_path
            if os.path.isabs(host_rel_or_abs) and os.path.exists(host_rel_or_abs):
                return host_rel_or_abs
            d_path = os.path.join(r"D:\fpga2026\work", host_rel_or_abs)
            if os.path.exists(d_path):
                return d_path
            return board_path

        tok_path = resolve_file("/mnt/qwen/tokenizer_64k.bin", "vocab/tokenizer_64k.bin")
        print("[*] Loading 64K C-BPE Tokenizer from {}...".format(tok_path))
        self.tok = CTokenizer(tok_path)

        embed_path = resolve_file('/mnt/qwen/embedding_64k.bin', 'vocab/embedding_64k.bin')
        print("[*] Memmapping 128MB 64K Embedding Matrix from {}...".format(embed_path))
        self.embed_raw = np.memmap(embed_path, dtype=np.uint16, mode='r', shape=(65536, 1024))

        weights_path = resolve_file('/mnt/qwen/qwen35_weights_64k.bin', 'qwen35_weights_64k.bin')
        print("[*] Memmapping 614MB 24-Layer BFP8 Weights Matrix from {}...".format(weights_path))
        self.weights_f = open(weights_path, 'rb')
        if os.name == 'nt':
            self.weights_mm = mmap.mmap(self.weights_f.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self.weights_mm = mmap.mmap(self.weights_f.fileno(), 0, prot=mmap.PROT_READ)

        manifest_path = resolve_file('/mnt/qwen/qwen35_weights_64k_manifest.json', 'qwen35_weights_64k_manifest.json')
        print("[*] Parsing Model Architecture Manifest (320 Tensors) from {}...".format(manifest_path))
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.num_layers = 24
        self.hidden_dim = 1024
        self.full_attn_layers = {3, 7, 11, 15, 19, 23}
        self.full_layer_map = {3: 0, 7: 1, 11: 2, 15: 3, 19: 4, 23: 5}
        self.linear_attn_layers = [l for l in range(self.num_layers) if l not in self.full_attn_layers]

        # Pre-load Conv1D weights and RMSNormGated weights for all 18 linear attention layers
        print("[*] Pre-loading Conv1D and RMSNormGated Weights for 18 Linear Layers...")
        self.conv_weights = {}
        self.norm_weights = {}
        self.gate_w = {}
        for l in self.linear_attn_layers:
            prefix = "model.language_model.layers.{}.".format(l)
            raw_conv, _ = self.get_tensor_data(prefix + "linear_attn.conv1d.weight")
            self.conv_weights[l] = np.frombuffer(raw_conv, dtype=np.float32).reshape(6144, 4)
            raw_norm, _ = self.get_tensor_data(prefix + "linear_attn.norm.weight")
            self.norm_weights[l] = np.frombuffer(raw_norm, dtype=np.float32)
            # 门控用的小张量每 token 都要读 5 次，这里一次性建好（含 -exp(A_log) 预处理），
            # 省掉每层 4 次 get_tensor_data 的 bytes 拷贝和 numpy 包装开销。
            raw_a, _ = self.get_tensor_data(prefix + "linear_attn.in_proj_a.weight")
            raw_b, _ = self.get_tensor_data(prefix + "linear_attn.in_proj_b.weight")
            raw_al, _ = self.get_tensor_data(prefix + "linear_attn.A_log")
            raw_dt, _ = self.get_tensor_data(prefix + "linear_attn.dt_bias")
            self.gate_w[l] = (
                np.frombuffer(raw_a, dtype=np.float32).reshape(16, 1024),
                np.frombuffer(raw_b, dtype=np.float32).reshape(16, 1024),
                -np.exp(np.frombuffer(raw_al, dtype=np.float32)),
                np.frombuffer(raw_dt, dtype=np.float32).copy(),
            )

        # Initialize Conv1D 3-tap history buffer and Gated DeltaNet state matrices
        # conv_state is [3][channels] so the C kernel can vectorise 4 channels at
        # a time (each tap is contiguous).  It was [channels][3] before.
        self.conv_states = LayerStateDict(self.linear_attn_layers, (3, 6144), dtype=np.float32)
        self.recurrent_states = LayerStateDict(self.linear_attn_layers, (16, 128, 128), dtype=np.float32)

        print("[*] Initializing PagedAttention Dynamic Memory Pool "
              "(%d Blocks / %d Context / %.1f MB)..."
              % (PAGED_NUM_BLOCKS, PAGED_NUM_BLOCKS * PAGED_BLOCK_SIZE,
                 2.0 * PAGED_NUM_BLOCKS * 6 * 2 * PAGED_BLOCK_SIZE * 256 * 4 / 1048576.0))
        self.paged_attn = PagedAttentionPool(num_layers=6, num_kv_heads=2, head_dim=256,
                                             block_size=PAGED_BLOCK_SIZE,
                                             num_blocks=PAGED_NUM_BLOCKS)
        self.cos_table, self.sin_table = precompute_rope_freqs(rotary_dim=64,
                                                               max_seq=ROPE_MAX_SEQ)

        self._build_fast_path()

        print("[*] Engine Initialized! Ready for 100% Real Neural Network Inference.\n")

    def reset_state(self):
        self.hw.reset_state()
        self.paged_attn.reset()
        for l in self.linear_attn_layers:
            self.conv_states[l].fill(0.0)
            self.recurrent_states[l].fill(0.0)

    # ------------------------------------------------------------------
    # Fast path.
    #
    # ndarray.ctypes.data costs ~34 us on this board (numpy 1.8.2 rebuilds
    # __array_interface__ on every access) and the old per-token loop called
    # it 1127 times, i.e. ~39 ms/token.  Everything the loop needs is resolved
    # once here and cached as a plain int.  Activation buffers are persistent
    # too, so the loop allocates nothing.
    # ------------------------------------------------------------------
    def _f32_view(self, name):
        v = self._f32_cache.get(name)
        if v is None:
            info = self.manifest['tensors'][name]
            raw = np.frombuffer(self.weights_mm, dtype=np.uint8,
                                count=info['byte_size'], offset=info['file_offset'])
            v = raw.view(np.float32)
            self._f32_cache[name] = v
        return v

    def _gemv_ent(self, name):
        v = self._gemv_cache.get(name)
        if v is None:
            info = self.manifest['tensors'][name]
            M, K = info['shape']
            raw = np.frombuffer(self.weights_mm, dtype=np.uint8,
                                count=info['byte_size'], offset=info['file_offset'])
            mant = raw[:M * K].view(np.int8)
            exps = raw[M * K: M * K + (M * K) // 64]
            y = np.empty(M, dtype=np.float32)
            v = (mant.ctypes.data, exps.ctypes.data, M, K, y, y.ctypes.data)
            self._gemv_cache[name] = v
        return v

    def _build_fast_path(self):
        K = self.kernels
        self._f32_cache = {}
        self._gemv_cache = {}
        self._k_rmsnorm = K.rmsnorm
        self._k_gemv = K.bfp8_gemv
        self._k_swiglu = K.swiglu
        self._k_conv = K.causal_conv1d_update_silu
        self._k_qk = K.qk_l2norm
        self._k_gdn = K.gdn_recurrent_step
        self._k_gated = K.rmsnorm_gated
        self._k_gates = K.gdn_gates
        self._k_lm = K.lm_head_argmax

        self._k_rmsnorm.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
        self._k_gemv.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_int, ctypes.c_int]
        self._k_swiglu.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_int]
        self._k_conv.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self._k_qk.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_int]
        self._k_gdn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._k_gated.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self._k_gates.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._k_lm.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_int]
        self._k_lm.restype = ctypes.c_int

        b = {}
        for nm, n in (("h", 1024), ("normed", 1024), ("post_norm", 1024),
                      ("mixed", 6144), ("attn_out", 2048), ("gated_out", 2048),
                      ("intermediate", 3584), ("final_h", 1024),
                      ("alpha", 16), ("beta", 16)):
            a = np.empty(n, dtype=np.float32)
            b[nm] = (a, a.ctypes.data)
        self._b = b
        self._h_arr, self._h_ptr = b["h"]
        self._final_ptr = b["final_h"][1]
        self._final_w_ptr = self._f32_view("model.language_model.norm.weight").ctypes.data
        self._embed_ptr = self.embed_raw.ctypes.data

        self._plan = []
        for l in range(self.num_layers):
            prefix = "model.language_model.layers.{}.".format(l)
            p = {
                'in_ln': self._f32_view(prefix + "input_layernorm.weight").ctypes.data,
                'post_ln': self._f32_view(prefix + "post_attention_layernorm.weight").ctypes.data,
                'mlp_gate': self._gemv_ent(prefix + "mlp.gate_proj.weight"),
                'mlp_up': self._gemv_ent(prefix + "mlp.up_proj.weight"),
                'mlp_down': self._gemv_ent(prefix + "mlp.down_proj.weight"),
                'full': l in self.full_attn_layers,
            }
            if p['full']:
                p['q'] = self._gemv_ent(prefix + "self_attn.q_proj.weight")
                p['k'] = self._gemv_ent(prefix + "self_attn.k_proj.weight")
                p['v'] = self._gemv_ent(prefix + "self_attn.v_proj.weight")
                p['o'] = self._gemv_ent(prefix + "self_attn.o_proj.weight")
                p['q_norm'] = self._f32_view(prefix + "self_attn.q_norm.weight")
                p['k_norm'] = self._f32_view(prefix + "self_attn.k_norm.weight")
                p['pool_idx'] = self.full_layer_map[l]
            else:
                p['qkv'] = self._gemv_ent(prefix + "linear_attn.in_proj_qkv.weight")
                p['z'] = self._gemv_ent(prefix + "linear_attn.in_proj_z.weight")
                p['o'] = self._gemv_ent(prefix + "linear_attn.out_proj.weight")
                p['conv_w'] = self.conv_weights[l].ctypes.data
                p['conv_s'] = self.conv_states[l].ctypes.data
                p['state'] = self.recurrent_states[l].ctypes.data
                wa, wb, nea, dt = self.gate_w[l]
                p['wa'] = wa.ctypes.data
                p['wb'] = wb.ctypes.data
                p['nea'] = nea.ctypes.data
                p['dt'] = dt.ctypes.data
                p['norm_w'] = self.norm_weights[l].ctypes.data
            self._plan.append(p)

        self._has_c_forward = bool(self.kernels and
                                   hasattr(self.kernels, 'forward_linear_layer') and
                                   hasattr(self.kernels, 'forward_mlp'))
        if self._has_c_forward:
            for l in range(self.num_layers):
                p = self._plan[l]
                if not p['full']:
                    c = LinearLayerPlanC()
                    c.in_ln_w = p['in_ln']
                    mp, ep, M, Kd, y, yp = p['qkv']
                    c.mant_qkv = mp; c.exp_qkv = ep; c.M_qkv = M; c.K_qkv = Kd
                    c.conv_w = p['conv_w']
                    c.conv_s = p['conv_s']
                    c.wa = p['wa']
                    c.wb = p['wb']
                    c.nea = p['nea']
                    c.dt = p['dt']
                    c.state = p['state']
                    mp, ep, M, Kd, z, zp = p['z']
                    c.mant_z = mp; c.exp_z = ep; c.M_z = M; c.K_z = Kd
                    c.norm_w = p['norm_w']
                    mp, ep, M, Kd, yo, yop = p['o']
                    c.mant_o = mp; c.exp_o = ep; c.M_o = M; c.K_o = Kd
                    c.post_ln_w = p['post_ln']
                    mp, ep, M, Kd, yg, ygp = p['mlp_gate']
                    c.mant_gate = mp; c.exp_gate = ep; c.M_gate = M; c.K_gate = Kd
                    mp, ep, M, Kd, yu, yup = p['mlp_up']
                    c.mant_up = mp; c.exp_up = ep; c.M_up = M; c.K_up = Kd
                    mp, ep, M, Kd, yd, ydp = p['mlp_down']
                    c.mant_down = mp; c.exp_down = ep; c.M_down = M; c.K_down = Kd
                    p['c_plan'] = c
                else:
                    c = MlpPlanC()
                    c.post_ln_w = p['post_ln']
                    mp, ep, M, Kd, yg, ygp = p['mlp_gate']
                    c.mant_gate = mp; c.exp_gate = ep; c.M_gate = M; c.K_gate = Kd
                    mp, ep, M, Kd, yu, yup = p['mlp_up']
                    c.mant_up = mp; c.exp_up = ep; c.M_up = M; c.K_up = Kd
                    mp, ep, M, Kd, yd, ydp = p['mlp_down']
                    c.mant_down = mp; c.exp_down = ep; c.M_down = M; c.K_down = Kd
                    p['mlp_c_plan'] = c

        self._has_c_forward_token = bool(self.kernels and hasattr(self.kernels, 'forward_token_c'))
        if self._has_c_forward_token:
            CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
            self.kernels.forward_token_c.argtypes = [
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                CALLBACK_TYPE, ctypes.c_void_p
            ]
            self.kernels.forward_token_c.restype = ctypes.c_int
            self._c_linear_plans = (LinearLayerPlanC * 18)()
            self._c_mlp_plans = (MlpPlanC * 24)()
            self._c_is_full = (ctypes.c_int * 24)()
            linear_idx = 0
            for l in range(self.num_layers):
                p = self._plan[l]
                if p['full']:
                    self._c_is_full[l] = 1
                    self._c_mlp_plans[l] = p['mlp_c_plan']
                else:
                    self._c_is_full[l] = 0
                    self._c_linear_plans[linear_idx] = p['c_plan']
                    linear_idx += 1
            self._c_full_attn_cb = CALLBACK_TYPE(self._on_full_attn_cb)

        # ---- Stage profiling (QWEN_PROFILE=1) -------------------------------
        # The whole per-token loop runs inside forward_token_c, so the only
        # meaningful place to measure is in C.  qwen_prof_* is a no-op unless
        # enabled here.  The first token (warmup) is discarded.
        self._prof_on = os.environ.get("QWEN_PROFILE", "") == "1"
        self._prof_started = False
        self._prof_tokens = 0
        if self._prof_on and hasattr(self.kernels, 'qwen_prof_enable'):
            self.kernels.qwen_prof_enable.argtypes = [ctypes.c_int]
            self.kernels.qwen_prof_reset.argtypes = []
            self.kernels.qwen_prof_get.argtypes = [ctypes.c_int]
            self.kernels.qwen_prof_get.restype = ctypes.c_double
            self.kernels.qwen_prof_count.argtypes = []
            self.kernels.qwen_prof_count.restype = ctypes.c_int
            self.kernels.qwen_prof_name.argtypes = [ctypes.c_int]
            self.kernels.qwen_prof_name.restype = ctypes.c_char_p
            self.kernels.qwen_prof_enable(1)
            atexit.register(self.dump_profile)
        elif self._prof_on:
            print("[PROF] WARNING: kernels have no qwen_prof_* symbols; "
                  "rebuild libqwen_kernels.so from qwen_kernels_opt.c")

    def dump_profile(self):
        if not getattr(self, '_prof_on', False):
            return
        if not hasattr(self.kernels, 'qwen_prof_count'):
            return
        ntok = self._prof_tokens
        if ntok <= 0:
            print("[PROF] no tokens measured")
            return
        QP_TOTAL = 14
        tot = self.kernels.qwen_prof_get(QP_TOTAL)
        per = tot / float(ntok)
        print("[PROF] tokens=%d  accumulated=%.1f ms  per-token=%.2f ms"
              % (ntok, tot, per))
        rows = []
        for i in range(self.kernels.qwen_prof_count()):
            ms = self.kernels.qwen_prof_get(i)
            if ms <= 0.0:
                continue
            rows.append((ms, self.kernels.qwen_prof_name(i).decode('ascii'), i))
        rows.sort(reverse=True)
        print("[PROF] %-22s %10s %8s %14s" % ("stage", "total_ms", "pct", "ms/token"))
        for ms, nm, i in rows:
            print("[PROF] %-22s %10.1f %7.1f%% %14.3f"
                  % (nm, ms, 100.0 * ms / tot if tot else 0.0, ms / float(ntok)))

    def _attn_post(self, pool, layer_idx, pos, p, yqgp, ykp, yvp):
        """rmsnorm_heads+rope+append_kv+attention+gate in one C call.

        Falls back to the numpy glue in _attn_post_numpy() on any problem, so
        this can never be less correct than the code it replaces."""
        return _attn_post(pool, layer_idx, pos, p, self, yqgp, ykp, yvp)

    def _attn_post_numpy(self, pool, layer_idx, pos, p):
        """The numpy implementation of the same glue (fallback / reference)."""
        return _attn_post_numpy(pool, layer_idx, pos, p, self)

    def _on_full_attn_cb(self, l, h_ptr, ctx):
        p = self._plan[l]
        b = self._b
        normed_ptr = b["normed"][1]
        self._k_rmsnorm(normed_ptr, h_ptr, p['in_ln'], 1024, 1e-6)

        pool = self.paged_attn
        if not pool.block_table:
            pool.allocate_slot()
        pos = pool.seq_len
        pool_idx = p['pool_idx']

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

    def _layer_fwd(self, l):
        p = self._plan[l]
        h_ptr = self._h_ptr
        h_arr = self._h_arr

        if self._has_c_forward and not p['full']:
            self.kernels.forward_linear_layer(ctypes.byref(p['c_plan']), h_ptr)
            return

        b = self._b
        kg = self._k_gemv
        kr = self._k_rmsnorm
        normed_ptr = b["normed"][1]

        kr(normed_ptr, h_ptr, p['in_ln'], 1024, 1e-6)

        if not p['full']:
            mp, ep, M, Kd, y, yp = p['qkv']
            kg(yp, normed_ptr, mp, ep, M, Kd)
            mixed_ptr = b["mixed"][1]
            self._k_conv(mixed_ptr, yp, p['conv_s'], p['conv_w'], 6144)
            q_ptr = mixed_ptr
            k_ptr = mixed_ptr + 8192
            v_ptr = mixed_ptr + 16384
            self._k_qk(q_ptr, k_ptr, 16, 128)
            self._k_gates(b["alpha"][1], b["beta"][1], normed_ptr,
                          p['wa'], p['wb'], p['nea'], p['dt'], 16, 1024)
            self._k_gdn(b["attn_out"][1], p['state'],
                        q_ptr, k_ptr, v_ptr,
                        b["alpha"][1], b["beta"][1], 16, 128)
            mp, ep, M, Kd, z, zp = p['z']
            kg(zp, normed_ptr, mp, ep, M, Kd)
            self._k_gated(b["gated_out"][1], b["attn_out"][1], zp,
                          p['norm_w'], 16, 128, 1e-6)
            mp, ep, M, Kd, yo, yop = p['o']
            kg(yop, b["gated_out"][1], mp, ep, M, Kd)
            attn_res = yo
        else:
            pool = self.paged_attn
            if not pool.block_table:
                pool.allocate_slot()
            pos = pool.seq_len
            pool_idx = p['pool_idx']

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

        np.add(h_arr, attn_res, out=h_arr)

        if self._has_c_forward and p['full']:
            self.kernels.forward_mlp(ctypes.byref(p['mlp_c_plan']), h_ptr)
            return

        post_ptr = b["post_norm"][1]
        kr(post_ptr, h_ptr, p['post_ln'], 1024, 1e-6)

        mp, ep, M, Kd, yg, ygp = p['mlp_gate']
        kg(ygp, post_ptr, mp, ep, M, Kd)
        mp, ep, M, Kd, yu, yup = p['mlp_up']
        kg(yup, post_ptr, mp, ep, M, Kd)
        self._k_swiglu(b["intermediate"][1], ygp, yup, 3584)
        mp, ep, M, Kd, yd, ydp = p['mlp_down']
        kg(ydp, b["intermediate"][1], mp, ep, M, Kd)
        np.add(h_arr, yd, out=h_arr)

    def forward_layer(self, l, h):
        # Public entry point kept for bench.py / interactive use.  The hot loop
        # calls _layer_fwd directly against the persistent h buffer.
        if h is self._h_arr:
            self._layer_fwd(l)
            return h
        save_a, save_p = self._h_arr, self._h_ptr
        self._h_arr, self._h_ptr = h, h.ctypes.data
        try:
            self._layer_fwd(l)
        finally:
            self._h_arr, self._h_ptr = save_a, save_p
        return h

    def get_tensor_data(self, name):
        info = self.manifest['tensors'][name]
        offset = info['file_offset']
        size = info['byte_size']
        return self.weights_mm[offset : offset + size], info

    def _tensor_views(self, name):
        # 零拷贝视图：原来 get_tensor_data 里的 weights_mm[a:b] 每次都会把整个张量
        # 拷成 bytes（单次最多 6 MB，一 token 累计约 600 MB 的 memcpy），这是
        # Python 侧 4 s 开销的主因。这里改成直接在 mmap 上开 numpy 视图并缓存。
        cache = getattr(self, '_view_cache', None)
        if cache is None:
            cache = self._view_cache = {}
        v = cache.get(name)
        if v is not None:
            return v
        info = self.manifest['tensors'][name]
        M, K = info['shape']
        raw = np.frombuffer(self.weights_mm, dtype=np.uint8,
                            count=info['byte_size'], offset=info['file_offset'])
        mant = raw[:M * K].view(np.int8)
        exps = raw[M * K: M * K + (M * K) // 64]
        v = (mant, exps, M, K)
        cache[name] = v
        return v

    def _y_buf(self, name, M):
        # 输出缓冲按张量名复用：C 内核会写满 y[0..M-1]，不需要 zeros；
        # 同一层里每个 gemv 用的张量名互不相同，且结果在被下一次同名调用覆盖前
        # 就已经消费完（qkv 会先切出 q/k/v 副本），所以复用是安全的。
        cache = getattr(self, '_y_cache', None)
        if cache is None:
            cache = self._y_cache = {}
        b = cache.get(name)
        if b is None or b.shape[0] != M:
            b = np.empty(M, dtype=np.float32)
            cache[name] = b
        return b

    def gemv_bfp8(self, x, tensor_name):
        mantissas, exps, M, K = self._tensor_views(tensor_name)
        y = self._y_buf(tensor_name, M)

        if self.kernels and hasattr(self.kernels, 'bfp8_gemv'):
            self.kernels.bfp8_gemv(
                y.ctypes.data,
                x.ctypes.data,
                mantissas.ctypes.data,
                exps.ctypes.data,
                M, K
            )
        else:
            num_blocks = K // 64
            scales = (2.0 ** (exps.astype(np.float32) - 134.0)).reshape(M, num_blocks)
            m_blocks = mantissas.reshape(M, num_blocks, 64).astype(np.float32)
            x_blocks = x.reshape(num_blocks, 64)
            dots = np.sum(m_blocks * x_blocks, axis=-1)
            y = np.sum(dots * scales, axis=-1)
        return y

    def rmsnorm(self, x, tensor_name, eps=1e-6):
        raw, info = self.get_tensor_data(tensor_name)
        w = np.frombuffer(raw, dtype=np.float32)
        y = np.zeros_like(x)
        if self.kernels and hasattr(self.kernels, 'rmsnorm'):
            self.kernels.rmsnorm(
                y.ctypes.data,
                x.ctypes.data,
                w.ctypes.data,
                len(x),
                eps
            )
        else:
            var = np.mean(x ** 2) + eps
            y = (x / np.sqrt(var)) * (1.0 + w)
        return y

    def forward_token(self, token_id):
        if getattr(self, '_has_c_forward_token', False):
            self.paged_attn.allocate_slot()
            best_id = self.kernels.forward_token_c(
                int(token_id),
                self._h_ptr,
                ctypes.byref(self._c_linear_plans),
                ctypes.byref(self._c_mlp_plans),
                ctypes.byref(self._c_is_full),
                self._final_w_ptr,
                self._embed_ptr,
                self._c_full_attn_cb,
                None
            )
            self.paged_attn.step_done()
            if getattr(self, '_prof_on', False):
                if not self._prof_started:
                    # discard the warmup token, then start accumulating
                    self.kernels.qwen_prof_reset()
                    self._prof_started = True
                else:
                    self._prof_tokens += 1
            return best_id

        # Fallback path:
        # 1. Allocate physical block slot in PagedAttention pool
        self.paged_attn.allocate_slot()

        # 2. Embedding lookup, bf16 -> f32 straight into the persistent h buffer
        t_vec_u16 = self.embed_raw[token_id % 65536]
        np.copyto(self._h_arr,
                  (t_vec_u16.astype(np.uint32) << 16).view(np.float32))

        # 3. Iterate through 24 layers.
        for l in range(self.num_layers):
            self._layer_fwd(l)

        # 4. Final RMSNorm into the persistent buffer
        self._k_rmsnorm(self._final_ptr, self._h_ptr,
                        self._final_w_ptr, 1024, 1e-6)

        # 5. LM_Head Argmax over 65,536 embeddings (via NEON SIMD)
        best_id = self._k_lm(self._final_ptr, self._embed_ptr, 65536, 1024)

        # 6. Mark token step completed
        self.paged_attn.step_done()
        return best_id

    def generate(self, prompt, max_new_tokens=16):
        input_ids = self.tok.encode(prompt)
        if not input_ids:
            return

        # Prefill prompt tokens to update state
        for tid in input_ids[:-1]:
            self.forward_token(tid)

        current_token = input_ids[-1]
        for step in range(max_new_tokens):
            next_token = self.forward_token(current_token)
            piece = self.tok.decode(next_token)
            sys.stdout.write(piece)
            sys.stdout.flush()
            current_token = next_token
        print()


def main():
    print("=" * 70)
    print("  Qwen3.5-0.8B Real 24-Layer Neural Network Inference Engine")
    print("  Physical Silicon: Zynq-7020 (ARM Cortex-A9 NEON + FPGA Core A)")
    print("  Memory Manager: vLLM-style PagedAttention Pool (2048 Context)")
    print("  Mode: 100% Real Floating-Point / BFP8 Calculation (Zero Mocks)")
    print("=" * 70)

    engine = QwenRealEngine()

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        print("\n--- Running Real 24-Layer Forward Pass on: {!r} ---".format(prompt))
        print("Assistant: ", end="", flush=True)
        engine.generate(prompt, max_new_tokens=16)
    else:
        print("\nEntering Interactive Real-Model Generation (Type /exit to quit, /clear to reset context):\n")
        while True:
            try:
                sys.stdout.write("User: ")
                sys.stdout.flush()
                prompt = sys.stdin.readline()
                if not prompt:
                    time.sleep(0.1)
                    continue
                prompt = prompt.strip()
            except KeyboardInterrupt:
                print("\n[Exiting chat]")
                break

            if not prompt:
                continue

            if prompt == "/exit":
                print("Bye!")
                break
            if prompt == "/clear":
                engine.reset_state()
                print("[FPGA & PagedAttention State Cleared!]\n")
                continue

            print("Assistant: ", end="", flush=True)
            engine.generate(prompt, max_new_tokens=16)


if __name__ == "__main__":
    main()
