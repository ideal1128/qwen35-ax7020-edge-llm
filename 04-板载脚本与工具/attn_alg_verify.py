#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attn_alg_verify.py -- validate the ALGORITHM and MEMORY LAYOUT of qwen_attn.c
against the original numpy PagedAttentionPool.compute_attention().

Runs on the host (no board, no compiler needed).  It ports full_attn_ref()
from qwen_attn.c to Python line for line, so if the two agree on random data
then the C code indexes k_pool/v_pool/block_table correctly.  The NEON path
is verified separately on the board by qwen_attn.c's -DATTN_SELFTEST.

ASCII only.
"""
import sys
import numpy as np

# ---- real shapes (Qwen3.5-0.8B full-attention layers) ---------------------
N_LAYERS = 6
N_KV_HEADS = 2
N_Q_HEADS = 8
HEAD_DIM = 256
BLOCK_SIZE = 16
SCALE = 0.0625   # 1/sqrt(256)


class Pool(object):
    """Mirror of PagedAttentionPool, numpy reference path."""

    def __init__(self, num_blocks=8):
        self.num_layers = N_LAYERS
        self.num_kv_heads = N_KV_HEADS
        self.head_dim = HEAD_DIM
        self.block_size = BLOCK_SIZE
        self.num_blocks = num_blocks
        self.k_pool = np.zeros((num_blocks, N_LAYERS, N_KV_HEADS,
                                BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
        self.v_pool = np.zeros((num_blocks, N_LAYERS, N_KV_HEADS,
                                BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
        self.free_blocks = list(range(num_blocks - 1, -1, -1))
        self.block_table = []
        self.seq_len = 0

    def allocate_slot(self):
        offset = self.seq_len % self.block_size
        if offset == 0:
            self.block_table.append(self.free_blocks.pop())
        return self.block_table[-1], offset

    def append_kv(self, layer_idx, k, v):
        b = self.block_table[-1]
        o = self.seq_len % self.block_size
        self.k_pool[b, layer_idx, :, o, :] = k
        self.v_pool[b, layer_idx, :, o, :] = v

    def step_done(self):
        self.seq_len += 1

    # ---- the ORIGINAL numpy implementation, verbatim semantics ----
    def compute_attention_numpy(self, layer_idx, q):
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
        k_exp = np.repeat(k_all, 4, axis=0)
        v_exp = np.repeat(v_all, 4, axis=0)
        logits = np.einsum('hd,htd->ht', q, k_exp) * SCALE
        max_l = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        out = np.einsum('ht,htd->hd', probs, v_exp).reshape(-1)
        return out


def full_attn_ref_py(out, q, k_pool, v_pool, block_table,
                     n_layers, n_kv_heads, n_q_heads, block_size,
                     head_dim, layer, total_tokens, scale):
    """Line-for-line port of full_attn_ref() in qwen_attn.c."""
    ratio = n_q_heads // n_kv_heads
    stride_l = n_kv_heads * block_size * head_dim
    stride_kh = block_size * head_dim
    stride_b = n_layers * stride_l

    # flatten to 1-D so we index exactly like C
    kf = k_pool.reshape(-1)
    vf = v_pool.reshape(-1)
    qf = q.reshape(-1)

    logits = np.zeros((n_q_heads, total_tokens), dtype=np.float64)
    probs = np.zeros((n_q_heads, total_tokens), dtype=np.float64)

    for h in range(n_q_heads):
        kh = h // ratio
        for t in range(total_tokens):
            b = int(block_table[t // block_size])
            o = t % block_size
            base_k = b * stride_b + layer * stride_l + kh * stride_kh + o * head_dim
            base_q = h * head_dim
            acc = 0.0
            for d in range(head_dim):
                acc += float(qf[base_q + d]) * float(kf[base_k + d])
            logits[h, t] = acc * scale
        mx = logits[h, 0]
        for t in range(1, total_tokens):
            if logits[h, t] > mx:
                mx = logits[h, t]
        s = 0.0
        for t in range(total_tokens):
            e = np.exp(logits[h, t] - mx)
            probs[h, t] = e
            s += e
        if s > 0.0:
            for t in range(total_tokens):
                probs[h, t] /= s
        for d in range(head_dim):
            out[h * head_dim + d] = 0.0
        for t in range(total_tokens):
            b = int(block_table[t // block_size])
            o = t % block_size
            base_v = b * stride_b + layer * stride_l + kh * stride_kh + o * head_dim
            p = probs[h, t]
            for d in range(head_dim):
                out[h * head_dim + d] += p * float(vf[base_v + d])
    return out


def run_case(n_tokens, layer, seed):
    rng = np.random.RandomState(seed)
    pool = Pool(num_blocks=8)

    # fill the pool in logical order using the same block-allocation logic
    for step in range(n_tokens):
        pool.allocate_slot()
        k = rng.randn(N_KV_HEADS, HEAD_DIM).astype(np.float32) * 0.5
        v = rng.randn(N_KV_HEADS, HEAD_DIM).astype(np.float32) * 0.5
        pool.append_kv(layer, k, v)
        pool.step_done()
    # seq_len was incremented n_tokens times; compute_attention uses seq_len+1,
    # so wind it back one to make total_tokens == n_tokens
    pool.seq_len = n_tokens - 1

    q = rng.randn(N_Q_HEADS, HEAD_DIM).astype(np.float32)

    ref = pool.compute_attention_numpy(layer, q)

    mine = np.zeros(N_Q_HEADS * HEAD_DIM, dtype=np.float64)
    full_attn_ref_py(mine, q, pool.k_pool, pool.v_pool,
                     pool.block_table, N_LAYERS, N_KV_HEADS, N_Q_HEADS,
                     BLOCK_SIZE, HEAD_DIM, layer, n_tokens, SCALE)

    diff = np.abs(mine - ref.astype(np.float64))
    max_abs = float(diff.max())
    denom = np.abs(ref.astype(np.float64)) + 1e-12
    max_rel = float((diff / denom).max())
    ok = max_abs < 1e-4
    print("  tokens=%4d layer=%d blocks=%2d  max_abs=%.3e max_rel=%.3e  %s"
          % (n_tokens, layer, len(pool.block_table), max_abs, max_rel,
             "PASS" if ok else "FAIL"))
    return ok


def main():
    print("=== qwen_attn.c algorithm verification (Python port vs numpy) ===")
    print("shapes: q_heads=%d kv_heads=%d head_dim=%d block=%d scale=%g"
          % (N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, SCALE))
    ok = True
    # single block, partial
    ok &= run_case(7, 3, 1)
    # exactly one full block
    ok &= run_case(16, 0, 2)
    # crosses a block boundary
    ok &= run_case(33, 1, 3)
    # several blocks
    ok &= run_case(80, 2, 4)
    # last full-attention layer index
    ok &= run_case(47, 5, 5)

    print("=== %s ===" % ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
