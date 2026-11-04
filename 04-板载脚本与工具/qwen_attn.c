/* ==========================================================================
 * qwen_attn.c -- full-attention kernel for Qwen3.5-0.8B on Cortex-A9 + NEON
 *
 * WHY THIS EXISTS
 * ---------------
 * The 6 full-attention layers were computed in numpy by
 * PagedAttentionPool.compute_attention().  That path:
 *   - builds a [8, T, 256] float32 copy of K and V via np.repeat (4.5 MB at
 *     T=544) on EVERY layer of EVERY token,
 *   - runs two einsums and a numpy softmax.
 * Measured cost: 66.6 ms/token at context 20, and roughly 460 ms/token at
 * context 544 -- which is what makes long conversations collapse from
 * 817 ms/tok to 1214 ms/tok.
 *
 * This kernel reads K and V exactly once per layer (no broadcast copy: the
 * GQA 4:1 mapping is done by index, not by materialising k_exp/v_exp) and
 * keeps the accumulators in NEON registers.  Target: ~15 ms at T=544.
 *
 * SHAPES (Qwen3.5-0.8B)
 *   n_q_heads  = 8      n_kv_heads = 2      ratio = 4  (GQA)
 *   head_dim   = 256    block_size = 16     n_layers = 6
 *   scale      = 1/sqrt(256) = 0.0625
 *
 * MEMORY LAYOUT
 *   k_pool / v_pool are numpy arrays of shape
 *       (num_blocks, n_layers, n_kv_heads, block_size, head_dim)
 *   C-contiguous, so the element [b][l][kh][o][d] lives at
 *       ((((b * n_layers + l) * n_kv_heads + kh) * block_size + o) * head_dim + d)
 *   Logical token t maps to physical block block_table[t / block_size] at
 *   offset t % block_size.
 *
 *   q   : [n_q_heads * head_dim]   float32, contiguous
 *   out : [n_q_heads * head_dim]   float32, contiguous
 *
 * BUILD (on the board)
 *   gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard \
 *       -DUSE_NEON -shared -fPIC -o libqwen_attn.so qwen_attn.c -lm -fopenmp
 *
 * SELF TEST
 *   Build with -DATTN_SELFTEST to get a main() that compares the NEON path
 *   against a scalar reference on random data.
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#if defined(USE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

#define ATTN_MAX_TOKENS 2048
#define ATTN_MAX_QHEADS 16
#define ATTN_MAX_KVHEADS 8
#define ATTN_MAX_HD 256
#define ATTN_ROT_DIM 64      /* partial rotary: first 64 of head_dim rotate */

/* logits and probabilities, [head][token]; 8*2048*4 = 64 KB */
static float g_logits[ATTN_MAX_QHEADS][ATTN_MAX_TOKENS];
static float g_probs[ATTN_MAX_QHEADS][ATTN_MAX_TOKENS];

/* Per-token row pointers, resolved once per (layer, kv head).
 *
 * The naive inner loop recomputed `block_table[t/block_size]`, a divide and a
 * modulo for every token, and phase 3 walked that lookup 64 times (once per
 * lane block) -- about 35k redundant lookups per layer.  Building the pointer
 * table once and then streaming it removes all of that.  2*2048*4 = 16 KB. */
static const float *g_kptr[ATTN_MAX_TOKENS];
static const float *g_vptr[ATTN_MAX_TOKENS];

/* Fill ptrs[t] with the address of the row for logical token t. */
static void build_ptrs(const float **ptrs, const float *pool,
                       const int32_t *block_table, int n_blocks_used,
                       size_t stride_b, size_t stride_l, size_t stride_kh,
                       int kh, int layer, int block_size, int head_dim,
                       int total_tokens)
{
    const float *base_l = pool + (size_t)layer * stride_l
                               + (size_t)kh * stride_kh;
    int t = 0;
    int bi;
    for (bi = 0; bi < n_blocks_used && t < total_tokens; bi++) {
        const float *base = base_l + (size_t)block_table[bi] * stride_b;
        int ntok = total_tokens - t;
        int o;
        if (ntok > block_size) ntok = block_size;
        for (o = 0; o < ntok; o++)
            ptrs[t++] = base + (size_t)o * head_dim;
    }
    /* defensive: if the block table was short, repeat the last row so we never
     * read out of bounds (this should not happen) */
    while (t < total_tokens) ptrs[t++] = ptrs[t - 2 < 0 ? 0 : t - 2];
}

/* --------------------------------------------------------------------------
 * scalar reference -- used by the self test and as the non-NEON fallback
 * -------------------------------------------------------------------------- */
static void full_attn_ref(float *out, const float *q,
                          const float *k_pool, const float *v_pool,
                          const int32_t *block_table,
                          int n_layers, int n_kv_heads, int n_q_heads,
                          int block_size, int head_dim,
                          int layer, int total_tokens, float scale)
{
    int h, kh, t, d, o, b;
    const int ratio = n_q_heads / n_kv_heads;
    const size_t stride_l = (size_t)n_kv_heads * block_size * head_dim;
    const size_t stride_kh = (size_t)block_size * head_dim;
    const size_t stride_b = (size_t)n_layers * stride_l;

    for (h = 0; h < n_q_heads; h++) {
        kh = h / ratio;
        /* logits */
        for (t = 0; t < total_tokens; t++) {
            b = block_table[t / block_size];
            o = t % block_size;
            const float *kp = k_pool + (size_t)b * stride_b
                                     + (size_t)layer * stride_l
                                     + (size_t)kh * stride_kh
                                     + (size_t)o * head_dim;
            const float *qp = q + (size_t)h * head_dim;
            float acc = 0.0f;
            for (d = 0; d < head_dim; d++) acc += qp[d] * kp[d];
            g_logits[h][t] = acc * scale;
        }
        /* softmax */
        float mx = g_logits[h][0];
        for (t = 1; t < total_tokens; t++)
            if (g_logits[h][t] > mx) mx = g_logits[h][t];
        float sum = 0.0f;
        for (t = 0; t < total_tokens; t++) {
            float e = expf(g_logits[h][t] - mx);
            g_probs[h][t] = e;
            sum += e;
        }
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (t = 0; t < total_tokens; t++) g_probs[h][t] *= inv;
        }
        /* output */
        for (d = 0; d < head_dim; d++) out[(size_t)h * head_dim + d] = 0.0f;
        for (t = 0; t < total_tokens; t++) {
            b = block_table[t / block_size];
            o = t % block_size;
            const float *vp = v_pool + (size_t)b * stride_b
                                     + (size_t)layer * stride_l
                                     + (size_t)kh * stride_kh
                                     + (size_t)o * head_dim;
            float p = g_probs[h][t];
            for (d = 0; d < head_dim; d++)
                out[(size_t)h * head_dim + d] += p * vp[d];
        }
    }
}

/* --------------------------------------------------------------------------
 * NEON path
 * -------------------------------------------------------------------------- */
#if HAVE_NEON

static inline float hsum4(float32x4_t v)
{
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}

/* ratio == 4, fully unrolled.
 *
 * Why a separate function: with `ratio` as a runtime argument the generic
 * version leaves a compare-and-branch in the innermost body, so gcc cannot
 * unroll it and the A9 pays a mispredict-ish stall on every 4 elements.
 * Every full-attention layer of this model is 8 q heads over 2 kv heads, i.e.
 * ratio is always 4, so this path is the one that actually runs. */
static inline void dot_group4(float *dst, const float *q, const float *kp,
                              int head_dim, float scale)
{
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    const float *q1 = q + head_dim;
    const float *q2 = q + 2 * head_dim;
    const float *q3 = q + 3 * head_dim;
    int d = 0;
    for (; d + 4 <= head_dim; d += 4) {
        float32x4_t kv = vld1q_f32(kp + d);
        a0 = vmlaq_f32(a0, vld1q_f32(q  + d), kv);
        a1 = vmlaq_f32(a1, vld1q_f32(q1 + d), kv);
        a2 = vmlaq_f32(a2, vld1q_f32(q2 + d), kv);
        a3 = vmlaq_f32(a3, vld1q_f32(q3 + d), kv);
    }
    {
        float s0 = hsum4(a0), s1 = hsum4(a1), s2 = hsum4(a2), s3 = hsum4(a3);
        for (; d < head_dim; d++) {
            float kv = kp[d];
            s0 += q[d] * kv;  s1 += q1[d] * kv;
            s2 += q2[d] * kv; s3 += q3[d] * kv;
        }
        dst[0] = s0 * scale; dst[1] = s1 * scale;
        dst[2] = s2 * scale; dst[3] = s3 * scale;
    }
}

/* logits for the `ratio` query heads that share one kv head, for one token.
 * Reads the K row once and multiplies it against all `ratio` q rows. */
static inline void dot_group(float *dst, const float *q, const float *kp,
                             int head_dim, int ratio, float scale)
{
    float32x4_t acc[4];
    int i;

    if (ratio == 4) { dot_group4(dst, q, kp, head_dim, scale); return; }

    for (i = 0; i < ratio && i < 4; i++) acc[i] = vdupq_n_f32(0.0f);

    int d = 0;
    for (; d + 4 <= head_dim; d += 4) {
        float32x4_t kv = vld1q_f32(kp + d);
        for (i = 0; i < ratio && i < 4; i++) {
            float32x4_t qv = vld1q_f32(q + (size_t)i * head_dim + d);
            acc[i] = vmlaq_f32(acc[i], qv, kv);
        }
    }
    for (i = 0; i < ratio && i < 4; i++) {
        float s = hsum4(acc[i]);
        /* tail */
        for (; d < head_dim; d++)
            s += q[(size_t)i * head_dim + d] * kp[d];
        dst[i] = s * scale;
    }
}

/* ratio == 4, fully unrolled.  The four probabilities are streamed from four
 * separate [2048] arrays; keeping the loop free of ratio tests lets gcc keep
 * all four accumulators in NEON registers across the whole t loop. */
static inline void axpy_group4(float *out, const float **vptrs,
                               int total_tokens, int head_dim,
                               const float *probs)
{
    const float *p0 = probs;
    const float *p1 = probs + ATTN_MAX_TOKENS;
    const float *p2 = probs + 2 * ATTN_MAX_TOKENS;
    const float *p3 = probs + 3 * ATTN_MAX_TOKENS;
    int d, t;

    for (d = 0; d + 4 <= head_dim; d += 4) {
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        for (t = 0; t < total_tokens; t++) {
            float32x4_t vv = vld1q_f32(vptrs[t] + d);
            a0 = vmlaq_n_f32(a0, vv, p0[t]);
            a1 = vmlaq_n_f32(a1, vv, p1[t]);
            a2 = vmlaq_n_f32(a2, vv, p2[t]);
            a3 = vmlaq_n_f32(a3, vv, p3[t]);
        }
        vst1q_f32(out + d, a0);
        vst1q_f32(out + head_dim + d, a1);
        vst1q_f32(out + 2 * head_dim + d, a2);
        vst1q_f32(out + 3 * head_dim + d, a3);
    }
    for (; d < head_dim; d++) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (t = 0; t < total_tokens; t++) {
            float vv = vptrs[t][d];
            s0 += p0[t] * vv; s1 += p1[t] * vv;
            s2 += p2[t] * vv; s3 += p3[t] * vv;
        }
        out[d] = s0;
        out[head_dim + d] = s1;
        out[2 * head_dim + d] = s2;
        out[3 * head_dim + d] = s3;
    }
}

/* out[ratio heads][head_dim] += sum_t probs[h][t] * v[t]
 * Accumulates head_dim/4 lanes at a time so the accumulators stay in
 * registers; V is read once per (kv head, lane block).  Row addresses come
 * from a prebuilt pointer table, so there is no block lookup in the hot loop. */
static inline void axpy_group(float *out, const float **vptrs,
                              int total_tokens, int head_dim,
                              const float *probs, int ratio)
{
    int d, t, i;

    if (ratio == 4) {
        axpy_group4(out, vptrs, total_tokens, head_dim, probs);
        return;
    }

    for (d = 0; d + 4 <= head_dim; d += 4) {
        float32x4_t acc[4];
        for (i = 0; i < ratio && i < 4; i++) acc[i] = vdupq_n_f32(0.0f);

        const float *p0 = probs;
        const float *p1 = probs + ATTN_MAX_TOKENS;
        const float *p2 = probs + 2 * ATTN_MAX_TOKENS;
        const float *p3 = probs + 3 * ATTN_MAX_TOKENS;

        for (t = 0; t < total_tokens; t++) {
            float32x4_t vv = vld1q_f32(vptrs[t] + d);
            acc[0] = vmlaq_n_f32(acc[0], vv, p0[t]);
            if (ratio > 1) acc[1] = vmlaq_n_f32(acc[1], vv, p1[t]);
            if (ratio > 2) acc[2] = vmlaq_n_f32(acc[2], vv, p2[t]);
            if (ratio > 3) acc[3] = vmlaq_n_f32(acc[3], vv, p3[t]);
        }
        for (i = 0; i < ratio && i < 4; i++)
            vst1q_f32(out + (size_t)i * head_dim + d, acc[i]);
    }
    /* head_dim tail (not a multiple of 4) */
    for (i = 0; i < ratio && i < 4; i++) {
        int dd;
        for (dd = d; dd < head_dim; dd++) {
            float s = 0.0f;
            for (t = 0; t < total_tokens; t++)
                s += probs[(size_t)i * ATTN_MAX_TOKENS + t] * vptrs[t][dd];
            out[(size_t)i * head_dim + dd] = s;
        }
    }
}

#endif /* HAVE_NEON */

/* --------------------------------------------------------------------------
 * attention core
 *
 * This is the body of the original public entry point; it is factored out so
 * attn_post_f32() can call it without going through the ctypes boundary.
 * -------------------------------------------------------------------------- */
static void full_attn_impl(float *out, const float *q,
                           const float *k_pool, const float *v_pool,
                           const int32_t *block_table,
                           int n_layers, int n_kv_heads, int n_q_heads,
                           int block_size, int head_dim,
                           int layer, int total_tokens, float scale)
{
    const int ratio = n_q_heads / n_kv_heads;
    const size_t stride_l = (size_t)n_kv_heads * block_size * head_dim;
    const size_t stride_kh = (size_t)block_size * head_dim;
    const size_t stride_b = (size_t)n_layers * stride_l;

    if (total_tokens <= 0) return;
    if (total_tokens > ATTN_MAX_TOKENS) total_tokens = ATTN_MAX_TOKENS;
    if (n_q_heads > ATTN_MAX_QHEADS) return;
    if (ratio > 4) {   /* more than 4 q heads per kv head: use the scalar path */
        full_attn_ref(out, q, k_pool, v_pool, block_table,
                      n_layers, n_kv_heads, n_q_heads,
                      block_size, head_dim, layer, total_tokens, scale);
        return;
    }

#if HAVE_NEON
    int kh, h, t;
    /* how many entries of block_table the logical token range actually spans */
    const int n_blocks = (total_tokens + block_size - 1) / block_size;

    /* ---- phase 1: logits, one kv head at a time ---- */
    for (kh = 0; kh < n_kv_heads; kh++) {
        build_ptrs(g_kptr, k_pool, block_table, n_blocks,
                   stride_b, stride_l, stride_kh, kh, layer,
                   block_size, head_dim, total_tokens);
        const float *qbase = q + (size_t)(kh * ratio) * head_dim;
        for (t = 0; t < total_tokens; t++) {
            float tmp[4];
            dot_group(tmp, qbase, g_kptr[t], head_dim, ratio, scale);
            for (h = 0; h < ratio; h++)
                g_logits[kh * ratio + h][t] = tmp[h];
        }
    }

    /* ---- phase 2: softmax per query head ---- */
    for (h = 0; h < n_q_heads; h++) {
        float *lg = g_logits[h];
        float *pr = g_probs[h];
        float mx = lg[0];
        for (t = 1; t < total_tokens; t++) if (lg[t] > mx) mx = lg[t];
        float sum = 0.0f;
        for (t = 0; t < total_tokens; t++) {
            float e = expf(lg[t] - mx);
            pr[t] = e;
            sum += e;
        }
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (t = 0; t < total_tokens; t++) pr[t] *= inv;
        }
    }

    /* ---- phase 3: weighted sum of V ---- */
    for (kh = 0; kh < n_kv_heads; kh++) {
        build_ptrs(g_vptr, v_pool, block_table, n_blocks,
                   stride_b, stride_l, stride_kh, kh, layer,
                   block_size, head_dim, total_tokens);
        axpy_group(out + (size_t)(kh * ratio) * head_dim, g_vptr,
                   total_tokens, head_dim, &g_probs[kh * ratio][0], ratio);
    }
#else
    full_attn_ref(out, q, k_pool, v_pool, block_table,
                  n_layers, n_kv_heads, n_q_heads,
                  block_size, head_dim, layer, total_tokens, scale);
#endif
}

/* public entry point (unchanged ABI) */
void full_attn_f32(float *out, const float *q,
                   const float *k_pool, const float *v_pool,
                   const int32_t *block_table,
                   int n_layers, int n_kv_heads, int n_q_heads,
                   int block_size, int head_dim,
                   int layer, int total_tokens, float scale)
{
    full_attn_impl(out, q, k_pool, v_pool, block_table,
                   n_layers, n_kv_heads, n_q_heads,
                   block_size, head_dim, layer, total_tokens, scale);
}

/* ==========================================================================
 * attn_post_f32 -- the numpy glue around full attention, done in C
 *
 * WHY
 * ---
 * The full-attention callback in qwen_real_engine.py used to run, per layer
 * and per token, in numpy:
 *
 *     q_and_gate = yqg.reshape(8, 2, 256)   # q gemv output: [head][0=q 1=gate][dim]
 *     q_raw = q_and_gate[:, 0, :]           # strided view
 *     gate  = q_and_gate[:, 1, :].reshape(2048)
 *     q_normed = rmsnorm_heads(q_raw, q_norm)
 *     k_normed = rmsnorm_heads(k_raw, k_norm)
 *     q_rot = apply_rope(q_normed, pos, cos_table, sin_table)
 *     k_rot = apply_rope(k_normed, pos, cos_table, sin_table)
 *     pool.append_kv(pool_idx, k_rot, v_raw)
 *     a_out = full_attn_f32(pool, pool_idx, q_rot)
 *     gated_attn = a_out * (1.0 / (1.0 + np.exp(-gate)))
 *
 * The four gemvs and the attention kernel are already C.  Everything else is
 * numpy, and on this board numpy 1.8.2 on a Cortex-A9 spends ~13-40 cycles per
 * element plus ~100 us of fixed overhead per ufunc call, so these 8x256 arrays
 * cost 17.8 ms/token (measured over 27 tokens: rmsnorm_heads 7.2 ms,
 * sigmoid gate 4.6 ms, rope 3.6 ms, q/gate slice 0.9 ms, append_kv 0.6 ms).
 *
 * attn_post_f32() replaces all of it with one call that writes the 2048 floats
 * that go straight into the o_proj gemv.
 *
 * FLOAT SEMANTICS -- must be bit-identical to numpy, not merely close
 * -------------------------------------------------------------------
 * A 1-ulp difference in the rmsnorm scale can flip a greedy argmax and change
 * the entire answer, so every operation reproduces numpy 1.8.2's order:
 *   - x**2          -> x*x   (numpy's fast_scalar_power turns **2 into a mul)
 *   - mean(x*x)     -> float32 accumulation in numpy's order, then / 256
 *   - var + eps     -> float32 add with the float32 value of 1e-6
 *   - sqrt/div/mul  -> sqrtf, float32 divide, float32 multiply (each rounded)
 *   - rope          -> x1*cos - x2*sin, two roundings; the A9 has no FMA so
 *                      gcc cannot fuse them even at -O3
 *   - sigmoid       -> expf from the same glibc numpy 1.8.2 calls
 * The one thing that cannot be decided from here is numpy's summation order
 * for mean(x*x): 1.8.2 accumulates sequentially, >= 1.9 uses pairwise
 * summation.  It is therefore selectable at run time via attn_set_sum_mode()
 * and the Python side calibrates it against the board's numpy at start-up;
 * if no mode matches bit-for-bit the engine keeps using its numpy path.
 * ========================================================================== */

/* bit 0: pairwise (numpy >= 1.9) instead of sequential (numpy <= 1.8)
 * bit 1: compute x**2 with powf instead of x*x */
static int g_sum_mode = 0;

void attn_set_sum_mode(int mode) { g_sum_mode = mode; }
int attn_get_sum_mode(void) { return g_sum_mode; }

/* numpy >= 1.9 pairwise summation, transcribed from
 * numpy/core/src/umath/loops.c.src (@TYPE@_pairwise_sum):
 *   n < 8     : sequential
 *   n <= 128  : 8 accumulators combined as ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))
 *   otherwise : split into two halves, each rounded down to a multiple of 8
 * head_dim = 256 is the only size used here, i.e. two 128-element blocks. */
static float pairwise_sum(const float *a, int n)
{
    if (n < 8) {
        float res = a[0];
        int i;
        for (i = 1; i < n; i++) res += a[i];
        return res;
    } else if (n <= 128) {
        float r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3];
        float r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
        float res;
        int i = 8;
        int end = n - (n % 8);
        for (; i < end; i += 8) {
            r0 += a[i];     r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3];
            r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7];
        }
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < n; i++) res += a[i];
        return res;
    } else {
        int n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
    }
}

/* sum(x[d] * x[d]) with numpy's accumulation order */
static float sum_squares(const float *x, int n)
{
    float s = 0.0f;
    int d;
    if (g_sum_mode & 2) {
        float sq[ATTN_MAX_HD];
        for (d = 0; d < n; d++) sq[d] = powf(x[d], 2.0f);
        if (g_sum_mode & 1) return pairwise_sum(sq, n);
        for (d = 0; d < n; d++) s += sq[d];
        return s;
    }
    if (g_sum_mode & 1) {
        float sq[ATTN_MAX_HD];
        for (d = 0; d < n; d++) sq[d] = x[d] * x[d];
        return pairwise_sum(sq, n);
    }
    for (d = 0; d < n; d++) s += x[d] * x[d];
    return s;
}

/* rmsnorm_heads(x[nh][hd], w[hd], eps) = (x / sqrt(mean(x*x) + eps)) * (1 + w)
 *
 * `stride` is the distance between consecutive heads in x, so the q gemv
 * output (which interleaves q and gate) can be read in place with stride
 * 2*head_dim.  `out` is always packed at [nh][hd]. */
void attn_rmsnorm_heads_f32(float *out, const float *x, int nh, int hd,
                            int stride, const float *w, float eps)
{
    int h, d;
    if (hd <= 0 || hd > ATTN_MAX_HD) return;
    for (h = 0; h < nh; h++) {
        const float *xp = x + (size_t)h * stride;
        float *op = out + (size_t)h * hd;
        float var = sum_squares(xp, hd) / (float)hd;
        float den = sqrtf(var + eps);
        for (d = 0; d < hd; d++) op[d] = (xp[d] / den) * (1.0f + w[d]);
    }
}

/* apply_rope(x[nh][hd], cos_row[32], sin_row[32]), in place.
 * Only the first ATTN_ROT_DIM dims rotate; dims ATTN_ROT_DIM..hd-1 pass
 * through untouched. */
void attn_rope_f32(float *x, int nh, int hd,
                   const float *cos_row, const float *sin_row)
{
    const int half = ATTN_ROT_DIM / 2;
    int h, i;
    if (hd < ATTN_ROT_DIM) return;
    for (h = 0; h < nh; h++) {
        float *xp = x + (size_t)h * hd;
        for (i = 0; i < half; i++) {
            float x1 = xp[i];
            float x2 = xp[i + half];
            float c = cos_row[i];
            float s = sin_row[i];
            xp[i] = x1 * c - x2 * s;
            xp[i + half] = x2 * c + x1 * s;
        }
    }
}

/* out[nh][hd] = attn[nh][hd] * sigmoid(gate), where the gate half of the q
 * gemv output is at [head][1][dim]. */
void attn_gate_f32(float *out, const float *attn, const float *q_raw,
                   int nh, int hd)
{
    int h, d;
    for (h = 0; h < nh; h++) {
        const float *g = q_raw + ((size_t)h * 2 + 1) * hd;
        const float *a = attn + (size_t)h * hd;
        float *o = out + (size_t)h * hd;
        for (d = 0; d < hd; d++)
            o[d] = a[d] * (1.0f / (1.0f + expf(-g[d])));
    }
}

/* Scratch space: 3 * 16 * 256 floats = 48 KB.  Static so the hot callback
 * never asks the stack for it. */
static float g_qn[ATTN_MAX_QHEADS * ATTN_MAX_HD];
static float g_kn[ATTN_MAX_KVHEADS * ATTN_MAX_HD];
static float g_attn[ATTN_MAX_QHEADS * ATTN_MAX_HD];

/* One call = rmsnorm_heads(q), rmsnorm_heads(k), rope(q), rope(k),
 * pool.append_kv, attention, sigmoid gate.
 *
 *   q_raw : [nh][2][hd]  q gemv output, [head][0]=q [head][1]=gate
 *   k_raw : [nkv][hd]    k gemv output
 *   v_raw : [nkv][hd]    v gemv output
 *   cos_row/sin_row : [32] the precomputed rope row for this position
 *   total_tokens : pool.seq_len + 1 (the token being generated is included)
 *
 * Returns 0 on success, non-zero if the caller asked for a shape this kernel
 * does not handle; Python then runs its numpy implementation instead. */
int attn_post_f32(float *gated_out,
                  const float *q_raw, const float *k_raw, const float *v_raw,
                  const float *q_norm_w, const float *k_norm_w,
                  const float *cos_row, const float *sin_row,
                  float eps, int nh, int nkv, int hd,
                  float *k_pool, float *v_pool, const int32_t *block_table,
                  int n_layers, int block_size, int layer,
                  int total_tokens, float scale)
{
    size_t stride_l, stride_kh, stride_b;
    int t, blk, off, kh;

    if (nh <= 0 || nh > ATTN_MAX_QHEADS) return -1;
    if (nkv <= 0 || nkv > ATTN_MAX_KVHEADS) return -2;
    if (nh % nkv != 0) return -3;
    if (nh / nkv > 4) return -4;
    if (hd < ATTN_ROT_DIM || hd > ATTN_MAX_HD || (hd & 3) != 0) return -5;
    if (total_tokens <= 0 || total_tokens > ATTN_MAX_TOKENS) return -6;
    if (block_size <= 0 || n_layers <= 0) return -7;
    if (layer < 0 || layer >= n_layers) return -8;

    /* 1. rmsnorm_heads(q) / rmsnorm_heads(k) */
    attn_rmsnorm_heads_f32(g_qn, q_raw, nh, hd, 2 * hd, q_norm_w, eps);
    attn_rmsnorm_heads_f32(g_kn, k_raw, nkv, hd, hd, k_norm_w, eps);

    /* 2. apply_rope on both */
    attn_rope_f32(g_qn, nh, hd, cos_row, sin_row);
    attn_rope_f32(g_kn, nkv, hd, cos_row, sin_row);

    /* 3. pool.append_kv(pool_idx, k_rot, v_raw).  The token being generated
     *    is at logical position total_tokens-1, which lives in the last
     *    block of the table at that block's offset. */
    t = total_tokens - 1;
    blk = block_table[t / block_size];
    off = t % block_size;
    stride_l = (size_t)nkv * block_size * hd;
    stride_kh = (size_t)block_size * hd;
    stride_b = (size_t)n_layers * stride_l;
    for (kh = 0; kh < nkv; kh++) {
        size_t base = (size_t)blk * stride_b + (size_t)layer * stride_l
                    + (size_t)kh * stride_kh + (size_t)off * hd;
        memcpy(k_pool + base, g_kn + (size_t)kh * hd,
               (size_t)hd * sizeof(float));
        memcpy(v_pool + base, v_raw + (size_t)kh * hd,
               (size_t)hd * sizeof(float));
    }

    /* 4. attention (exactly the kernel full_attn_f32 exposes) */
    full_attn_impl(g_attn, g_qn, k_pool, v_pool, block_table,
                   n_layers, nkv, nh, block_size, hd, layer, total_tokens,
                   scale);

    /* 5. gated_attn = a_out * sigmoid(gate) */
    attn_gate_f32(gated_out, g_attn, q_raw, nh, hd);
    return 0;
}

/* --------------------------------------------------------------------------
 * self test
 * -------------------------------------------------------------------------- */
#ifdef ATTN_SELFTEST

static float frand(void)
{
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static int run_case(int n_blocks, int total_tokens, int block_size,
                    int n_layers, int n_kv_heads, int n_q_heads,
                    int head_dim, int layer, float scale)
{
    size_t pool_elems = (size_t)n_blocks * n_layers * n_kv_heads * block_size * head_dim;
    float *k_pool = (float *)malloc(pool_elems * sizeof(float));
    float *v_pool = (float *)malloc(pool_elems * sizeof(float));
    int32_t *bt = (int32_t *)malloc((size_t)n_blocks * sizeof(int32_t));
    float *q = (float *)malloc((size_t)n_q_heads * head_dim * sizeof(float));
    float *o_neon = (float *)malloc((size_t)n_q_heads * head_dim * sizeof(float));
    float *o_ref  = (float *)malloc((size_t)n_q_heads * head_dim * sizeof(float));
    size_t i;
    int b, j;

    if (!k_pool || !v_pool || !bt || !q || !o_neon || !o_ref) return -1;

    for (i = 0; i < pool_elems; i++) {
        k_pool[i] = frand() * 0.5f;
        v_pool[i] = frand() * 0.5f;
    }
    for (i = 0; i < (size_t)n_q_heads * head_dim; i++) q[i] = frand();

    /* sequential block table: block j is physical block j */
    for (j = 0; j < n_blocks; j++) bt[j] = j;

    memset(o_neon, 0, (size_t)n_q_heads * head_dim * sizeof(float));
    memset(o_ref, 0, (size_t)n_q_heads * head_dim * sizeof(float));

    full_attn_f32(o_neon, q, k_pool, v_pool, bt,
                  n_layers, n_kv_heads, n_q_heads, block_size, head_dim,
                  layer, total_tokens, scale);
    full_attn_ref(o_ref, q, k_pool, v_pool, bt,
                  n_layers, n_kv_heads, n_q_heads, block_size, head_dim,
                  layer, total_tokens, scale);

    {
        double max_abs = 0.0, max_rel = 0.0;
        int n = n_q_heads * head_dim;
        for (i = 0; i < (size_t)n; i++) {
            double a = fabs((double)o_neon[i] - (double)o_ref[i]);
            double denom = fabs((double)o_ref[i]) + 1e-12;
            if (a > max_abs) max_abs = a;
            if (a / denom > max_rel) max_rel = a / denom;
        }
        printf("case blocks=%d T=%4d bs=%2d H=%d/%d D=%d  max_abs=%.3e max_rel=%.3e  %s\n",
               n_blocks, total_tokens, block_size, n_q_heads, n_kv_heads,
               head_dim, max_abs, max_rel,
               (max_abs < 1e-4) ? "PASS" : "FAIL");
        free(k_pool); free(v_pool); free(bt); free(q); free(o_neon); free(o_ref);
        return (max_abs < 1e-4) ? 0 : 1;
    }
}

int main(void)
{
    int fails = 0;
    printf("=== qwen_attn self test (NEON=%d) ===\n", HAVE_NEON);

    /* real shape, short context */
    fails += run_case(4, 16, 16, 6, 2, 8, 256, 0, 0.0625f);
    /* real shape, single block, partial */
    fails += run_case(1, 7, 16, 6, 2, 8, 256, 3, 0.0625f);
    /* real shape, crosses a block boundary */
    fails += run_case(4, 33, 16, 6, 2, 8, 256, 1, 0.0625f);
    /* longer: 544 tokens over 34 blocks */
    fails += run_case(34, 544, 16, 6, 2, 8, 256, 2, 0.0625f);
    /* full 2048 */
    fails += run_case(128, 2048, 16, 6, 2, 8, 256, 5, 0.0625f);

    printf("=== %s (%d failures) ===\n", fails ? "FAIL" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
#endif /* ATTN_SELFTEST */
