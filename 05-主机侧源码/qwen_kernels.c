// =============================================================================
//  qwen_kernels.c  --  M2a version
//
//  与原始版本相比只做三件事，ABI 完全不变（Python 侧 ctypes 签名一个字都不用改）：
//    1. bfp8_gemv        : 标量内积 -> NEON 8 路展开 + int8->float 向量化
//    2. lm_head_argmax   : 标量 -> NEON bf16 点积 + 双线程分块归约（确定性合并）
//    3. gdn_recurrent_step / causal_conv1d_update_silu : 按 head / channel 开 OpenMP
//
//  编译（板上）：
//    gcc -std=c99 -O3 -mfpu=neon -mfloat-abi=hard -fopenmp -shared -fPIC \
//        qwen_kernels.c -o libqwen_kernels.so -lm
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#endif

// 软件预取：实测把 PS 侧读带宽从 818 拉到 1259 MB/s（见 membw.c）
#ifndef PFD
#define PFD 256
#endif
#ifndef PFL
#define PFL 0
#endif

static float EXP_SCALE[256];
static int TABLE_INITIALIZED = 0;

void init_kernels() {
    if (!TABLE_INITIALIZED) {
        for (int i = 0; i < 256; i++) {
            EXP_SCALE[i] = ldexpf(1.0f, i - 134);
        }
        TABLE_INITIALIZED = 1;
    }
}

// ---------------------------------------------------------------------------
// 1. RMSNorm  [n] (n=1024)
// ---------------------------------------------------------------------------
void rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
    float sum_sq = 0.0f;
#if defined(USE_NEON)
    float32x4_t s4 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        s4 = vmlaq_f32(s4, v, v);
    }
    float32x2_t s2 = vadd_f32(vget_low_f32(s4), vget_high_f32(s4));
    s2 = vpadd_f32(s2, s2);
    sum_sq = vget_lane_f32(s2, 0);
    for (; i < n; i++) sum_sq += x[i] * x[i];
#else
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
#endif
    float rsqrt_val = 1.0f / sqrtf((sum_sq / (float)n) + eps);
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * rsqrt_val * (1.0f + w[i]);
    }
}

// ---------------------------------------------------------------------------
// 2. BFP8 GEMV:  y[M] = W[M,K] * x[K]，W 是 block-64 块浮点（int8 尾数 + 每块 1 字节指数）
//
//    NEON 关键点：int8 不能直接乘 float，必须 vmovl_s8 -> vmovl_s16 -> vcvtq_f32_s32。
//    每 8 个权重：1 次 s8 载入 + 2 次 widen + 2 次 cvt + 2 次 vmla = 7 条指令 / 8 元素，
//    比标量（每元素 1 次 cvt + 1 次 fma）快 2~3 倍，且不再依赖 GCC 4.9 的自动向量化。
// ---------------------------------------------------------------------------
#if defined(USE_NEON)
static inline float dot64_bfp8_neon(const int8_t* m, const float* x) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (int i = 0; i < 64; i += 8) {
        int8x8_t w8 = vld1_s8(m + i);
        int16x8_t w16 = vmovl_s8(w8);
        float32x4_t flo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
        float32x4_t fhi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
        acc0 = vmlaq_f32(acc0, flo, vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, fhi, vld1q_f32(x + i + 4));
    }
    float32x4_t acc = vaddq_f32(acc0, acc1);
    float32x2_t s2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    s2 = vpadd_f32(s2, s2);
    return vget_lane_f32(s2, 0);
}
#endif

void bfp8_gemv(float* y, const float* x, const int8_t* mantissas,
               const uint8_t* exps, int M, int K) {
    if (!TABLE_INITIALIZED) init_kernels();
    const int num_blocks = K / 64;
    const int KB = num_blocks * 64;

#if defined(USE_NEON)
    // ---- 整数通路 -----------------------------------------------------------
    // A9 的 VCVT.F32.S32 吞吐极差（实测约 4 周期/元素），float 版内积只能跑到
    // 388 MB/s。改成：x 每 64 元素块量化成 int16（BFP16，块内独立指数），
    // 权重保持 int8，内积用 VMLAL.S16 全整数累加，最后每块乘一次浮点缩放。
    // 量化只做一次（K<=8192），所有 M 行复用。
    static int16_t g_xq[8192];
#if defined(X8)
    static int8_t g_xq8[8192];
#endif
    static float g_xscale[128];
    if (K <= 8192 && num_blocks <= 128) {
#if defined(X8)
        // ---- int8 activation path -------------------------------------------
        // x is quantised to int8 per 64-block instead of int16.  That lets the
        // inner product use VMULL.S8 (int8x8 -> int16x8) plus VPADALQ.S16
        // (pairwise accumulate into int32), which removes the 2x VMOVL.S8 widen
        // per 16 weights that the int16 path needs: 7 NEON ops per 16 weights
        // per row instead of 9.  Measured issue-bound, so ops/byte is what pays.
        for (int b = 0; b < num_blocks; b++) {
            const float* xb = x + b * 64;
            float amax = 0.0f;
            for (int i = 0; i < 64; i++) {
                float a = fabsf(xb[i]);
                if (a > amax) amax = a;
            }
            int8_t* d = g_xq8 + b * 64;
            if (amax > 1e-20f) {
                float q = 127.0f / amax;
                g_xscale[b] = amax / 127.0f;
                for (int i = 0; i < 64; i++) d[i] = (int8_t)lrintf(xb[i] * q);
            } else {
                g_xscale[b] = 0.0f;
                for (int i = 0; i < 64; i++) d[i] = 0;
            }
        }
        const int npairs8 = M >> 1;
        #pragma omp parallel for schedule(static) if (npairs8 >= 32)
        for (int p = 0; p < npairs8; p++) {
            const int m = p * 2;
            const int8_t* r0 = mantissas + (size_t)m * K;
            const int8_t* r1 = r0 + K;
            const uint8_t* e0 = exps + (size_t)m * num_blocks;
            const uint8_t* e1 = e0 + num_blocks;
            float s0 = 0.0f, s1 = 0.0f;
            for (int b = 0; b < num_blocks; b++) {
                const int8_t* xb = g_xq8 + b * 64;
                const int8_t* w0 = r0 + b * 64;
                const int8_t* w1 = r1 + b * 64;
                int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
                int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0);
                for (int i = 0; i < 64; i += 16) {
                    __builtin_prefetch(w0 + i + PFD, 0, 0);
                    __builtin_prefetch(w1 + i + PFD, 0, 0);
                    int8x16_t v0 = vld1q_s8(w0 + i);
                    int8x16_t v1 = vld1q_s8(w1 + i);
                    int8x8_t xa = vld1_s8(xb + i);
                    int8x8_t xc = vld1_s8(xb + i + 8);
                    a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(v0), xa));
                    a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(v0), xc));
                    c0 = vpadalq_s16(c0, vmull_s8(vget_low_s8(v1), xa));
                    c1 = vpadalq_s16(c1, vmull_s8(vget_high_s8(v1), xc));
                }
                int32x4_t A = vaddq_s32(a0, a1);
                int32x4_t C = vaddq_s32(c0, c1);
                int32_t sa = vgetq_lane_s32(A, 0) + vgetq_lane_s32(A, 1) +
                             vgetq_lane_s32(A, 2) + vgetq_lane_s32(A, 3);
                int32_t sc = vgetq_lane_s32(C, 0) + vgetq_lane_s32(C, 1) +
                             vgetq_lane_s32(C, 2) + vgetq_lane_s32(C, 3);
                s0 += (float)sa * EXP_SCALE[e0[b]] * g_xscale[b];
                s1 += (float)sc * EXP_SCALE[e1[b]] * g_xscale[b];
            }
            y[m] = s0;
            y[m + 1] = s1;
        }
        if (M & 1) {   // odd row tail
            const int m = M - 1;
            const int8_t* row_m = mantissas + (size_t)m * K;
            const uint8_t* row_e = exps + (size_t)m * num_blocks;
            float sum = 0.0f;
            for (int b = 0; b < num_blocks; b++) {
                const int8_t* wb = row_m + b * 64;
                const int8_t* xb = g_xq8 + b * 64;
                int32x4_t acc0 = vdupq_n_s32(0);
                int32x4_t acc1 = vdupq_n_s32(0);
                for (int i = 0; i < 64; i += 16) {
                    int8x16_t w = vld1q_s8(wb + i);
                    int8x8_t x0 = vld1_s8(xb + i);
                    int8x8_t x1 = vld1_s8(xb + i + 8);
                    acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(w), x0));
                    acc1 = vpadalq_s16(acc1, vmull_s8(vget_high_s8(w), x1));
                }
                int32x4_t acc = vaddq_s32(acc0, acc1);
                int32_t s = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) +
                            vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);
                sum += (float)s * EXP_SCALE[row_e[b]] * g_xscale[b];
            }
            for (int i = KB; i < K; i++) sum += (float)row_m[i] * x[i];
            y[m] = sum;
        }
        return;
#else
        for (int b = 0; b < num_blocks; b++) {
            const float* xb = x + b * 64;
            float amax = 0.0f;
            for (int i = 0; i < 64; i++) {
                float a = fabsf(xb[i]);
                if (a > amax) amax = a;
            }
            int16_t* d = g_xq + b * 64;
            if (amax > 1e-20f) {
                float q = 16384.0f / amax;
                g_xscale[b] = amax / 16384.0f;
                for (int i = 0; i < 64; i++) d[i] = (int16_t)lrintf(xb[i] * q);
            } else {
                g_xscale[b] = 0.0f;
                for (int i = 0; i < 64; i++) d[i] = 0;
            }
        }
        // 两行一组：x 的 int16 载入被两行共享，每 16 元素的 NEON 指令从 9 条降到 7.5 条，
        // 让内核从「NEON 发射受限」转为「DDR 带宽受限」。
        const int npairs = M >> 1;
        #pragma omp parallel for schedule(static) if (npairs >= 32)
        for (int p = 0; p < npairs; p++) {
            const int m = p * 2;
            const int8_t* r0 = mantissas + (size_t)m * K;
            const int8_t* r1 = r0 + K;
            const uint8_t* e0 = exps + (size_t)m * num_blocks;
            const uint8_t* e1 = e0 + num_blocks;
            float s0 = 0.0f, s1 = 0.0f;
            for (int b = 0; b < num_blocks; b++) {
                const int16_t* xb = g_xq + b * 64;
                const int8_t* w0 = r0 + b * 64;
                const int8_t* w1 = r1 + b * 64;
                int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
                int32x4_t c0 = vdupq_n_s32(0), c1 = vdupq_n_s32(0);
                for (int i = 0; i < 64; i += 16) {
                    __builtin_prefetch(w0 + i + PFD, 0, PFL);
                    __builtin_prefetch(w1 + i + PFD, 0, PFL);
                    int8x16_t v0 = vld1q_s8(w0 + i);
                    int8x16_t v1 = vld1q_s8(w1 + i);
                    int16x8_t v0l = vmovl_s8(vget_low_s8(v0));
                    int16x8_t v0h = vmovl_s8(vget_high_s8(v0));
                    int16x8_t v1l = vmovl_s8(vget_low_s8(v1));
                    int16x8_t v1h = vmovl_s8(vget_high_s8(v1));
                    int16x8_t xa = vld1q_s16(xb + i);
                    int16x8_t xc = vld1q_s16(xb + i + 8);
                    a0 = vmlal_s16(a0, vget_low_s16(v0l), vget_low_s16(xa));
                    a0 = vmlal_s16(a0, vget_high_s16(v0l), vget_high_s16(xa));
                    a1 = vmlal_s16(a1, vget_low_s16(v0h), vget_low_s16(xc));
                    a1 = vmlal_s16(a1, vget_high_s16(v0h), vget_high_s16(xc));
                    c0 = vmlal_s16(c0, vget_low_s16(v1l), vget_low_s16(xa));
                    c0 = vmlal_s16(c0, vget_high_s16(v1l), vget_high_s16(xa));
                    c1 = vmlal_s16(c1, vget_low_s16(v1h), vget_low_s16(xc));
                    c1 = vmlal_s16(c1, vget_high_s16(v1h), vget_high_s16(xc));
                }
                int32x4_t A = vaddq_s32(a0, a1);
                int32x4_t C = vaddq_s32(c0, c1);
                int32_t sa = vgetq_lane_s32(A, 0) + vgetq_lane_s32(A, 1) +
                             vgetq_lane_s32(A, 2) + vgetq_lane_s32(A, 3);
                int32_t sc = vgetq_lane_s32(C, 0) + vgetq_lane_s32(C, 1) +
                             vgetq_lane_s32(C, 2) + vgetq_lane_s32(C, 3);
                s0 += (float)sa * EXP_SCALE[e0[b]] * g_xscale[b];
                s1 += (float)sc * EXP_SCALE[e1[b]] * g_xscale[b];
            }
            y[m] = s0;
            y[m + 1] = s1;
        }
        if (M & 1) {   // 奇数行尾巴
            const int m = M - 1;
            const int8_t* row_m = mantissas + (size_t)m * K;
            const uint8_t* row_e = exps + (size_t)m * num_blocks;
            float sum = 0.0f;
            for (int b = 0; b < num_blocks; b++) {
                const int8_t* wb = row_m + b * 64;
                const int16_t* xb = g_xq + b * 64;
                int32x4_t acc0 = vdupq_n_s32(0);
                int32x4_t acc1 = vdupq_n_s32(0);
                for (int i = 0; i < 64; i += 16) {
                    int8x16_t w = vld1q_s8(wb + i);
                    int16x8_t wlo = vmovl_s8(vget_low_s8(w));
                    int16x8_t whi = vmovl_s8(vget_high_s8(w));
                    int16x8_t x0 = vld1q_s16(xb + i);
                    int16x8_t x1 = vld1q_s16(xb + i + 8);
                    acc0 = vmlal_s16(acc0, vget_low_s16(wlo), vget_low_s16(x0));
                    acc0 = vmlal_s16(acc0, vget_high_s16(wlo), vget_high_s16(x0));
                    acc1 = vmlal_s16(acc1, vget_low_s16(whi), vget_low_s16(x1));
                    acc1 = vmlal_s16(acc1, vget_high_s16(whi), vget_high_s16(x1));
                }
                int32x4_t acc = vaddq_s32(acc0, acc1);
                int32_t s = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) +
                            vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);
                sum += (float)s * EXP_SCALE[row_e[b]] * g_xscale[b];
            }
            for (int i = KB; i < K; i++) sum += (float)row_m[i] * x[i];
            y[m] = sum;
        }
        return;
#endif
    }
#endif

    #pragma omp parallel for schedule(static) if (M >= 64)
    for (int m = 0; m < M; m++) {
        const int8_t* row_m = mantissas + (size_t)m * K;
        const uint8_t* row_e = exps + (size_t)m * num_blocks;
        float sum = 0.0f;
#if defined(USE_NEON)
        for (int b = 0; b < num_blocks; b++) {
            sum += dot64_bfp8_neon(row_m + b * 64, x + b * 64) * EXP_SCALE[row_e[b]];
        }
#else
        for (int b = 0; b < num_blocks; b++) {
            float scale = EXP_SCALE[row_e[b]];
            const int8_t* blk_m = row_m + b * 64;
            const float* blk_x = x + b * 64;
            float block_dot = 0.0f;
            #pragma GCC unroll 8
            for (int i = 0; i < 64; i++) block_dot += (float)blk_m[i] * blk_x[i];
            sum += block_dot * scale;
        }
#endif
        for (int i = KB; i < K; i++) sum += (float)row_m[i] * x[i];
        y[m] = sum;
    }
}

// ---------------------------------------------------------------------------
// Fast vector exp for 4 floats: exp(x) = 2^n * poly(2^f).
// Relative error ~1e-7 (about 1.7 ulp), i.e. the same order as libm expf, so
// SiLU / softplus results stay numerically indistinguishable from the scalar
// expf path while removing the per-element libm call.
// ---------------------------------------------------------------------------
#if defined(USE_NEON)
static inline float32x4_t exp_neon(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    float32x4_t fx = vmulq_f32(x, vdupq_n_f32(1.44269504088896341f));
    float32x4_t magic = vdupq_n_f32(12582912.0f);   // 1.5 * 2^23
    float32x4_t t = vaddq_f32(fx, magic);
    float32x4_t n = vsubq_f32(t, magic);            // round(fx)
    float32x4_t f = vsubq_f32(fx, n);               // [-0.5, 0.5]
    float32x4_t p = vdupq_n_f32(1.5403530393381608e-4f);
    p = vmlaq_f32(vdupq_n_f32(1.3333558146428443e-3f), p, f);
    p = vmlaq_f32(vdupq_n_f32(9.6181291076284770e-3f), p, f);
    p = vmlaq_f32(vdupq_n_f32(5.5504108664821580e-2f), p, f);
    p = vmlaq_f32(vdupq_n_f32(2.4022650695910070e-1f), p, f);
    p = vmlaq_f32(vdupq_n_f32(6.9314718055994530e-1f), p, f);
    p = vmlaq_f32(vdupq_n_f32(1.0f), p, f);
    int32x4_t e = vcvtq_s32_f32(n);
    e = vaddq_s32(e, vdupq_n_s32(127));
    float32x4_t sc = vreinterpretq_f32_s32(vshlq_n_s32(e, 23));
    return vmulq_f32(p, sc);
}

// ARMv7 NEON has no vector divide.  Doing the four scalar VDIV.F32 keeps the
// result bit-identical to the original `x / (1 + expf(-x))` expression, and at
// ~14 cycles per lane it costs well under a millisecond per token.
static inline float32x4_t div_neon(float32x4_t a, float32x4_t b) {
    float ta[4], tb[4];
    vst1q_f32(ta, a);
    vst1q_f32(tb, b);
    ta[0] /= tb[0];
    ta[1] /= tb[1];
    ta[2] /= tb[2];
    ta[3] /= tb[3];
    return vld1q_f32(ta);
}
#endif

void swiglu(float* y, const float* gate, const float* up, int n) {
    int i = 0;
#if defined(USE_NEON)
    #pragma omp parallel for schedule(static) if (n >= 2048)
    for (i = 0; i < (n & ~3); i += 4) {
        float32x4_t g = vld1q_f32(gate + i);
        float32x4_t e = exp_neon(vnegq_f32(g));
        float32x4_t s = div_neon(g, vaddq_f32(vdupq_n_f32(1.0f), e));
        vst1q_f32(y + i, vmulq_f32(s, vld1q_f32(up + i)));
    }
#else
    #pragma omp parallel for schedule(static) if (n >= 2048)
    for (i = 0; i < n; i++) {
        float g = gate[i];
        y[i] = (g / (1.0f + expf(-g))) * up[i];
    }
#endif
    for (i = (n & ~3); i < n; i++) {
        float g = gate[i];
        y[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void add_residual(float* x, const float* res, int n) {
    for (int i = 0; i < n; i++) x[i] += res[i];
}

// ---------------------------------------------------------------------------
// 3. LM Head argmax：embedding 表是 bf16，65536 x 1024 = 128 MB/token
//    bf16 -> float 就是把 16 位搬到高半字（vshll_n_u16 ... 16），零成本。
//    双线程分块 + 确定性归约（分数相同时取 id 小的），保证与单线程结果一致。
// ---------------------------------------------------------------------------
#if defined(USE_NEON)
static inline float dot_bf16_neon(const uint16_t* row, const float* h, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __builtin_prefetch(row + i + PFD, 0, PFL);
        uint16x8_t u = vld1q_u16(row + i);
        float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u), 16));
        float32x4_t hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u), 16));
        acc0 = vmlaq_f32(acc0, lo, vld1q_f32(h + i));
        acc1 = vmlaq_f32(acc1, hi, vld1q_f32(h + i + 4));
    }
    float32x4_t acc = vaddq_f32(acc0, acc1);
    float32x2_t s2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    s2 = vpadd_f32(s2, s2);
    float s = vget_lane_f32(s2, 0);
    for (; i < n; i++) {
        uint32_t u32 = ((uint32_t)row[i]) << 16;
        float v; memcpy(&v, &u32, 4);
        s += h[i] * v;
    }
    return s;
}
#endif

#if defined(USE_NEON)
static inline void dot_bf16_neon2(const uint16_t* r0, const uint16_t* r1,
                                  const float* h, int n, float* o0, float* o1) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t b0 = vdupq_n_f32(0.0f), b1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __builtin_prefetch(r0 + i + PFD, 0, PFL);
        __builtin_prefetch(r1 + i + PFD, 0, PFL);
        uint16x8_t u0 = vld1q_u16(r0 + i);
        uint16x8_t u1 = vld1q_u16(r1 + i);
        float32x4_t h0 = vld1q_f32(h + i);
        float32x4_t h1 = vld1q_f32(h + i + 4);
        a0 = vmlaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u0), 16)), h0);
        a1 = vmlaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u0), 16)), h1);
        b0 = vmlaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u1), 16)), h0);
        b1 = vmlaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u1), 16)), h1);
    }
    float32x4_t A = vaddq_f32(a0, a1), B = vaddq_f32(b0, b1);
    float32x2_t sa = vpadd_f32(vadd_f32(vget_low_f32(A), vget_high_f32(A)),
                               vadd_f32(vget_low_f32(A), vget_high_f32(A)));
    float32x2_t sb = vpadd_f32(vadd_f32(vget_low_f32(B), vget_high_f32(B)),
                               vadd_f32(vget_low_f32(B), vget_high_f32(B)));
    float s0 = vget_lane_f32(sa, 0), s1 = vget_lane_f32(sb, 0);
    for (; i < n; i++) {
        uint32_t u32;
        float v;
        u32 = ((uint32_t)r0[i]) << 16; memcpy(&v, &u32, 4); s0 += h[i] * v;
        u32 = ((uint32_t)r1[i]) << 16; memcpy(&v, &u32, 4); s1 += h[i] * v;
    }
    *o0 = s0;
    *o1 = s1;
}
#endif

#ifndef GDPF
#define GDPF 2
#endif

#define ARGMAX_MAX_THREADS 8

int lm_head_argmax(const float* h, const uint16_t* embed_table,
                   int vocab_size, int hidden_dim) {
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
    if (nthreads > ARGMAX_MAX_THREADS) nthreads = ARGMAX_MAX_THREADS;
    if (nthreads < 1) nthreads = 1;
#endif
    float part_score[ARGMAX_MAX_THREADS];
    int   part_id[ARGMAX_MAX_THREADS];
    int   t;
    for (t = 0; t < ARGMAX_MAX_THREADS; t++) { part_score[t] = -1e30f; part_id[t] = 0; }

    #pragma omp parallel num_threads(nthreads) if (nthreads > 1)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
#else
        const int tid = 0, nth = 1;
#endif
        const int lo = (int)((long long)vocab_size * tid / nth);
        const int hi = (int)((long long)vocab_size * (tid + 1) / nth);
        float best_score = -1e30f;
        int best_id = lo;
        int v = lo;
#if defined(USE_NEON)
        for (; v + 1 < hi; v += 2) {
            float s0, s1;
            dot_bf16_neon2(embed_table + (size_t)v * hidden_dim,
                           embed_table + (size_t)(v + 1) * hidden_dim,
                           h, hidden_dim, &s0, &s1);
            if (s0 > best_score) { best_score = s0; best_id = v; }
            if (s1 > best_score) { best_score = s1; best_id = v + 1; }
        }
        if (v < hi) {
            float s0;
            s0 = dot_bf16_neon(embed_table + (size_t)v * hidden_dim, h, hidden_dim);
            if (s0 > best_score) { best_score = s0; best_id = v; }
        }
#else
        for (; v < hi; v++) {
            const uint16_t* row_u16 = embed_table + (size_t)v * hidden_dim;
            float score = 0.0f;
            #pragma GCC unroll 8
            for (int i = 0; i < hidden_dim; i++) {
                uint32_t u32 = ((uint32_t)row_u16[i]) << 16;
                float val; memcpy(&val, &u32, 4);
                score += h[i] * val;
            }
            if (score > best_score) { best_score = score; best_id = v; }
        }
#endif
        part_score[tid] = best_score;
        part_id[tid] = best_id;
    }

    // 确定性合并：按 tid 顺序，分数严格更大才替换（相同分数保留更小的 id）
    float best_score = -1e30f;
    int best_id = 0;
    for (t = 0; t < nthreads; t++) {
        if (part_score[t] > best_score) {
            best_score = part_score[t];
            best_id = part_id[t];
        }
    }
    return best_id;
}

// ---------------------------------------------------------------------------
// 4. Causal Depthwise Conv1D (4-tap) + SiLU
//
//    conv_state layout is now [3][channels] (was [channels][3]).  That makes
//    each tap a contiguous vector, so the whole 4-tap dot + SiLU can run 4
//    channels at a time in NEON with VLD4 deinterleaving the [channels][4]
//    weight rows.  The old [channels][3] layout forced a scalar stride-3
//    gather and one libm expf per channel.
// ---------------------------------------------------------------------------
void causal_conv1d_update_silu(
    float* y, const float* x, float* conv_state, const float* weight, int channels) {
    float* s0 = conv_state;
    float* s1 = conv_state + channels;
    float* s2 = conv_state + 2 * channels;
    int c = 0;
#if defined(USE_NEON)
    #pragma omp parallel for schedule(static) if (channels >= 512)
    for (c = 0; c < (channels & ~3); c += 4) {
        float32x4_t xc = vld1q_f32(x + c);
        float32x4x4_t wv = vld4q_f32(weight + (size_t)c * 4);
        float32x4_t t0 = vld1q_f32(s0 + c);
        float32x4_t t1 = vld1q_f32(s1 + c);
        float32x4_t t2 = vld1q_f32(s2 + c);
        float32x4_t d = vmulq_f32(wv.val[0], t0);
        d = vmlaq_f32(d, wv.val[1], t1);
        d = vmlaq_f32(d, wv.val[2], t2);
        d = vmlaq_f32(d, wv.val[3], xc);
        float32x4_t e = exp_neon(vnegq_f32(d));
        vst1q_f32(y + c, div_neon(d, vaddq_f32(vdupq_n_f32(1.0f), e)));
        vst1q_f32(s0 + c, t1);
        vst1q_f32(s1 + c, t2);
        vst1q_f32(s2 + c, xc);
    }
#endif
    #pragma omp parallel for schedule(static) if (channels >= 512)
    for (c = (channels & ~3); c < channels; c++) {
        float xc = x[c];
        float dot = weight[c * 4] * s0[c] + weight[c * 4 + 1] * s1[c] +
                    weight[c * 4 + 2] * s2[c] + weight[c * 4 + 3] * xc;
        y[c] = dot / (1.0f + expf(-dot));
        s0[c] = s1[c];
        s1[c] = s2[c];
        s2[c] = xc;
    }
}

// ---------------------------------------------------------------------------
// 5. Q/K per-head L2-norm
// ---------------------------------------------------------------------------
void qk_l2norm(float* q, float* k, int num_heads, int head_dim) {
    const float q_scale = 1.0f / sqrtf((float)head_dim);
    const float eps = 1e-6f;

    #pragma omp parallel for schedule(static) if (num_heads >= 4)
    for (int h = 0; h < num_heads; h++) {
        float* kh = k + h * head_dim;
        float* qh = q + h * head_dim;

        float k_sq = 0.0f, q_sq = 0.0f;
#if defined(USE_NEON)
        {
            float32x4_t a4 = vdupq_n_f32(0.0f), b4 = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 4 <= head_dim; i += 4) {
                float32x4_t kv = vld1q_f32(kh + i);
                float32x4_t qv = vld1q_f32(qh + i);
                a4 = vmlaq_f32(a4, kv, kv);
                b4 = vmlaq_f32(b4, qv, qv);
            }
            float32x2_t a2 = vpadd_f32(vadd_f32(vget_low_f32(a4), vget_high_f32(a4)),
                                       vadd_f32(vget_low_f32(a4), vget_high_f32(a4)));
            float32x2_t b2 = vpadd_f32(vadd_f32(vget_low_f32(b4), vget_high_f32(b4)),
                                       vadd_f32(vget_low_f32(b4), vget_high_f32(b4)));
            k_sq = vget_lane_f32(a2, 0);
            q_sq = vget_lane_f32(b2, 0);
            for (; i < head_dim; i++) { k_sq += kh[i] * kh[i]; q_sq += qh[i] * qh[i]; }
        }
#else
        for (int i = 0; i < head_dim; i++) { k_sq += kh[i] * kh[i]; q_sq += qh[i] * qh[i]; }
#endif
        float k_rnorm = 1.0f / sqrtf(k_sq + eps);
        float q_rnorm = q_scale / sqrtf(q_sq + eps);
#if defined(USE_NEON)
        {
            float32x4_t k4 = vdupq_n_f32(k_rnorm), q4 = vdupq_n_f32(q_rnorm);
            for (int i = 0; i + 4 <= head_dim; i += 4) {
                vst1q_f32(kh + i, vmulq_f32(vld1q_f32(kh + i), k4));
                vst1q_f32(qh + i, vmulq_f32(vld1q_f32(qh + i), q4));
            }
        }
#else
        for (int i = 0; i < head_dim; i++) { kh[i] *= k_rnorm; qh[i] *= q_rnorm; }
#endif
    }
}

// ---------------------------------------------------------------------------
// 6. GDN 单步递推，按 head 并行（每个 head 的 state 切片互不相交）
// ---------------------------------------------------------------------------
void gdn_recurrent_step(
    float* attn_out, float* state, const float* q, const float* k, const float* v,
    const float* alpha, const float* beta, int num_heads, int head_dim) {

    #pragma omp parallel for schedule(static) if (num_heads >= 4)
    for (int h = 0; h < num_heads; h++) {
        float kv_mem[128];
        float qs[128];
        float delta[128];
        float* S = state + (size_t)h * head_dim * head_dim;
        const float* qh = q + h * head_dim;
        const float* kh = k + h * head_dim;
        const float* vh = v + h * head_dim;
        float* outh = attn_out + h * head_dim;
        const float a = alpha[h];
        const float b = beta[h];
        const int N = head_dim * head_dim;

#if defined(USE_NEON)
        // One read of S: accumulate k^T S and q^T S in a single sweep and do
        // NOT store anything.  Then write S' = a*S + k*delta^T in a second
        // sweep.  The output is recovered algebraically:
        //     q^T S' = a*(q^T S) + (q.k)*delta
        // which is exact.  This drops the per-element out accumulator from the
        // second sweep and removes one full 1 MB read+write of state per layer
        // (4 MB -> 3 MB traffic), and the first sweep becomes read-only.
        float32x4_t a4 = vdupq_n_f32(a);
        memset(kv_mem, 0, head_dim * sizeof(float));
        memset(qs, 0, head_dim * sizeof(float));
        for (int i = 0; i < head_dim; i++) {
            const float* S_row = S + i * head_dim;
            __builtin_prefetch(S_row + head_dim * GDPF, 0, 0);
            float32x4_t ki4 = vdupq_n_f32(kh[i]);
            float32x4_t qi4 = vdupq_n_f32(qh[i]);
            for (int j = 0; j + 4 <= head_dim; j += 4) {
                float32x4_t s = vld1q_f32(S_row + j);
                vst1q_f32(kv_mem + j, vmlaq_f32(vld1q_f32(kv_mem + j), ki4, s));
                vst1q_f32(qs + j, vmlaq_f32(vld1q_f32(qs + j), qi4, s));
            }
        }

        float32x4_t b4 = vdupq_n_f32(b);
        for (int j = 0; j + 4 <= head_dim; j += 4)
            vst1q_f32(delta + j,
                      vmulq_f32(vsubq_f32(vld1q_f32(vh + j),
                                          vmulq_f32(vld1q_f32(kv_mem + j), a4)), b4));

        float qk = 0.0f;
        for (int j = 0; j < head_dim; j++) qk += qh[j] * kh[j];
        float32x4_t qk4 = vdupq_n_f32(qk);
        for (int j = 0; j + 4 <= head_dim; j += 4)
            vst1q_f32(outh + j,
                      vmlaq_f32(vmulq_f32(vld1q_f32(qs + j), a4), qk4,
                                vld1q_f32(delta + j)));

        for (int i = 0; i < head_dim; i++) {
            float* S_row = S + i * head_dim;
            __builtin_prefetch(S_row + head_dim * GDPF, 1, 0);
            float32x4_t ki4 = vdupq_n_f32(kh[i]);
            for (int j = 0; j + 4 <= head_dim; j += 4) {
                vst1q_f32(S_row + j,
                          vmlaq_f32(vmulq_f32(vld1q_f32(S_row + j), a4), ki4,
                                    vld1q_f32(delta + j)));
            }
        }
#else
        for (int idx = 0; idx < N; idx++) S[idx] *= a;
        memset(kv_mem, 0, head_dim * sizeof(float));
        for (int i = 0; i < head_dim; i++)
            for (int j = 0; j < head_dim; j++) kv_mem[j] += kh[i] * S[i * head_dim + j];
        for (int j = 0; j < head_dim; j++) delta[j] = b * (vh[j] - kv_mem[j]);
        for (int i = 0; i < head_dim; i++)
            for (int j = 0; j < head_dim; j++) S[i * head_dim + j] += kh[i] * delta[j];
        memset(outh, 0, head_dim * sizeof(float));
        for (int i = 0; i < head_dim; i++)
            for (int j = 0; j < head_dim; j++) outh[j] += qh[i] * S[i * head_dim + j];
#endif
    }
}

// ---------------------------------------------------------------------------
// 7. Per-head RMSNormGated
// ---------------------------------------------------------------------------
void rmsnorm_gated(float* y, const float* x, const float* z, const float* norm_w,
                   int num_heads, int head_dim, float eps) {
    #pragma omp parallel for schedule(static) if (num_heads >= 4)
    for (int h = 0; h < num_heads; h++) {
        const float* xh = x + h * head_dim;
        const float* zh = z + h * head_dim;
        float* yh = y + h * head_dim;
        float sum_sq = 0.0f;
#if defined(USE_NEON)
        {
            float32x4_t s4 = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 4 <= head_dim; i += 4) {
                float32x4_t xv = vld1q_f32(xh + i);
                s4 = vmlaq_f32(s4, xv, xv);
            }
            float32x2_t s2 = vadd_f32(vget_low_f32(s4), vget_high_f32(s4));
            s2 = vpadd_f32(s2, s2);
            sum_sq = vget_lane_f32(s2, 0);
            for (; i < head_dim; i++) sum_sq += xh[i] * xh[i];
        }
#else
        for (int i = 0; i < head_dim; i++) sum_sq += xh[i] * xh[i];
#endif
        float rsqrt_val = 1.0f / sqrtf((sum_sq / (float)head_dim) + eps);
        for (int i = 0; i < head_dim; i++) {
            float zi = zh[i];
            yh[i] = xh[i] * rsqrt_val * norm_w[i] * (zi / (1.0f + expf(-zi)));
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Gated-DeltaNet gate scalars, fused into one call.
//
//    Replaces, per linear layer: two 16x1024 numpy matvecs (~200 us each on this
//    board) plus six tiny-array ufuncs (sigmoid, softplus, exp) with a single
//    ctypes call.  Same float32 formula as the numpy version:
//        a = W_a . x ; b = W_b . x
//        beta  = sigmoid(b)
//        g     = neg_exp_alog * log1p(exp(a + dt))
//        alpha = exp(g)
// ---------------------------------------------------------------------------
void gdn_gates(float* alpha, float* beta, const float* x,
               const float* wa, const float* wb,
               const float* neg_exp_alog, const float* dt, int H, int K) {
    for (int h = 0; h < H; h++) {
        const float* ra = wa + (size_t)h * K;
        const float* rb = wb + (size_t)h * K;
        float av = 0.0f, bv = 0.0f;
        int i = 0;
#if defined(USE_NEON)
        {
            float32x4_t a4 = vdupq_n_f32(0.0f), b4 = vdupq_n_f32(0.0f);
            for (; i + 4 <= K; i += 4) {
                float32x4_t xv = vld1q_f32(x + i);
                a4 = vmlaq_f32(a4, vld1q_f32(ra + i), xv);
                b4 = vmlaq_f32(b4, vld1q_f32(rb + i), xv);
            }
            float32x2_t t;
            t = vpadd_f32(vadd_f32(vget_low_f32(a4), vget_high_f32(a4)),
                          vadd_f32(vget_low_f32(a4), vget_high_f32(a4)));
            av = vget_lane_f32(t, 0);
            t = vpadd_f32(vadd_f32(vget_low_f32(b4), vget_high_f32(b4)),
                          vadd_f32(vget_low_f32(b4), vget_high_f32(b4)));
            bv = vget_lane_f32(t, 0);
        }
#endif
        for (; i < K; i++) { av += ra[i] * x[i]; bv += rb[i] * x[i]; }
        beta[h] = 1.0f / (1.0f + expf(-bv));
        float z = av + dt[h];
        alpha[h] = expf(neg_exp_alog[h] * log1pf(expf(z)));
    }
}
