/* pl_apply.c -- finish the fabric GEMV on the CPU.
 *
 * The fabric returns one int32 block dot product per 64 weights:
 *
 *   dots[m][b] = sum_{i<64} w_int8[m][b*64+i] * x_int16[b*64+i]
 *
 * and the BFP8 exponent plus the per-block x scale are applied here:
 *
 *   y[m] = sum_b dots[m][b] * 2**(exps[m][b] - 134) * xscale[b]
 *
 * This is the same expression qwen_kernels.c evaluates, in the same order, so
 * the result is bit-identical to the engine's own bfp8_gemv.
 *
 * Why it exists: numpy evaluates the same thing but costs about 11 ms per GEMV
 * on this 766 MHz Cortex-A9 (1.8 s per token), which is more than the DMA it is
 * post-processing.  In C it is about 0.5 ms per GEMV.
 *
 * Build (on the board):
 *   gcc -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -shared -fPIC \
 *       -o /mnt/qwen/libpl_apply.so /mnt/qwen/pl_apply.c -lm
 */

#include <math.h>
#include <stdint.h>

static float EXP_SCALE[256];
static int exp_ready = 0;

static void init_exp(void)
{
    int i;
    for (i = 0; i < 256; ++i)
        EXP_SCALE[i] = (float)ldexp(1.0, i - 134);
    exp_ready = 1;
}

/* x quantisation, exactly as qwen_kernels.c does it:
 *   amax = max|x| over the 64-element block
 *   amax > 1e-20 : q = 16384/amax, xscale = amax/16384, xq = lrintf(x*q)
 *   else         : xscale = 0, xq = 0
 */
void pl_quant(const float *x, int16_t *xq, float *xscale, int nb)
{
    int b, i;
    for (b = 0; b < nb; ++b) {
        const float *xb = x + b * 64;
        float amax = 0.0f, q, xs;
        for (i = 0; i < 64; ++i) {
            float v = fabsf(xb[i]);
            if (v > amax)
                amax = v;
        }
        if (amax > 1e-20f) {
            q = 16384.0f / amax;
            xs = amax / 16384.0f;
        } else {
            q = 0.0f;
            xs = 0.0f;
        }
        for (i = 0; i < 64; ++i)
            xq[b * 64 + i] = (int16_t)lrintf(xb[i] * q);
        xscale[b] = xs;
    }
}

void pl_apply(const int32_t *dots, const uint8_t *exps,
              const float *xscale, float *y, int M, int nb)
{
    int m, b;
    if (!exp_ready)
        init_exp();
    for (m = 0; m < M; ++m) {
        const int32_t *d = dots + (long)m * nb;
        const uint8_t *e = exps + (long)m * nb;
        float acc = 0.0f;
        for (b = 0; b < nb; ++b)
            acc += (float)d[b] * (EXP_SCALE[e[b]] * xscale[b]);
        y[m] = acc;
    }
}
