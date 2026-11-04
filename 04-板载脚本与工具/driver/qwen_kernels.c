#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

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

void rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rsqrt_val = 1.0f / sqrtf((sum_sq / (float)n) + eps);
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * rsqrt_val * w[i];
    }
}

void bfp8_gemv(float* y, const float* x, const int8_t* mantissas, const uint8_t* exps, int M, int K) {
    if (!TABLE_INITIALIZED) init_kernels();
    int num_blocks = K / 64;

    for (int m = 0; m < M; m++) {
        float sum = 0.0f;
        const int8_t* row_m = mantissas + m * K;
        const uint8_t* row_e = exps + m * num_blocks;

        for (int b = 0; b < num_blocks; b++) {
            float scale = EXP_SCALE[row_e[b]];
            const int8_t* blk_m = row_m + b * 64;
            const float* blk_x = x + b * 64;
            float block_dot = 0.0f;

            #pragma GCC unroll 8
            for (int i = 0; i < 64; i++) {
                block_dot += (float)blk_m[i] * blk_x[i];
            }
            sum += block_dot * scale;
        }
        y[m] = sum;
    }
}

void swiglu(float* y, const float* gate, const float* up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        y[i] = silu * up[i];
    }
}

void add_residual(float* x, const float* res, int n) {
    for (int i = 0; i < n; i++) {
        x[i] += res[i];
    }
}

int lm_head_argmax(const float* h, const uint16_t* embed_table, int vocab_size, int hidden_dim) {
    float best_score = -1e30f;
    int best_id = 0;

    for (int t = 0; t < vocab_size; t++) {
        const uint16_t* row_u16 = embed_table + t * hidden_dim;
        float score = 0.0f;

        #pragma GCC unroll 8
        for (int i = 0; i < hidden_dim; i++) {
            // BF16 to float32 conversion
            uint32_t u32 = ((uint32_t)row_u16[i]) << 16;
            float val;
            memcpy(&val, &u32, 4);
            score += h[i] * val;
        }

        if (score > best_score) {
            best_score = score;
            best_id = t;
        }
    }
    return best_id;
}

