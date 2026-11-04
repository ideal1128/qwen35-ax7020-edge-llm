// =============================================================================
// qwen_sampling.c -- temperature / top-k / top-p sampling kernels for
//                    Qwen3.5-0.8B inference on Zynq-7020 (Cortex-A9 x2, NEON)
//
// Target toolchain: arm-linux-gnueabihf-gcc 4.9.2, C99, OpenMP, NEON.
// Host toolchain:   any C99 compiler (scalar fallback path is used on x86).
//
// This file provides exactly two public functions and a host self test:
//
//   int lm_head_topk(const float* h, const uint16_t* embed_table,
//                    int vocab_size, int hidden_dim,
//                    int k, int* out_ids, float* out_scores);
//
//       Streaming top-k scan of the LM head.  Never materializes the
//       65536-float logits array (256 KB) and never re-reads the table.
//       Same NEON dot-product path as lm_head_argmax in qwen_kernels_opt.c:
//       bf16 -> f32 via vreinterpretq_f32_u32(vshll_n_u16(x, 16)),
//       vmlaq_f32 accumulation, two rows per pass, 2-thread OpenMP split.
//       Each thread keeps a small sorted insertion table of k entries; the
//       two tables are merged at the end.  Results are sorted by score
//       descending; ties are broken by ascending token id (same winner as
//       argmax for k == 1).
//
//   int sample_from_topk(const int* ids, const float* scores, int n,
//                        float temperature, int top_k, float top_p,
//                        unsigned int* rng_state);
//
//       temperature / top-k / top-p sampling from an already sorted
//       (descending) candidate list.  Uses an in-file xorshift32 PRNG, so a
//       given rng_state reproduces exactly.  temperature < 1e-6 (including
//       <= 0) degrades to argmax = ids[0].
//
// Build (x86 host self test):
//   gcc -O2 -DSAMPLING_SELFTEST -o qwen_sampling_test.exe qwen_sampling.c -lm
//
// Build (ARM board, shared library loaded from Python via ctypes):
//   arm-linux-gnueabihf-gcc -O3 -march=armv7-a -mfpu=neon -mfloat-abi=hard
//     -fopenmp -DUSE_NEON -shared -fPIC -o libqwen_sampling.so qwen_sampling.c -lm
//   (the two lines are one command; the line break is only for readability)
//
// NEON selection:
//   NEON code is compiled when the compiler defines __ARM_NEON / __ARM_NEON__
//   (gcc does this for -mfpu=neon), or when the caller passes -DUSE_NEON as
//   in the board command above.  Define SAMPLING_NO_NEON to force the portable
//   scalar path (used for the x86 host test).
//
// Performance expectation:
//   lm_head_topk reads the same 128 MiB bf16 table as lm_head_argmax and does
//   the same number of MACs.  The only extra work per row is one float compare
//   against the current k-th best score (kept in a register) plus a rare
//   sorted insertion (roughly k * ln(vocab/k) inserts per thread, e.g. about
//   360 for k = 50).  The scan is DRAM-bandwidth bound (measured argmax:
//   128 MiB / 114 ms = 1122 MB/s, 89% of the theoretical limit), so the
//   expected cost is within ~1-2% of lm_head_argmax, i.e. about 115-117 ms
//   for k in the usual 20..100 range on the board.  The merge of two k-entry
//   tables is O(k^2) worst case but only k entries, negligible.
//
// Notes:
//   - C99 only.  Declarations are placed at block starts so the file also
//     builds in gnu90 mode (the board gcc 4.9.2 default).
//   - Pure ASCII source.
//   - No global mutable state in the library path.
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(SAMPLING_NO_NEON) && \
    (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(USE_NEON))
#include <arm_neon.h>
#define SAMPLING_NEON 1
#endif

// Prefetch distance/level, kept identical to qwen_kernels_opt.c so that the
// two kernels behave the same on the board.
#ifndef PFD
#define PFD 256
#endif
#ifndef PFL
#define PFL 0
#endif

// ---------------------------------------------------------------------------
// bf16 -> f32
// The embedding table stores the high 16 bits of an IEEE-754 float.  Shifting
// left by 16 reconstructs the float bit pattern.  The scalar version uses
// memcpy to avoid strict-aliasing undefined behaviour; the compiler folds it
// into a single shift/move at -O2.
// ---------------------------------------------------------------------------
static float bf16_to_f32(uint16_t u) {
    uint32_t bits = ((uint32_t)u) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// Per-thread top-k insertion table.
// Kept sorted by score descending; ties keep insertion order, and because rows
// are visited in ascending token order the tie order is ascending token id.
// ---------------------------------------------------------------------------
typedef struct {
    int    cap;
    int    count;
    int*   ids;
    float* scores;
} tk_table;

static void tk_offer(tk_table* t, int id, float score) {
    int pos;
    if (t->count < t->cap) {
        pos = t->count;
        while (pos > 0 && t->scores[pos - 1] < score) {
            t->scores[pos] = t->scores[pos - 1];
            t->ids[pos] = t->ids[pos - 1];
            --pos;
        }
        t->scores[pos] = score;
        t->ids[pos] = id;
        t->count++;
    } else if (t->cap > 0 && score > t->scores[t->cap - 1]) {
        pos = t->cap - 1;
        while (pos > 0 && t->scores[pos - 1] < score) {
            t->scores[pos] = t->scores[pos - 1];
            t->ids[pos] = t->ids[pos - 1];
            --pos;
        }
        t->scores[pos] = score;
        t->ids[pos] = id;
    }
}

static float tk_threshold(const tk_table* t) {
    if (t->count < t->cap) return -FLT_MAX;
    return t->scores[t->cap - 1];
}

// Merge src into dst.  Both tables are sorted descending.  Inserting src in
// descending order keeps dst sorted, and the strict ">" test inside tk_offer
// makes equal scores keep dst's entries first.  dst is the first half of the
// vocabulary, so its ids are smaller and the ascending-id tie rule holds.
static void tk_merge(tk_table* dst, const tk_table* src) {
    int i;
    for (i = 0; i < src->count; i++) {
        tk_offer(dst, src->ids[i], src->scores[i]);
    }
}

// ---------------------------------------------------------------------------
// Scan one vocabulary range [start, end) into a top-k table.
// ---------------------------------------------------------------------------
#if defined(SAMPLING_NEON)
static void scan_range_neon(const float* h, const uint16_t* embed_table,
                            int hidden_dim, int start, int end, tk_table* tk) {
    float thr = -FLT_MAX;
    int t = start;

    // Two rows per pass: the h[] loads are shared by both dot products,
    // exactly as in lm_head_argmax.
    for (; t + 1 < end; t += 2) {
        const uint16_t* rowA = embed_table + (size_t)t * (size_t)hidden_dim;
        const uint16_t* rowB = rowA + hidden_dim;
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f);
        float32x4_t b1 = vdupq_n_f32(0.0f);
        float scoreA;
        float scoreB;
        int i = 0;

        for (; i + 8 <= hidden_dim; i += 8) {
            float32x4_t h0;
            float32x4_t h1;
            uint16x8_t uA;
            uint16x8_t uB;
            __builtin_prefetch(rowA + i + PFD, 0, PFL);
            __builtin_prefetch(rowB + i + PFD, 0, PFL);
            h0 = vld1q_f32(h + i);
            h1 = vld1q_f32(h + i + 4);
            uA = vld1q_u16(rowA + i);
            a0 = vmlaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(uA), 16)), h0);
            a1 = vmlaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(uA), 16)), h1);
            uB = vld1q_u16(rowB + i);
            b0 = vmlaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(uB), 16)), h0);
            b1 = vmlaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(uB), 16)), h1);
        }
        {
            float32x4_t accA = vaddq_f32(a0, a1);
            float32x2_t sA = vadd_f32(vget_low_f32(accA), vget_high_f32(accA));
            float32x4_t accB = vaddq_f32(b0, b1);
            float32x2_t sB = vadd_f32(vget_low_f32(accB), vget_high_f32(accB));
            sA = vpadd_f32(sA, sA);
            sB = vpadd_f32(sB, sB);
            scoreA = vget_lane_f32(sA, 0);
            scoreB = vget_lane_f32(sB, 0);
        }
        for (; i < hidden_dim; i++) {
            scoreA += h[i] * bf16_to_f32(rowA[i]);
            scoreB += h[i] * bf16_to_f32(rowB[i]);
        }
        if (scoreA > thr) {
            tk_offer(tk, t, scoreA);
            thr = tk_threshold(tk);
        }
        if (scoreB > thr) {
            tk_offer(tk, t + 1, scoreB);
            thr = tk_threshold(tk);
        }
    }

    for (; t < end; t++) {
        const uint16_t* row = embed_table + (size_t)t * (size_t)hidden_dim;
        float32x4_t c0 = vdupq_n_f32(0.0f);
        float32x4_t c1 = vdupq_n_f32(0.0f);
        float score;
        int i = 0;
        for (; i + 8 <= hidden_dim; i += 8) {
            uint16x8_t u;
            __builtin_prefetch(row + i + PFD, 0, PFL);
            u = vld1q_u16(row + i);
            c0 = vmlaq_f32(c0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u), 16)),
                           vld1q_f32(h + i));
            c1 = vmlaq_f32(c1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u), 16)),
                           vld1q_f32(h + i + 4));
        }
        {
            float32x4_t acc = vaddq_f32(c0, c1);
            float32x2_t s2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            s2 = vpadd_f32(s2, s2);
            score = vget_lane_f32(s2, 0);
        }
        for (; i < hidden_dim; i++) score += h[i] * bf16_to_f32(row[i]);
        if (score > thr) {
            tk_offer(tk, t, score);
            thr = tk_threshold(tk);
        }
    }
}
#endif  // SAMPLING_NEON

#if !defined(SAMPLING_NEON)
static void scan_range_scalar(const float* h, const uint16_t* embed_table,
                              int hidden_dim, int start, int end, tk_table* tk) {
    float thr = -FLT_MAX;
    int t;
    for (t = start; t < end; t++) {
        const uint16_t* row = embed_table + (size_t)t * (size_t)hidden_dim;
        float score = 0.0f;
        int i;
        for (i = 0; i < hidden_dim; i++) {
            score += h[i] * bf16_to_f32(row[i]);
        }
        if (score > thr) {
            tk_offer(tk, t, score);
            thr = tk_threshold(tk);
        }
    }
}
#endif  // !SAMPLING_NEON

static void scan_range(const float* h, const uint16_t* embed_table,
                       int hidden_dim, int start, int end, tk_table* tk) {
#if defined(SAMPLING_NEON)
    scan_range_neon(h, embed_table, hidden_dim, start, end, tk);
#else
    scan_range_scalar(h, embed_table, hidden_dim, start, end, tk);
#endif
}

// ---------------------------------------------------------------------------
// Public: streaming top-k LM head scan.
// Returns min(k, vocab_size) (0 on invalid arguments / allocation failure).
// out_ids / out_scores must hold at least k entries (min(k, vocab_size) used).
// ---------------------------------------------------------------------------
int lm_head_topk(const float* h, const uint16_t* embed_table,
                 int vocab_size, int hidden_dim,
                 int k, int* out_ids, float* out_scores) {
    int kk;
    int* ids_buf;
    float* score_buf;
    tk_table tabs[2];
    int n;
    int i;

    if (!h || !embed_table || !out_ids || !out_scores) return 0;
    if (vocab_size <= 0 || hidden_dim <= 0 || k <= 0) return 0;

    kk = (k < vocab_size) ? k : vocab_size;

    ids_buf = (int*)malloc((size_t)kk * 2u * sizeof(int));
    score_buf = (float*)malloc((size_t)kk * 2u * sizeof(float));
    if (!ids_buf || !score_buf) {
        free(ids_buf);
        free(score_buf);
        return 0;
    }

    for (i = 0; i < 2; i++) {
        tabs[i].cap = kk;
        tabs[i].count = 0;
        tabs[i].ids = ids_buf + (size_t)i * (size_t)kk;
        tabs[i].scores = score_buf + (size_t)i * (size_t)kk;
    }

#if defined(_OPENMP)
    {
        int scanned[2];
        int chunk = vocab_size / 2;
        scanned[0] = 0;
        scanned[1] = 0;

        #pragma omp parallel num_threads(2)
        {
            int tid = omp_get_thread_num();
            int start = tid * chunk;
            int end = (tid == 1) ? vocab_size : (start + chunk);
            scan_range(h, embed_table, hidden_dim, start, end, &tabs[tid]);
            scanned[tid] = 1;
        }

        // If the runtime honoured fewer than two threads (OMP_DYNAMIC etc.),
        // redo the scan serially so the result is still correct.  This is
        // never expected on the board, where the runtime is not dynamic.
        if (!scanned[0] || !scanned[1]) {
            tabs[0].count = 0;
            tabs[1].count = 0;
            scan_range(h, embed_table, hidden_dim, 0, vocab_size, &tabs[0]);
        }

        tk_merge(&tabs[0], &tabs[1]);
    }
#else
    scan_range(h, embed_table, hidden_dim, 0, vocab_size, &tabs[0]);
#endif

    n = tabs[0].count;
    if (n > 0) {
        memcpy(out_ids, tabs[0].ids, (size_t)n * sizeof(int));
        memcpy(out_scores, tabs[0].scores, (size_t)n * sizeof(float));
    }

    free(ids_buf);
    free(score_buf);
    return n;
}

// ---------------------------------------------------------------------------
// xorshift32 PRNG.  Deterministic and self-contained (no rand()).
// A zero state is remapped to a fixed non-zero constant so the generator can
// never lock up at zero.
// ---------------------------------------------------------------------------
static unsigned int xorshift32_next(unsigned int* state) {
    unsigned int x = *state;
    if (x == 0u) x = 0x9E3779B9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Uniform float in [0, 1).
static float xorshift32_unit(unsigned int* state) {
    return (float)(xorshift32_next(state) >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------------
// Candidate sorting helper (only used if sample_from_topk is called with an
// unsorted list; lm_head_topk always produces a sorted list).
// ---------------------------------------------------------------------------
typedef struct {
    int   id;
    float score;
} sample_pair;

static int sample_pair_cmp(const void* a, const void* b) {
    const sample_pair* pa = (const sample_pair*)a;
    const sample_pair* pb = (const sample_pair*)b;
    if (pa->score > pb->score) return -1;
    if (pa->score < pb->score) return 1;
    if (pa->id < pb->id) return -1;
    if (pa->id > pb->id) return 1;
    return 0;
}

// ---------------------------------------------------------------------------
// Public: temperature / top-k / top-p sampling.
//
//   ids / scores : candidate list, expected sorted by score descending (the
//                  output of lm_head_topk).  An unsorted list is sorted on a
//                  temporary copy, so the function is still correct.
//   n            : number of candidates.
//   temperature  : <= 0 or < 1e-6 -> argmax (ids[0] of the sorted list).
//   top_k        : > 0 -> keep only the first min(top_k, n) candidates.
//   top_p        : in (0, 1] -> keep the smallest prefix whose cumulative
//                  probability exceeds top_p (at least one token).  top_p <= 0
//                  keeps one token, top_p > 1 keeps all.
//   rng_state    : xorshift32 state, updated in place.  Same state + same
//                  arguments => same result.
//
// Returns the selected token id, or -1 for invalid arguments.
// ---------------------------------------------------------------------------
int sample_from_topk(const int* ids, const float* scores, int n,
                     float temperature, int top_k, float top_p,
                     unsigned int* rng_state) {
    sample_pair* scratch = NULL;
    int sorted;
    int m;
    int i;
    int keep;
    int chosen;
    float maxf;
    float inv_t;
    double sum_all;
    double sum_keep;
    double cum;
    double r;
    double u;
    float p_thresh;

#define SMP_IDX(idx) (scratch ? scratch[(idx)].id : ids[(idx)])
#define SMP_SCORE(idx) (scratch ? scratch[(idx)].score : scores[(idx)])

    if (!ids || !scores || n <= 0) return -1;
    if (!rng_state) return ids[0];

    sorted = 1;
    for (i = 1; i < n; i++) {
        if (scores[i] > scores[i - 1]) {
            sorted = 0;
            break;
        }
    }
    if (!sorted) {
        scratch = (sample_pair*)malloc((size_t)n * sizeof(sample_pair));
        if (scratch) {
            for (i = 0; i < n; i++) {
                scratch[i].id = ids[i];
                scratch[i].score = scores[i];
            }
            qsort(scratch, (size_t)n, sizeof(sample_pair), sample_pair_cmp);
        }
    }

    // Greedy mode: exact argmax, bit-for-bit reproducible.
    if (!(temperature >= 1e-6f)) {
        chosen = SMP_IDX(0);
        free(scratch);
        return chosen;
    }

    m = n;
    if (top_k > 0 && top_k < m) m = top_k;

    maxf = SMP_SCORE(0);
    for (i = 1; i < m; i++) {
        float s = SMP_SCORE(i);
        if (s > maxf) maxf = s;
    }
    inv_t = 1.0f / temperature;

    // Pass 1: normalizer.  expf((s - max) / T) cannot overflow because the
    // largest exponent is 0.
    sum_all = 0.0;
    for (i = 0; i < m; i++) {
        sum_all += (double)expf((SMP_SCORE(i) - maxf) * inv_t);
    }
    if (!(sum_all > 0.0)) {
        chosen = SMP_IDX(0);
        free(scratch);
        return chosen;
    }

    // top-p: smallest prefix whose cumulative probability exceeds top_p.
    // top_p is clamped to [0, 1]; NaN and <= 0 keep one token, > 1 keeps all.
    p_thresh = top_p;
    if (!(p_thresh > 0.0f)) p_thresh = 0.0f;
    if (p_thresh > 1.0f) p_thresh = 1.0f;

    keep = m;
    cum = 0.0;
    for (i = 0; i < m; i++) {
        cum += (double)expf((SMP_SCORE(i) - maxf) * inv_t) / sum_all;
        if (cum > (double)p_thresh) {
            keep = i + 1;
            break;
        }
    }
    if (keep < 1) keep = 1;

    // Renormalize over the kept set before drawing, so the kept tokens'
    // probabilities sum to one.
    sum_keep = 0.0;
    for (i = 0; i < keep; i++) {
        sum_keep += (double)expf((SMP_SCORE(i) - maxf) * inv_t);
    }
    if (!(sum_keep > 0.0)) {
        chosen = SMP_IDX(0);
        free(scratch);
        return chosen;
    }

    u = (double)xorshift32_unit(rng_state);
    r = u * sum_keep;
    cum = 0.0;
    chosen = keep - 1;
    for (i = 0; i < keep; i++) {
        cum += (double)expf((SMP_SCORE(i) - maxf) * inv_t);
        if (r < cum) {
            chosen = i;
            break;
        }
    }

    chosen = SMP_IDX(chosen);
    free(scratch);
    return chosen;

#undef SMP_IDX
#undef SMP_SCORE
}

// ===========================================================================
// Host self test
// ===========================================================================
#ifdef SAMPLING_SELFTEST

static unsigned int g_test_rng = 0x13579BDFu;

static uint16_t f32_to_bf16_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static void fill_random_bf16(uint16_t* tab, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        unsigned int r = xorshift32_next(&g_test_rng);
        int v = (int)((r >> 16) & 0xFFFFu) - 32768;
        float f = (float)v * (1.0f / 16384.0f);
        tab[i] = f32_to_bf16_bits(f);
    }
}

static void fill_random_f32(float* x, int n) {
    int i;
    for (i = 0; i < n; i++) {
        unsigned int r = xorshift32_next(&g_test_rng);
        int v = (int)((r >> 16) & 0xFFFFu) - 32768;
        x[i] = (float)v * (1.0f / 16384.0f);
    }
}

// Brute force: full logits then qsort by (score desc, id asc).
static void brute_force_rank(const float* h, const uint16_t* tab,
                             int vocab, int hidden, sample_pair* ranked) {
    int t;
    int i;
    for (t = 0; t < vocab; t++) {
        const uint16_t* row = tab + (size_t)t * (size_t)hidden;
        float s = 0.0f;
        for (i = 0; i < hidden; i++) s += h[i] * bf16_to_f32(row[i]);
        ranked[t].score = s;
        ranked[t].id = t;
    }
    qsort(ranked, (size_t)vocab, sizeof(sample_pair), sample_pair_cmp);
}

static int g_failures = 0;

static void report(int ok, const char* name) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) g_failures++;
}

// Compare lm_head_topk against brute force for one (vocab, hidden, k).
static void check_topk_against_brute(const char* name,
                                     const float* h, const uint16_t* tab,
                                     int vocab, int hidden, int k,
                                     const sample_pair* ranked) {
    int* ids = (int*)malloc((size_t)k * sizeof(int));
    float* scores = (float*)malloc((size_t)k * sizeof(float));
    int n;
    int expect;
    int i;
    int ok = 1;

    if (!ids || !scores) {
        printf("[FAIL] %s (malloc failed)\n", name);
        g_failures++;
        free(ids);
        free(scores);
        return;
    }

    n = lm_head_topk(h, tab, vocab, hidden, k, ids, scores);
    expect = (k < vocab) ? k : vocab;
    if (n != expect) ok = 0;

    for (i = 0; ok && i < n; i++) {
        if (ids[i] != ranked[i].id) ok = 0;
        if (i > 0 && scores[i] > scores[i - 1]) ok = 0;   // must be descending
    }
    if (ok) {
        for (i = 0; i < n; i++) {
            float ref = ranked[i].score;
            float d = scores[i] - ref;
            float tol = 1e-3f * (1.0f + (ref < 0.0f ? -ref : ref));
            if (d < 0.0f) d = -d;
            if (d > tol) {
                ok = 0;
                break;
            }
        }
    }

    printf("[%s] %s (vocab=%d hidden=%d k=%d -> n=%d)\n",
           ok ? "PASS" : "FAIL", name, vocab, hidden, k, n);
    if (!ok) g_failures++;
    free(ids);
    free(scores);
}

// The OpenMP path splits the vocabulary in half and merges two top-k tables.
// Exercise exactly that code path without requiring an OpenMP runtime: scan
// the two halves into separate tables and merge, then compare with a single
// full-range scan.
static void check_merge_equals_full(const float* h, const uint16_t* tab,
                                    int vocab, int hidden, int k) {
    tk_table full;
    tk_table ta;
    tk_table tb;
    int* fi = (int*)malloc((size_t)k * sizeof(int));
    int* ai = (int*)malloc((size_t)k * sizeof(int));
    int* bi = (int*)malloc((size_t)k * sizeof(int));
    float* fs = (float*)malloc((size_t)k * sizeof(float));
    float* as = (float*)malloc((size_t)k * sizeof(float));
    float* bs = (float*)malloc((size_t)k * sizeof(float));
    int i;
    int ok = 1;

    if (!fi || !ai || !bi || !fs || !as || !bs) {
        printf("[FAIL] merge test (malloc failed)\n");
        g_failures++;
        free(fi); free(ai); free(bi);
        free(fs); free(as); free(bs);
        return;
    }

    full.cap = k; full.count = 0; full.ids = fi; full.scores = fs;
    ta.cap = k; ta.count = 0; ta.ids = ai; ta.scores = as;
    tb.cap = k; tb.count = 0; tb.ids = bi; tb.scores = bs;

    scan_range(h, tab, hidden, 0, vocab, &full);
    scan_range(h, tab, hidden, 0, vocab / 2, &ta);
    scan_range(h, tab, hidden, vocab / 2, vocab, &tb);
    tk_merge(&ta, &tb);

    if (ta.count != full.count) ok = 0;
    for (i = 0; ok && i < full.count; i++) {
        if (ta.ids[i] != full.ids[i]) ok = 0;
        if (ta.scores[i] != full.scores[i]) ok = 0;
    }

    report(ok, "parallel half-merge result == full-range scan");
    free(fi); free(ai); free(bi);
    free(fs); free(as); free(bs);
}

// Distribution check: 10000 draws with temperature = 1, top_p = 1 must track
// the softmax probabilities.  The requirement is 5% tolerance on the
// high-frequency tokens (p >= 0.20); deviations of every token are printed.
static void check_distribution(void) {
    float scores[4];
    int ids[4];
    int counts[4];
    double p[4];
    double sum;
    double max_rel;
    unsigned int st;
    int i;
    int j;
    int ok = 1;

    scores[0] = 0.0f; scores[1] = 0.0f; scores[2] = -1.0f; scores[3] = -1.0f;
    ids[0] = 100; ids[1] = 101; ids[2] = 102; ids[3] = 103;
    for (i = 0; i < 4; i++) counts[i] = 0;

    sum = 0.0;
    for (i = 0; i < 4; i++) {
        p[i] = exp((double)scores[i]);
        sum += p[i];
    }
    for (i = 0; i < 4; i++) p[i] /= sum;

    st = 0xC0FFEE01u;
    for (i = 0; i < 10000; i++) {
        int id = sample_from_topk(ids, scores, 4, 1.0f, 0, 1.0f, &st);
        for (j = 0; j < 4; j++) {
            if (id == ids[j]) counts[j]++;
        }
    }

    max_rel = 0.0;
    for (i = 0; i < 4; i++) {
        double emp = (double)counts[i] / 10000.0;
        double dev = emp - p[i];
        double rel;
        if (dev < 0.0) dev = -dev;
        rel = dev / p[i];
        if (rel > max_rel) max_rel = rel;
        printf("  token %d: softmax=%.5f empirical=%.5f rel_dev=%.2f%%\n",
               ids[i], p[i], emp, rel * 100.0);
        if (p[i] >= 0.20 && rel > 0.05) ok = 0;
    }
    printf("  max relative deviation = %.2f%% (limit 5%% for p>=0.20)\n",
           max_rel * 100.0);
    report(ok, "temperature=1 top_p=1 distribution matches softmax (5%)");
}

static void check_greedy_and_repro(void) {
    float scores[4];
    int ids[4];
    unsigned int st;
    int i;
    int ok;

    scores[0] = 0.0f; scores[1] = -0.5f; scores[2] = -1.0f; scores[3] = -1.5f;
    ids[0] = 7; ids[1] = 8; ids[2] = 9; ids[3] = 10;

    // temperature <= 0 or < 1e-6 -> argmax.
    st = 0xDEADBEEFu;
    ok = (sample_from_topk(ids, scores, 4, 0.0f, 0, 1.0f, &st) == ids[0]);
    ok = ok && (sample_from_topk(ids, scores, 4, -1.0f, 0, 1.0f, &st) == ids[0]);
    ok = ok && (sample_from_topk(ids, scores, 4, 1e-7f, 0, 1.0f, &st) == ids[0]);
    report(ok, "temperature=0 / -1 / 1e-7 -> argmax (ids[0])");

    // top_k = 1 always returns scores[0].
    st = 0x11223344u;
    ok = 1;
    for (i = 0; i < 1000; i++) {
        if (sample_from_topk(ids, scores, 4, 1.0f, 1, 1.0f, &st) != ids[0]) {
            ok = 0;
            break;
        }
    }
    report(ok, "top_k=1 always returns scores[0] (1000 draws)");

    // Reproducibility: same seed -> identical sequence; different seed -> not.
    {
        unsigned int s1 = 0xDEADBEEFu;
        unsigned int s2 = 0xDEADBEEFu;
        unsigned int s3 = 0x12345678u;
        int same = 1;
        int diff = 0;
        for (i = 0; i < 1000; i++) {
            int a = sample_from_topk(ids, scores, 4, 1.0f, 0, 1.0f, &s1);
            int b = sample_from_topk(ids, scores, 4, 1.0f, 0, 1.0f, &s2);
            int c = sample_from_topk(ids, scores, 4, 1.0f, 0, 1.0f, &s3);
            if (a != b) same = 0;
            if (a != c) diff = 1;
        }
        report(same, "same rng seed -> identical 1000-draw sequence");
        report(diff, "different rng seed -> different sequence");
    }
}

// top-k / top-p truncation behaviour on a list whose probabilities are
// 0.3655, 0.3655, 0.1345, 0.1345.
static void check_truncation(void) {
    float scores[4];
    int ids[4];
    int counts[4];
    unsigned int st;
    int i;
    int j;
    int ok;

    scores[0] = 0.0f; scores[1] = 0.0f; scores[2] = -1.0f; scores[3] = -1.0f;
    ids[0] = 100; ids[1] = 101; ids[2] = 102; ids[3] = 103;

    // top_p = 0.5: 0.3655 is not enough, 0.731 is -> keep the first 2.
    st = 0xABCD0001u;
    ok = 1;
    for (i = 0; i < 5000; i++) {
        int id = sample_from_topk(ids, scores, 4, 1.0f, 0, 0.5f, &st);
        if (id != ids[0] && id != ids[1]) {
            ok = 0;
            break;
        }
    }
    report(ok, "top_p=0.5 keeps exactly the first 2 candidates");

    // top_p = 0.3: first probability 0.3655 > 0.3 -> keep only the argmax.
    st = 0xABCD0002u;
    ok = 1;
    for (i = 0; i < 1000; i++) {
        if (sample_from_topk(ids, scores, 4, 1.0f, 0, 0.3f, &st) != ids[0]) {
            ok = 0;
            break;
        }
    }
    report(ok, "top_p=0.3 keeps only the argmax");

    // A tiny top_p still keeps at least one token.
    st = 0xABCD0003u;
    ok = 1;
    for (i = 0; i < 1000; i++) {
        if (sample_from_topk(ids, scores, 4, 1.0f, 0, 1e-9f, &st) != ids[0]) {
            ok = 0;
            break;
        }
    }
    report(ok, "top_p=1e-9 keeps only the argmax");

    // top_k = 2 with top_p = 1 -> only the first 2.
    st = 0xABCD0004u;
    ok = 1;
    for (i = 0; i < 5000; i++) {
        int id = sample_from_topk(ids, scores, 4, 1.0f, 2, 1.0f, &st);
        if (id != ids[0] && id != ids[1]) {
            ok = 0;
            break;
        }
    }
    report(ok, "top_k=2 with top_p=1 keeps exactly the first 2");

    // top_p = 1 with no top_k -> every candidate can be emitted.
    st = 0xABCD0005u;
    for (i = 0; i < 4; i++) counts[i] = 0;
    for (i = 0; i < 10000; i++) {
        int id = sample_from_topk(ids, scores, 4, 1.0f, 0, 1.0f, &st);
        for (j = 0; j < 4; j++) {
            if (id == ids[j]) counts[j]++;
        }
    }
    ok = (counts[0] > 0 && counts[1] > 0 && counts[2] > 0 && counts[3] > 0);
    report(ok, "top_p=1 top_k=0 can emit every candidate");

    // top_k larger than n must behave like no truncation.
    st = 0xABCD0006u;
    ok = 1;
    for (i = 0; i < 1000; i++) {
        int id = sample_from_topk(ids, scores, 4, 1.0f, 10, 1.0f, &st);
        if (id < ids[0] || id > ids[3]) ok = 0;
    }
    report(ok, "top_k > n is harmless");
}

// If the caller passes an unsorted list (not produced by lm_head_topk), the
// function sorts a temporary copy; greedy mode must still find the true argmax.
static void check_unsorted(void) {
    int ids[3];
    float scores[3];
    unsigned int st;
    int ok;

    ids[0] = 10; ids[1] = 11; ids[2] = 12;
    scores[0] = 0.0f; scores[1] = 2.0f; scores[2] = 1.0f;

    st = 0x5555AAAAu;
    ok = (sample_from_topk(ids, scores, 3, 0.0f, 0, 1.0f, &st) == ids[1]);
    report(ok, "unsorted input: temperature=0 returns true argmax (id 11)");

    st = 0x5555AAAAu;
    ok = (sample_from_topk(ids, scores, 3, 1.0f, 0, 1e-9f, &st) == ids[1]);
    report(ok, "unsorted input: top_p~0 returns true argmax (id 11)");
}

static void check_edges(void) {
    int ids[4];
    float scores[4];
    int ok;

    ids[0] = 1; ids[1] = 2; ids[2] = 3; ids[3] = 4;
    scores[0] = 0.0f; scores[1] = -0.1f; scores[2] = -0.2f; scores[3] = -0.3f;

    ok = (sample_from_topk(ids, scores, 0, 1.0f, 0, 1.0f, NULL) == -1);
    report(ok, "n=0 returns -1");

    ok = (sample_from_topk(NULL, scores, 4, 1.0f, 0, 1.0f, NULL) == -1);
    report(ok, "NULL ids returns -1");

    ok = (sample_from_topk(ids, scores, 1, 1.0f, 0, 1.0f, NULL) == ids[0]);
    report(ok, "n=1 returns the only candidate");

    ok = (lm_head_topk(NULL, NULL, 0, 0, 0, NULL, NULL) == 0);
    report(ok, "lm_head_topk invalid arguments return 0");
}

int main(void) {
    int vocab;
    int hidden;
    uint16_t* tab;
    float* h;
    sample_pair* ranked;
    int one_id;
    float one_score;
    int n;

    printf("=== qwen_sampling self test ===\n");
#if defined(SAMPLING_NEON)
    printf("dot-product path: NEON\n");
#else
    printf("dot-product path: portable scalar fallback\n");
#endif

    // ---------------------------------------------------------------
    // Case 1: small table, exhaustive brute-force comparison.
    // ---------------------------------------------------------------
    printf("--- case 1: vocab=1000 hidden=64, brute-force comparison ---\n");
    vocab = 1000;
    hidden = 64;
    tab = (uint16_t*)malloc((size_t)vocab * (size_t)hidden * sizeof(uint16_t));
    h = (float*)malloc((size_t)hidden * sizeof(float));
    ranked = (sample_pair*)malloc((size_t)vocab * sizeof(sample_pair));
    if (!tab || !h || !ranked) {
        printf("[FAIL] case 1 allocation failed\n");
        return 1;
    }
    fill_random_bf16(tab, (size_t)vocab * (size_t)hidden);
    fill_random_f32(h, hidden);
    brute_force_rank(h, tab, vocab, hidden, ranked);

    check_topk_against_brute("brute force match", h, tab, vocab, hidden, 1, ranked);
    check_topk_against_brute("brute force match", h, tab, vocab, hidden, 5, ranked);
    check_topk_against_brute("brute force match", h, tab, vocab, hidden, 20, ranked);
    check_topk_against_brute("brute force match", h, tab, vocab, hidden, 100, ranked);
    check_topk_against_brute("brute force match", h, tab, vocab, hidden, 1000, ranked);

    n = lm_head_topk(h, tab, vocab, hidden, 1, &one_id, &one_score);
    report(n == 1 && one_id == ranked[0].id,
           "k=1 equals brute-force argmax");

    check_merge_equals_full(h, tab, vocab, hidden, 20);

    n = lm_head_topk(h, tab, vocab, hidden, 0, &one_id, &one_score);
    report(n == 0, "k=0 returns 0");

    {
        int* big_ids = (int*)malloc((size_t)vocab * sizeof(int));
        float* big_scores = (float*)malloc((size_t)vocab * sizeof(float));
        if (big_ids && big_scores) {
            n = lm_head_topk(h, tab, vocab, hidden, vocab + 50, big_ids, big_scores);
            report(n == vocab, "k > vocab returns vocab entries");
        } else {
            printf("[FAIL] guard test allocation failed\n");
            g_failures++;
        }
        free(big_ids);
        free(big_scores);
    }

    free(tab);
    free(h);
    free(ranked);

    // ---------------------------------------------------------------
    // Case 2: hidden_dim not a multiple of 8 (NEON tail handling).
    // ---------------------------------------------------------------
    printf("--- case 2: hidden=67 (NEON/scalar tail) ---\n");
    vocab = 500;
    hidden = 67;
    tab = (uint16_t*)malloc((size_t)vocab * (size_t)hidden * sizeof(uint16_t));
    h = (float*)malloc((size_t)hidden * sizeof(float));
    ranked = (sample_pair*)malloc((size_t)vocab * sizeof(sample_pair));
    if (!tab || !h || !ranked) {
        printf("[FAIL] case 2 allocation failed\n");
        return 1;
    }
    fill_random_bf16(tab, (size_t)vocab * (size_t)hidden);
    fill_random_f32(h, hidden);
    brute_force_rank(h, tab, vocab, hidden, ranked);
    check_topk_against_brute("hidden not multiple of 8", h, tab, vocab, hidden, 7, ranked);
    free(tab);
    free(h);
    free(ranked);

    // ---------------------------------------------------------------
    // Case 3: the real board shape (65536 x 1024 bf16 = 128 MiB table).
    // ---------------------------------------------------------------
    printf("--- case 3: realistic shape vocab=65536 hidden=1024 ---\n");
    vocab = 65536;
    hidden = 1024;
    tab = (uint16_t*)malloc((size_t)vocab * (size_t)hidden * sizeof(uint16_t));
    h = (float*)malloc((size_t)hidden * sizeof(float));
    ranked = (sample_pair*)malloc((size_t)vocab * sizeof(sample_pair));
    if (!tab || !h || !ranked) {
        printf("[FAIL] case 3 allocation failed (need 128 MiB)\n");
        g_failures++;
    } else {
        fill_random_bf16(tab, (size_t)vocab * (size_t)hidden);
        fill_random_f32(h, hidden);
        brute_force_rank(h, tab, vocab, hidden, ranked);
        check_topk_against_brute("realistic shape", h, tab, vocab, hidden, 1, ranked);
        check_topk_against_brute("realistic shape", h, tab, vocab, hidden, 20, ranked);
        check_topk_against_brute("realistic shape", h, tab, vocab, hidden, 64, ranked);
        free(tab);
        free(h);
        free(ranked);
    }

    // ---------------------------------------------------------------
    // Sampling tests.
    // ---------------------------------------------------------------
    printf("--- sampling tests ---\n");
    check_greedy_and_repro();
    check_distribution();
    check_truncation();
    check_unsorted();
    check_edges();

    printf("================================\n");
    if (g_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d TEST(S) FAILED\n", g_failures);
    }
    return (g_failures == 0) ? 0 : 1;
}

#endif  // SAMPLING_SELFTEST
