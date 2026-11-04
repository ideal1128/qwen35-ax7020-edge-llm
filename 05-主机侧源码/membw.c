// membw.c - 测量 Zynq-7020 PS 侧真实可持续读带宽（NEON 累加，纯读流）
// 用法: ./membw <file> [threads]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <arm_neon.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <file> [threads]\n", argv[0]); return 1; }
    const char* path = argv[1];
    int nth = (argc > 2) ? atoi(argv[2]) : 2;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t n = (size_t)st.st_size;
    const int8_t* p = (const int8_t*)mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }

    // 预热：触碰每 4 KB 一页，避免首次缺页计入计时
    volatile int8_t sink = 0;
    size_t wi;
    for (wi = 0; wi < n; wi += 4096) sink += p[wi];

    double t0 = now_s();
    int64_t total = 0;
    #pragma omp parallel num_threads(nth) reduction(+:total)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), nt = omp_get_num_threads();
#else
        int tid = 0, nt = 1;
#endif
        size_t lo = n * tid / nt;
        size_t hi = n * (tid + 1) / nt;
        lo &= ~(size_t)15; hi &= ~(size_t)15;
        int16x8_t acc = vdupq_n_s16(0);
        size_t i = lo;
        for (; i + 16 <= hi; i += 16) {
#ifdef PF
            __builtin_prefetch(p + i + PF, 0, 0);
#endif
            int8x16_t v = vld1q_s8(p + i);
            acc = vpadalq_s8(acc, v);
        }
        int32x4_t a32 = vpaddlq_s16(acc);
        int32_t s = vgetq_lane_s32(a32, 0) + vgetq_lane_s32(a32, 1) +
                    vgetq_lane_s32(a32, 2) + vgetq_lane_s32(a32, 3);
        total += s;
    }
    double dt = now_s() - t0;

    printf("file=%s bytes=%zu threads=%d time=%.3f s bw=%.1f MB/s (sink=%lld)\n",
           path, n, nth, dt, n / dt / 1e6, (long long)total);
    munmap((void*)p, n);
    close(fd);
    return 0;
}
