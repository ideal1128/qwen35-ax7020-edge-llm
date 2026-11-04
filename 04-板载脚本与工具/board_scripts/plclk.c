/* plclk.c -- measure PL AXI-Lite read latency, try raising FCLK0, restore.
 *
 * The PL clock divider lives in SLCR FPGA0_CLK_CTRL (0xF8000170).
 * Current value on this board is 0x00200500.  If the effective divider is
 * DIVISOR[25:20] * DIVISOR1[13:8] then it is 2*5 = 10 -> 100 MHz from a
 * 1000 MHz IO PLL.  We try 0x00100500 (1*5 = 5 -> 200 MHz) and measure whether
 * PL read latency actually drops, then restore the original value.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double lat_ns(volatile uint32_t *p, long n)
{
    uint32_t s = 0;
    double t0 = now_ns();
    for (long i = 0; i < n; i++) s += p[0];
    double t1 = now_ns();
    if (s == 0xdeadbeefu) printf("!");
    return (t1 - t0) / (double)n;
}

int main(int argc, char **argv)
{
    long n = 300000;
    if (argc > 1) n = atol(argv[1]);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return 1; }

    volatile uint32_t *slcr = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                   MAP_SHARED, fd, 0xF8000000);
    volatile uint32_t *pl   = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                   MAP_SHARED, fd, 0x43C00000);
    if (slcr == MAP_FAILED || pl == MAP_FAILED) { perror("mmap"); return 1; }

    volatile uint32_t *fclk0 = slcr + (0x170 / 4);

    uint32_t orig = *fclk0;
    printf("FPGA0_CLK_CTRL orig = 0x%08x\n", orig);

    double a = lat_ns(pl, n);
    printf("PL  read latency (orig clock) = %.1f ns\n", a);

    /* try 200 MHz: DIVISOR=1, DIVISOR1=5, DIVISOR0=0 */
    uint32_t fast = 0x00100500u;
    *fclk0 = fast;
    usleep(20000);
    uint32_t rb = *fclk0;
    printf("FPGA0_CLK_CTRL wrote 0x%08x, readback 0x%08x\n", fast, rb);

    double b = lat_ns(pl, n);
    printf("PL  read latency (200 MHz try) = %.1f ns\n", b);

    /* restore */
    *fclk0 = orig;
    usleep(20000);
    printf("FPGA0_CLK_CTRL restored to 0x%08x (readback 0x%08x)\n", orig, *fclk0);

    double c = lat_ns(pl, n);
    printf("PL  read latency (restored) = %.1f ns\n", c);
    printf("RATIO orig/try = %.3f\n", (b > 0.0) ? a / b : 0.0);
    return 0;
}
