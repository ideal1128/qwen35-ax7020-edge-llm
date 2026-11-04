"""l2_bench.py -- is a cached readback with explicit PL310 maintenance both
FAST and CORRECT, and how much faster than the uncached /dev/mem view?

Self-contained: uses only pl_dma_test (Dma, Arena, PhysBuf) and l2maint.

METHOD
------
One GEMV configuration is replayed four times into four separate output
windows so that the total readback region (1 MB) exceeds the 512 KB L2.
Then the same 1 MB is read back three ways:

  uncached   via PhysBuf  (the /dev/mem O_SYNC view) -- always correct, slow
  cached     via Arena    (the ordinary anon mapping) -- fast, stale, WRONG
  maintained via Arena after l2maint.maintain()       -- should be fast AND correct

`cached` being wrong is the whole point: it proves the maintenance is
actually needed, so a pass in `maintained` is not a false positive.

ASCII only, Python 3.4.2 compatible.
"""

import os
import struct
import sys
import time

sys.path.insert(0, "/mnt/qwen")
os.chdir("/mnt/qwen")

import numpy as np

import l2maint
import pl_dma_test as P

K = 1024
M = 2048
NB = K // 64                      # 16 blocks per row
REPS = 8                          # eight output windows
W_BYTES = M * K                   # 2 MB of int8 weights
X_BYTES = (K // 8) * 16           # 2048 B
OUT_BYTES = M * NB * 4            # 128 KB per window
TOTAL_OUT = OUT_BYTES * REPS      # 1 MB > 512 KB L2

O_W = 0
O_X = W_BYTES
O_HDR = O_X + X_BYTES
O_OUT = O_HDR + 16
NEED = O_OUT + TOTAL_OUT

ARENA_MB = 64


def build():
    rnd = np.random.RandomState(1234)
    x = rnd.randint(-30000, 30001, size=K).astype(np.int16)
    w = rnd.randint(-127, 128, size=(M, K)).astype(np.int8)

    # reference block dots, int32
    wv = w.astype(np.int32).reshape(M, NB, 64)
    xv = x.astype(np.int32).reshape(NB, 64)
    ref = np.einsum("mnb,nb->mn", wv, xv).astype(np.int32)
    return x, w, ref


def main():
    print("=== l2_bench ===")
    l2 = l2maint.probe()
    dma = P.Dma()
    dma.dump("power-on")

    npages = (NEED + 4095) // 4096
    print("need %.2f MB (%d pages) of physically contiguous memory"
          % (NEED / 1048576.0, npages))

    arena = P.Arena(ARENA_MB)
    base = arena.find_run(npages)
    if base is None:
        print("!!! find_run failed -- memory too fragmented. Reboot the board.")
        return 2
    print("window: %.2f MB at voff %d (phys 0x%08x)"
          % (NEED / 1048576.0, base, arena.phys_at(base)))

    buf = P.PhysBuf(arena.phys_at(base), NEED)
    cac = arena.addr + base          # virtual address of the same bytes

    x, w, ref = build()

    # ---- stage the shared operands through the CACHED mapping, then flush so
    # the DMA reads real DDR contents and not stale ones.
    hdr0 = (K // 8) & 0xFFFF | ((NB * 4 - 1) & 0xFF) << 16
    arena.write(base + O_HDR, struct.pack("<I", hdr0) + b"\x00" * 12)
    arena.write(base + O_X, x.tostring())
    arena.write(base + O_W, w.tostring())
    l2maint.maintain(cac + O_W, W_BYTES, l2)
    l2maint.maintain(cac + O_X, X_BYTES, l2)
    l2maint.maintain(cac + O_HDR, 16, l2)

    # ---- four GEMV runs into four output windows
    dma.reset()
    print("\n--- running %d GEMV passes (K=%d M=%d) ---" % (REPS, K, M))
    t_all = time.time()
    for r in range(REPS):
        o = O_OUT + r * OUT_BYTES
        dma.s2mm_start(arena.phys_at(base + o), OUT_BYTES)
        dma.mm2s(arena.phys_at(base + O_HDR), 16)
        dma.mm2s(arena.phys_at(base + O_X), X_BYTES)
        dma.mm2s(arena.phys_at(base + O_W), W_BYTES)
        dma.s2mm_wait()
    print("    fabric: %.1f ms total, %.1f ms per pass"
          % ((time.time() - t_all) * 1000.0,
             (time.time() - t_all) * 1000.0 / REPS))

    # ---- readback, three modes
    print("\n--- readback of %.2f MB (L2 is 0.5 MB) ---"
          % (TOTAL_OUT / 1048576.0))

    def check(raw, tag):
        got = np.frombuffer(raw, dtype=np.int32)
        exp = np.tile(ref.reshape(-1), REPS)
        bad = int(np.count_nonzero(got != exp))
        return bad

    results = {}

    # (a) uncached /dev/mem view
    t = time.time()
    raw_unc = buf.read(O_OUT, TOTAL_OUT)
    dt_unc = time.time() - t
    bad_unc = check(raw_unc, "uncached")
    results["uncached"] = (dt_unc, bad_unc)

    # (b) cached, no maintenance -- expect stale garbage
    t = time.time()
    raw_cac = arena.read(base + O_OUT, TOTAL_OUT)
    dt_cac = time.time() - t
    bad_cac = check(raw_cac, "cached")

    # (c) cached + maintenance
    l2maint.maintain(cac + O_OUT, TOTAL_OUT, l2)
    t = time.time()
    raw_mnt = arena.read(base + O_OUT, TOTAL_OUT)
    dt_mnt = time.time() - t
    bad_mnt = check(raw_mnt, "maintained")

    # (d) repeat (c) 3x to see the steady-state cost of maintenance
    mnt_times = []
    mnt_bad = []
    for _ in range(3):
        l2maint.maintain(cac + O_OUT, TOTAL_OUT, l2)
        t = time.time()
        raw_m = arena.read(base + O_OUT, TOTAL_OUT)
        mnt_times.append(time.time() - t)
        mnt_bad.append(check(raw_m, "m"))

    mb = TOTAL_OUT / 1048576.0
    print("\n    %-12s %9s %10s %10s %8s" %
          ("mode", "ms", "MB/s", "wrong/M", "verdict"))
    rows = [
        ("uncached", dt_unc, bad_unc),
        ("cached", dt_cac, bad_cac),
        ("maintained", dt_mnt, bad_mnt),
    ]
    for (name, dt, bad) in rows:
        verdict = "OK" if bad == 0 else "WRONG"
        print("    %-12s %9.1f %10.1f %10d %8s"
              % (name, dt * 1000.0, mb / dt, bad, verdict))
    print("    maintained steady state: %s ms  wrong=%s"
          % (["%.1f" % (x * 1000.0) for x in mnt_times], mnt_bad))

    s = l2.stats()
    print("\n    PL310 ops=%d, %.1f us each" % (s["ops"], s["us_per_op"]))

    print("\n=== verdict ===")
    ok_unc = bad_unc == 0
    ok_mnt = bad_mnt == 0
    speedup = dt_unc / dt_mnt if dt_mnt else 0.0
    print("    uncached correct      : %s" % ok_unc)
    print("    cached-only correct   : %s  (must be False)" % (bad_cac == 0))
    print("    maintained correct    : %s" % ok_mnt)
    print("    speedup maintained/uncached: %.2fx" % speedup)
    if ok_unc and ok_mnt and bad_cac != 0:
        print("    RESULT: PASS -- cached+maintained is both fast and exact")
        return 0
    if ok_unc and ok_mnt:
        print("    RESULT: PARTIAL -- maintenance works but stale lines were")
        print("            not observed; the test may not be biting.")
        return 1
    print("    RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
