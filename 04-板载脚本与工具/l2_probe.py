"""l2_probe.py -- safe staged measurement of the readback question.

Reuses pl_dma_test.Arena (32 MB, mlock'd, pages pre-faulted, pagemap-checked)
instead of hand-rolling physical addressing: the earlier hand-rolled 64 MB
unlocked arena let the kernel migrate pages, the DMA wrote outside the
intended range and hard-hung the board.

STAGE 1: cached vs uncached read bandwidth, measured with a real memcpy so
         Python allocation overhead does not dominate.
STAGE 2: 4 KB fabric transfer -- is the result visible in the cached mapping
         without maintenance (expected NO) and with maintenance (expected YES)?

ASCII only, Python 3.4.2 compatible.
"""

import ctypes
import os
import struct
import sys
import time

sys.path.insert(0, "/mnt/qwen")
os.chdir("/mnt/qwen")

import l2maint
import pl_dma_test as P

PAGE = 4096
_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def mb(n):
    return n / 1048576.0


def bench_memcpy(dst, src_addr, n, reps, l2, flush_between):
    """Time a raw memcpy out of a mapping, so we measure memory not Python."""
    t = time.time()
    for _ in range(reps):
        if flush_between:
            # evict the destination so every rep re-reads the source from DDR
            l2.clean_invalidate_all()
        _libc.memcpy(dst, ctypes.c_void_p(src_addr), ctypes.c_size_t(n))
    dt = time.time() - t
    return dt, mb(n * reps) / dt


def main():
    print("=== l2_probe ===")
    l2 = l2maint.probe()

    arena = P.Arena(32)
    WIN = 1024 * 1024
    base = arena.find_run(256)
    if base is None:
        print("!! find_run(256) failed")
        return 2
    phys = arena.phys_at(base)
    cac_addr = arena.addr + base
    print("    window 1 MB: voff %d  phys 0x%08x" % (base, phys))

    # uncached alias of exactly that window
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    mm = mmap_new(fd, WIN, phys)
    unc_view = (ctypes.c_char * WIN).from_buffer(mm)
    unc_addr = ctypes.addressof(unc_view)

    # ---- verify the alias really is these pages before trusting it.
    # The write goes through the CACHED mapping, so it must be cleaned out to
    # DDR first -- otherwise the uncached /dev/mem read legitimately sees the
    # old content and we would mistake cache incoherence for a bad address.
    arena.write(base, b"ALIAS-CHECK-0001")
    l2maint.maintain(cac_addr, 16, l2)
    if bytes(mm[0:16]) != b"ALIAS-CHECK-0001":
        print("!! /dev/mem alias does NOT match the anon pages -- aborting")
        print("   read back: %r" % (bytes(mm[0:16]),))
        return 3
    print("    /dev/mem alias verified")

    dst = ctypes.create_string_buffer(WIN)

    print("\n=== stage 1: read bandwidth (1 MB, raw memcpy) ===")
    dt, bw = bench_memcpy(dst, cac_addr, WIN, 8, l2, False)
    print("    cached, no flush  : %7.1f ms  %8.1f MB/s" % (dt * 1000.0, bw))
    dt2, bw2 = bench_memcpy(dst, cac_addr, WIN, 8, l2, True)
    print("    cached, L2 flushed: %7.1f ms  %8.1f MB/s" % (dt2 * 1000.0, bw2))
    dt3, bw3 = bench_memcpy(dst, unc_addr, WIN, 8, l2, False)
    print("    uncached /dev/mem : %7.1f ms  %8.1f MB/s" % (dt3 * 1000.0, bw3))
    if bw3:
        print("    speedup maintained/uncached: %.2fx" % (bw / bw3))

    # ---- stage 2: tiny GEMV, coherence
    print("\n=== stage 2: 4 KB fabric transfer, cached visibility ===")
    dma = P.Dma()
    dma.dump("power-on")

    K, M = 64, 1
    nb = K // 64
    x_beats = K // 8
    w_bytes = M * K
    out_bytes = M * nb * 4

    o_hdr, o_x = 0, 16
    o_w = o_x + x_beats * 16
    o_out = o_w + w_bytes
    need = o_out + out_bytes
    print("    layout: need %d B (hdr %d, x %d, w %d, out %d)"
          % (need, o_hdr, o_x, o_w, o_out))

    x = [((i * 977) % 60001) - 30000 for i in range(K)]
    w = [((i * 37) % 255) - 127 for i in range(M * K)]

    hdr0 = (x_beats & 0xFFFF) | (((nb * 4 - 1) & 0xFF) << 16)
    arena.write(base + o_hdr, struct.pack("<I", hdr0) + b"\x00" * 12)
    xb = bytearray()
    for v in x:
        xb += struct.pack("<h", v)
    arena.write(base + o_x, bytes(xb))
    arena.write(base + o_w, bytes(bytearray((v & 0xFF) for v in w)))

    ref = 0
    for j in range(64):
        ref += w[j] * x[j]

    # poison the destination then flush, so a stale read is unmistakable
    arena.write(base + o_out, b"\x5a" * out_bytes)
    l2maint.maintain(cac_addr, need, l2)

    print("    GEMV K=64 M=1, expect %d" % ref)
    dma.reset()
    dma.s2mm_start(phys + o_out, out_bytes)
    dma.mm2s(phys + o_hdr, 16)
    dma.mm2s(phys + o_x, x_beats * 16)
    dma.mm2s(phys + o_w, w_bytes)
    dma.s2mm_wait()

    got_unc = struct.unpack("<i", bytes(mm[o_out:o_out + 4]))[0]
    got_stale = struct.unpack("<i", bytes(arena.read(base + o_out, 4)))[0]
    l2maint.maintain(cac_addr + o_out, out_bytes, l2)
    got_mnt = struct.unpack("<i", bytes(arena.read(base + o_out, 4)))[0]

    print("    expected      : %d" % ref)
    print("    uncached      : %d  %s" % (got_unc, "OK" if got_unc == ref else "WRONG"))
    print("    cached stale  : %d  %s"
          % (got_stale, "OK -- no staleness observed" if got_stale == ref else "STALE (as predicted)"))
    print("    cached + maint: %d  %s" % (got_mnt, "OK" if got_mnt == ref else "WRONG"))

    s = l2.stats()
    print("\n    PL310 ops=%d, %.1f us each" % (s["ops"], s["us_per_op"]))

    print("\n=== summary ===")
    print("    uncached correct     : %s" % (got_unc == ref))
    print("    maintained correct   : %s" % (got_mnt == ref))
    print("    stale-without-maint  : %s (True means the test is biting)"
          % (got_stale != ref))
    return 0 if (got_unc == ref and got_mnt == ref) else 1


def mmap_new(fd, n, phys):
    import mmap
    m = mmap.mmap(fd, n, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=phys)
    return m


if __name__ == "__main__":
    sys.exit(main())
