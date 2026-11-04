"""l2maint.py -- PL310 L2 (and L1) cache maintenance for the Zynq-7020.

ASCII only, Python 3.4.2 compatible.

THE BUG THIS FIXES
------------------
The fabric writes the GEMV block dots into DDR over S_AXI_HP.  That path is
NOT coherent with the ARM caches, so after the DMA completes the CPU has to
discard any cached copy of the destination before reading it.

The previous attempt used the Linux cacheflush syscall (__ARM_NR_cacheflush,
0x0f0002) and got fast, wrong results.  On ARMv7 that syscall only maintains
L1.  The PL310 L2 sits between the core and DDR and keeps serving the stale
lines, so the CPU reads the previous contents of the buffer.  It looks right
on the first call and wrong afterwards, which is how it survived one test.

THE FIX
-------
The PL310 is MMIO at 0xF8F02000.  Two registers do the whole job:

  0x77C  Clean and Invalidate by Way  write a way mask, 0xFF = all 8 ways
  0x730  Cache Sync                   bit0 = 1 while the operation drains

0x77C writes dirty lines back to DDR and then invalidates them, so it is safe
for memory the CPU owns: it can only make the cache colder.  This is the
operation the Xilinx standalone BSP performs
(XPS_L2CC_CACHE_INV_CLN_BY_WAY_OFFSET = 0x77C, way mask 0xFF).

The way mask is NOT "any value".  0x77C is a bit-per-way mask on the PL310;
writing 0 would touch no ways and writing 1 would clean only way 0.  0xFF
covers all 8 ways of the 512 KB / 8-way part in the Zynq-7020.

ORDERING MATTERS IN BOTH DIRECTIONS (this is the part that is easy to get
wrong, and getting it wrong is exactly how the destination test fails):

maintain() must be called BEFORE the DMA as well as after it.  The pre-DMA
call is not an optimisation, it is what makes the post-DMA call correct.

  * before the DMA it writes back anything the CPU left dirty in the
    destination range (for example a zero-fill) and then invalidates it, so
    DDR and the CPU's view agree;
  * after the DMA the same clean+invalidate finds nothing dirty in the
    destination, so its clean phase cannot scribble the CPU's stale zeros
    over the fabric's output, and the invalidate phase forces the CPU to
    re-read the real data from DDR.

Skip the pre-DMA call and the post-DMA clean phase writes the CPU's zeros
back over the block sums -- a clean+invalidate of a dirty destination is a
silent corruption.  pl_l2_check.py tests exactly this case.

L1 must be invalidated as well.  A PL310 operation does not touch the core's
L1 D-cache, and a stale L1 line still hits after L2 has been fixed, so
maintain() also issues cacheflush for the range (clean+invalidate on L1).

USAGE
-----
    import l2maint
    l2 = l2maint.L2()

    l2maint.maintain(va, nbytes, l2)   # before the DMA
    dma.s2mm_start(phys, nbytes); ...
    dma.s2mm_wait()
    l2maint.maintain(va, nbytes, l2)   # after the DMA, before the CPU reads

`va` is the ORDINARY CACHED virtual address of the buffer (the anonymous
mmap), not the /dev/mem view.
"""

import ctypes
import mmap
import os
import struct
import time

PL310_BASE = 0xF8F02000

REG_CACHE_ID = 0x000          # cache ID, informational
REG_CACHE_TYPE = 0x004        # cache type, informational
REG_CACHE_SYNC = 0x730        # bit0 = 1 while a maintenance op is draining
REG_INV_CLN_BY_WAY = 0x77C    # clean+invalidate all ways, arg is a way mask

WAY_MASK_ALL = 0xFF           # Zynq-7020 PL310: 512 KB, 8-way

# __ARM_NR_cacheflush: clean+invalidate the L1 D-cache for a user range.
_ARM_NR_CACHEFLUSH = 0x0f0002
_libc = ctypes.CDLL('libc.so.6', use_errno=True)
_libc.syscall.restype = ctypes.c_long


def cacheflush(addr, nbytes):
    """L1 clean+invalidate for [addr, addr+nbytes).  Returns 0 on success.

    This is the ONLY thing the previous attempt did, and on its own it is not
    enough: it leaves the PL310 L2 stale.
    """
    try:
        return _libc.syscall(ctypes.c_long(_ARM_NR_CACHEFLUSH),
                             ctypes.c_ulong(addr),
                             ctypes.c_ulong(addr + nbytes),
                             ctypes.c_ulong(0))
    except Exception:
        return -1


class L2(object):
    """PL310 maintenance operations.  One instance per process is enough."""

    def __init__(self, base=PL310_BASE, verbose=False):
        self.fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        self.regs = mmap.mmap(self.fd, 0x1000, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.ops = 0
        self.seconds = 0.0
        if verbose:
            print('l2: PL310 @0x%08x id=0x%08x type=0x%08x'
                  % (base, self.rd(REG_CACHE_ID), self.rd(REG_CACHE_TYPE)))

    # ---------------------------------------------------------- registers
    def rd(self, off):
        return struct.unpack('<I', self.regs[off:off + 4])[0]

    def wr(self, off, val):
        self.regs[off:off + 4] = struct.pack('<I', val)

    def sync(self, timeout=1.0):
        """Wait for the maintenance operation to drain.  True if it did."""
        t0 = time.time()
        while self.rd(REG_CACHE_SYNC) & 1:
            if time.time() - t0 > timeout:
                return False
        return True

    # ---------------------------------------------------------- operations
    def clean_invalidate_all(self):
        """Clean+invalidate the entire L2.  Safe: dirty lines go to DDR first.

        Returns True if the operation drained within the timeout.
        """
        t0 = time.time()
        self.wr(REG_INV_CLN_BY_WAY, WAY_MASK_ALL)
        ok = self.sync()
        # One more read of the sync register orders the maintenance behind the
        # strongly-ordered MMIO write before the caller issues its loads.
        self.rd(REG_CACHE_SYNC)
        dt = time.time() - t0
        self.ops += 1
        self.seconds += dt
        return ok

    # alias: the clean+invalidate form is the only one that is safe globally,
    # so "clean" and "invalidate" both map to it here.
    clean_all = clean_invalidate_all
    invalidate_all = clean_invalidate_all

    def stats(self):
        return {'ops': self.ops, 'seconds': self.seconds,
                'us_per_op': (self.seconds / self.ops * 1e6) if self.ops else 0.0}


def maintain(va, nbytes, l2=None):
    """Make `nbytes` at cached VA `va` coherent with the fabric.

    Call this BEFORE an MM2S that sources the range (so the PL sees the CPU's
    writes) and AFTER an S2MM that fills it (so the CPU sees the PL's writes).
    Both calls are needed for the destination case; see the module docstring.

    Returns True on success.
    """
    if l2 is None:
        l2 = _default_l2()
    ok_l1 = (cacheflush(int(va), int(nbytes)) == 0)
    ok_l2 = l2.clean_invalidate_all()
    return ok_l1 and ok_l2


_default = [None]


def _default_l2():
    if _default[0] is None:
        _default[0] = L2()
    return _default[0]


def probe(verbose=True):
    """Print the PL310 identity so a log shows which controller was hit."""
    l2 = L2()
    cid = l2.rd(REG_CACHE_ID)
    ctype = l2.rd(REG_CACHE_TYPE)
    if verbose:
        print('l2: PL310 @0x%08x  cache id=0x%08x  cache type=0x%08x'
              % (PL310_BASE, cid, ctype))
        print('l2: using way mask 0x%02x (all 8 ways of the 512 KB part)'
              % WAY_MASK_ALL)
    return l2
