#!/usr/bin/env python3
"""Minimal /dev/mem peek/poke for Zynq bring-up.  Usage: devmem.py r ADDR [N] | w ADDR VALUE"""
import mmap, os, struct, sys

PAGE = 0x1000

def _map(addr, length=4):
    base = addr & ~(PAGE - 1)
    off = addr - base
    f = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    m = mmap.mmap(f, PAGE * 2, offset=base)
    return f, m, off

def rd(addr):
    f, m, off = _map(addr)
    v = struct.unpack('<I', m[off:off + 4])[0]
    m.close(); os.close(f)
    return v

def wr(addr, val):
    f, m, off = _map(addr)
    m[off:off + 4] = struct.pack('<I', val & 0xFFFFFFFF)
    m.close(); os.close(f)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    op = sys.argv[1]
    addr = int(sys.argv[2], 0)
    if op == 'r':
        n = int(sys.argv[3], 0) if len(sys.argv) > 3 else 1
        for i in range(n):
            print('0x%08X = 0x%08X' % (addr + 4 * i, rd(addr + 4 * i)))
    elif op == 'w':
        wr(addr, int(sys.argv[3], 0))
        print('wrote 0x%08X <- 0x%08X' % (addr, int(sys.argv[3], 0)))
    else:
        print(__doc__); sys.exit(1)
