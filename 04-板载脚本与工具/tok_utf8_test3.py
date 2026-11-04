"""tok_utf8_test3.py -- confirm that encode() silently DROPS characters.

Evidence so far:
  * every one of the 65,535 non-empty vocab entries decodes to valid UTF-8
    bytes on its own, so per-token decoding cannot be lossy -> output is fine;
  * but U+1F600, U+20000, U+D55C encode to ZERO tokens;
  * U+9F98 U+9F9C encode to ONE token that decodes to only U+9F9C.

If encode() really drops what it cannot represent, then sandwiching an
unsupported character between ASCII letters must still round-trip to just the
letters.  That is what this checks.

ASCII/hex output only.  Python 3.4.2 compatible.
"""

import ctypes
import os
import sys

BIN = "/mnt/qwen/tokenizer_64k.bin"
LIB = "/mnt/qwen/libtokenizer.so"


def bhex(b):
    return " ".join("%02X" % x for x in b)


def main():
    lib = ctypes.CDLL(LIB)
    lib.tokenizer_create.restype = ctypes.c_void_p
    lib.tokenizer_create.argtypes = [ctypes.c_char_p]
    lib.tokenizer_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                     ctypes.c_void_p, ctypes.c_int]
    lib.tokenizer_decode.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_char_p, ctypes.c_int]
    p = lib.tokenizer_create(BIN.encode("utf-8"))

    def rt(name, cps):
        src = "".join(chr(c) for c in cps)
        sb = src.encode("utf-8")
        arr = (ctypes.c_int * 1024)()
        n = max(0, lib.tokenizer_encode(p, sb, arr, 1024))
        out = b""
        for i in range(n):
            buf = ctypes.create_string_buffer(256)
            lib.tokenizer_decode(p, arr[i], buf, 256)
            out += buf.value
        print("\n%s" % name)
        print("    in  : %s" % bhex(sb))
        print("    n   : %d  ids=%s" % (n, [arr[i] for i in range(n)]))
        print("    out : %s" % bhex(out))
        print("    lost: %d bytes  %s" % (len(sb) - len(out),
                                          "DROP CONFIRMED" if len(out) < len(sb) else "complete"))

    print("=== tok_utf8_test3 : does encode() drop unsupported characters? ===")
    # A <unsupported> B  -- if B survives and the middle is gone, it dropped
    rt("A + U+9F98 + B", [0x41, 0x9F98, 0x42])
    rt("A + U+1F600 + B", [0x41, 0x1F600, 0x42])
    rt("A + U+D55C + B", [0x41, 0xD55C, 0x42])
    # supported text around an unsupported one
    rt("ni-hao + emoji + shi-jie", [0x4F60, 0x597D, 0x1F600, 0x4E16, 0x754C])
    # all supported
    rt("ni-hao + shi-jie (control)", [0x4F60, 0x597D, 0x4E16, 0x754C])
    return 0


if __name__ == "__main__":
    sys.exit(main())
