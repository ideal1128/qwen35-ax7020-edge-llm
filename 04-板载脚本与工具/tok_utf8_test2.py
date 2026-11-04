"""tok_utf8_test2.py -- hex-only diagnosis of the C tokenizer.

The previous run showed two surprising things:
  * every token's bytes ARE valid UTF-8 on their own (0 bad tokens), so
    per-token decoding is NOT the problem I suspected;
  * but the encode->decode round trip lost characters, and a 4-byte
    (non-BMP) input produced ZERO tokens.

This run prints only ASCII/hex so the serial console encoding cannot distort
the result.

Python 3.4.2 compatible.
"""

import ctypes
import os
import sys

BIN = "/mnt/qwen/tokenizer_64k.bin"
LIB = "/mnt/qwen/libtokenizer.so"


def hexs(s):
    return " ".join("%04X" % ord(c) for c in s)


def bhex(b):
    return " ".join("%02X" % x for x in b)


CASES = [
    ("cjk-common  U+4F60 U+597D", [0x4F60, 0x597D]),
    ("cjk-mid     U+4E16 U+754C", [0x4E16, 0x754C]),
    ("cjk-high    U+9F98 U+9F9C", [0x9F98, 0x9F9C]),
    ("cjk-high2   U+9FA0 U+9F8D", [0x9FA0, 0x9F8D]),
    ("cjk-ext-b   U+20000", [0x20000]),
    ("emoji       U+1F600", [0x1F600]),
    ("ascii       ABC", [0x41, 0x42, 0x43]),
    ("latin1      U+00E9", [0x00E9]),
    ("hangul      U+D55C", [0xD55C]),
]


def main():
    if not (os.path.exists(LIB) and os.path.exists(BIN)):
        print("!! missing lib or bin")
        return 2

    lib = ctypes.CDLL(LIB)
    lib.tokenizer_create.restype = ctypes.c_void_p
    lib.tokenizer_create.argtypes = [ctypes.c_char_p]
    lib.tokenizer_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                     ctypes.c_void_p, ctypes.c_int]
    lib.tokenizer_decode.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_char_p, ctypes.c_int]
    p = lib.tokenizer_create(BIN.encode("utf-8"))

    print("=== tok_utf8_test2 (hex only) ===")
    print("name | src_utf8 | n_tok | ids | dec_utf8 | roundtrip")

    for name, cps in CASES:
        src = "".join(chr(c) for c in cps)
        sb = src.encode("utf-8")
        arr = (ctypes.c_int * 1024)()
        n = lib.tokenizer_encode(p, sb, arr, 1024)
        n = max(0, n)
        ids = [arr[i] for i in range(n)]

        pieces = []
        for t in ids:
            buf = ctypes.create_string_buffer(256)
            lib.tokenizer_decode(p, t, buf, 256)
            pieces.append(buf.value)
        out = b"".join(pieces)

        print("\n%s" % name)
        print("    src_utf8 : %s" % bhex(sb))
        print("    n_tok    : %d   ids=%s" % (n, ids))
        print("    dec_utf8 : %s" % bhex(out))
        print("    match    : %s" % ("YES" if out == sb else "NO"))

    print("\n=== summary ===")
    print("    (round trip NO on a case means the encoder lost input)")

    # how many vocab entries decode to bytes that are NOT valid UTF-8 alone?
    bad = 0
    nonempty = 0
    for t in range(65536):
        buf = ctypes.create_string_buffer(256)
        lib.tokenizer_decode(p, t, buf, 256)
        b = buf.value
        if not b:
            continue
        nonempty += 1
        try:
            b.decode("utf-8")
        except UnicodeDecodeError:
            bad += 1
    print("    vocab entries with bytes      : %d / 65536" % nonempty)
    print("    of those, NOT valid UTF-8     : %d" % bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
