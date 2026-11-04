"""tok_utf8_test.py -- is the per-token UTF-8 decode actually lossy?

Mechanism under test
--------------------
qwen_tokenizer.py does, for EVERY token id:

    buf = create_string_buffer(256)
    lib.tokenizer_decode(tok, tid, buf, 256)
    return buf.value.decode("utf-8", errors="replace")

and decode_batch() joins those already-decoded strings.  If a single token's
byte sequence is only PART of a multi-byte UTF-8 character, that per-token
decode cannot succeed and produces U+FFFD.

buf.value is bytes, so the raw bytes are NOT lost -- they are destroyed by
decoding too early.  This test compares:

    A) per-token decode then join   (what the code does today)
    B) join raw bytes then decode   (the proposed fix)

ASCII only in the source; the test strings themselves are built from code
points so this file stays pure ASCII.

Python 3.4.2 compatible.
"""

import ctypes
import os
import sys

BIN = "/mnt/qwen/tokenizer_64k.bin"
LIB = "/mnt/qwen/libtokenizer.so"

# code-point built so the file stays ASCII
CASES = [
    ("common han", [0x4F60, 0x597D, 0x4E16, 0x754C]),                  # 你好世界
    ("rare han", [0x9F98, 0x9F9C, 0x9FA0, 0x9F8D]),                    # rare glyphs
    ("mixed han+ascii", [0x4E2D, 0x6587, 0x41, 0x42, 0x43, 0x6D4B, 0x8BD5]),
    ("emoji", [0x1F600, 0x1F680, 0x1F4A1]),                            # >BMP, 4-byte
    ("punct", [0x3002, 0x3001, 0xFF0C, 0xFF1F, 0x201C, 0x201D]),
    ("mixed all", [0x4F60, 0x597D, 0x1F680, 0x9F98, 0x41, 0x3002]),
]


def main():
    if not os.path.exists(LIB):
        print("!! %s missing -- cannot test" % LIB)
        return 2
    if not os.path.exists(BIN):
        print("!! %s missing -- cannot test" % BIN)
        return 2

    lib = ctypes.CDLL(LIB)
    lib.tokenizer_create.restype = ctypes.c_void_p
    lib.tokenizer_create.argtypes = [ctypes.c_char_p]
    lib.tokenizer_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                     ctypes.c_void_p, ctypes.c_int]
    lib.tokenizer_decode.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_char_p, ctypes.c_int]
    lib.tokenizer_destroy.argtypes = [ctypes.c_void_p]

    p = lib.tokenizer_create(BIN.encode("utf-8"))
    if not p:
        print("!! tokenizer_create returned NULL")
        return 2
    print("=== tok_utf8_test ===")

    def enc(text):
        arr = (ctypes.c_int * 1024)()
        n = lib.tokenizer_encode(p, text.encode("utf-8"), arr, 1024)
        return [arr[i] for i in range(max(0, n))]

    def dec_bytes(tid):
        buf = ctypes.create_string_buffer(256)
        lib.tokenizer_decode(p, tid, buf, 256)
        return buf.value

    total_bad_tokens = 0
    total_tokens = 0
    lossy_cases = 0

    for name, cps in CASES:
        src = "".join(chr(c) for c in cps)
        ids = enc(src)
        if not ids:
            print("\n[%s] encode returned 0 ids -- skipping" % name)
            continue

        pieces = [dec_bytes(t) for t in ids]
        total_tokens += len(ids)

        # how many individual tokens are not valid UTF-8 on their own?
        bad = []
        for k, b in enumerate(pieces):
            try:
                b.decode("utf-8")
            except UnicodeDecodeError:
                bad.append(k)
        total_bad_tokens += len(bad)

        a = "".join(b.decode("utf-8", errors="replace") for b in pieces)
        bfix = b"".join(pieces).decode("utf-8", errors="replace")

        ok_a = (a == src)
        ok_b = (bfix == src)
        if not ok_a:
            lossy_cases += 1

        print("\n[%s] %d tokens, %d not valid UTF-8 alone"
              % (name, len(ids), len(bad)))
        print("    ids      : %s" % ids)
        print("    token len: %s" % [len(x) for x in pieces])
        print("    A per-tok: %r  %s" % (a, "OK" if ok_a else "LOSSY"))
        print("    B bytes  : %r  %s" % (bfix, "OK" if ok_b else "LOSSY"))
        if not ok_a:
            print("    A has %d replacement chars, B has %d"
                  % (a.count("\ufffd"), bfix.count("\ufffd")))

    print("\n=== summary ===")
    print("    tokens tested          : %d" % total_tokens)
    print("    tokens not valid alone : %d" % total_bad_tokens)
    print("    cases where A is lossy : %d / %d" % (lossy_cases, len(CASES)))

    if lossy_cases == 0:
        print("\n    VERDICT: the per-token decode is NOT lossy on this vocab.")
        print("    My concern was wrong -- Chinese output is fine as-is.")
    else:
        print("\n    VERDICT: the per-token decode IS lossy. Accumulate the raw")
        print("    bytes and decode once (fix B) instead.")

    lib.tokenizer_destroy(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
