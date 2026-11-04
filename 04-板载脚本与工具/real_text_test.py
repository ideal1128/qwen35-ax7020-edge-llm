"""real_text_test.py -- can the REAL engine produce readable text?

Every verification so far printed token ids only.  The deployed /mnt/qwen/real_chat
points at a 6 KB stub chat_board.py whose "decode" is hardcoded to
"token_<id>" and whose LM head only searches the first 1024 embeddings, so the
board has never actually shown readable output.

This drives the real QwenRealEngine and decodes with the real tokenizer, which
we separately proved safe: all 65,535 non-empty vocab entries decode to valid
UTF-8 on their own.

ASCII-only source; the prompt is built from code points.
Python 3.4.2 compatible.
"""

import os
import sys
import time

sys.path.insert(0, "/mnt/qwen")
os.chdir("/mnt/qwen")
os.environ["QWEN_CPU_ONLY"] = "1"


class CpuOnlyHw(object):
    def __init__(self, *a, **k):
        self.mock = True

    def reset_state(self, *a, **k):
        pass

    def close(self, *a, **k):
        pass


import qwen_real_engine as E
E.FastBoardHw = CpuOnlyHw
from qwen_real_engine import QwenRealEngine

N = int(os.environ.get("N_TOK", "12"))
# U+4F60 U+597D U+FF0C U+4ECB U+7ECD U+4E00 U+4E0B U+4F60 U+81EA U+5DF1 U+3002
PROMPT = os.environ.get("PROMPT") or "".join(
    chr(c) for c in [0x4F60, 0x597D, 0xFF0C, 0x4ECB, 0x7ECD,
                     0x4E00, 0x4E0B, 0x4F60, 0x81EA, 0x5DF1, 0x3002])


def main():
    t0 = time.time()
    eng = QwenRealEngine()
    print("[T] init = %.3f s" % (time.time() - t0))
    tok = eng.tok

    # 1. do the ChatML control tokens survive the C tokenizer?
    print("\n[T] --- control token check ---")
    for name, want in (("<|im_start|>", 65504), ("<|im_end|>", 65505)):
        try:
            got = tok.encode(name)
        except Exception as ex:
            got = "EXC %r" % (ex,)
        print("    %-14s -> %s   (expect [%d])" % (name, got, want))

    # 2. plain prompt round trip
    ids = tok.encode(PROMPT)
    back = "".join(tok.decode(i) for i in ids)
    print("\n[T] prompt ids   = %s" % ids)
    print("    prompt bytes = %s" % " ".join("%02X" % b for b in PROMPT.encode("utf-8")))
    print("    roundtrip    = %s" % ("OK" if back == PROMPT else "LOSSY: %r" % (back,)))

    # 3. generate
    print("\n[T] --- generating %d tokens (first one includes weight staging) ---" % N)
    cur = ids[-1]
    t = time.time()
    cur = eng.forward_token(cur)
    print("[T] warmup %.1f ms -> id %d" % ((time.time() - t) * 1000.0, cur))

    out_ids = [cur]
    times = []
    for i in range(N - 1):
        t = time.time()
        cur = eng.forward_token(cur)
        times.append(time.time() - t)
        out_ids.append(cur)

    # 4. decode
    text = "".join(tok.decode(i) for i in out_ids)
    print("\n[T] out ids  = %s" % out_ids)
    print("[T] out text = %r" % (text,))
    print("[T] out utf8 = %s" % " ".join("%02X" % b for b in text.encode("utf-8")))
    if times:
        avg = sum(times) / float(len(times))
        print("[T] avg = %.1f ms  -> %.4f tok/s" % (avg * 1000.0, 1.0 / avg))

    # 5. is it degenerate?
    print("\n[T] --- degeneracy ---")
    print("    unique ids : %d / %d" % (len(set(out_ids)), len(out_ids)))
    print("    readable   : %s" % ("YES" if text.strip() else "NO"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
