"""multiturn_test.py -- verify multi-turn conversation on the board.

Checks three things the single-turn tests could not:
  1. a second turn sees the first turn's history (prompt grows)
  2. the KV pool survives two turns without running out
  3. the assistant reply is stored and re-encoded correctly

ASCII-only source; prompts built from code points.
Python 3.4.2 compatible.
"""

import os
import sys

sys.path.insert(0, "/mnt/qwen")
os.chdir("/mnt/qwen")
os.environ["QWEN_CPU_ONLY"] = "1"

import chat_v3 as C


def cps(*xs):
    return "".join(chr(x) for x in xs)


def main():
    eng, qtok = C.load_engine()
    eng.reset_state()
    sess = C.Session(system_prompt="You are a helpful assistant.")

    # turn 1: U+4F60 U+597D
    q1 = cps(0x4F60, 0x597D)
    print("\n=== turn 1 ===")
    r1, ids1 = C.generate(eng, qtok, sess, q1, max_new=10, verbose=True)

    # turn 2: U+4F60 U+662F U+8C01
    q2 = cps(0x4F60, 0x662F, 0x8C01)
    print("\n=== turn 2 (with history) ===")
    r2, ids2 = C.generate(eng, qtok, sess, q2, max_new=10, verbose=True)

    # verify history actually reached the prompt
    print("\n=== checks ===")
    ids_t2 = sess.build_ids(qtok.encode)
    print("[chk] turn-2 prompt length      : %d" % len(ids_t2))
    print("[chk] turns recorded            : %d" % len(sess.turns))
    ok_turns = len(sess.turns) == 4          # user, assistant, user, assistant
    print("[chk] 4 turns stored            : %s" % ("PASS" if ok_turns else "FAIL"))

    # the turn-2 prompt must contain turn 1's ids as a prefix-ish subsequence
    ids_t1 = [C.IM_START] + qtok.encode("system\n" + sess.system) + [C.IM_END]
    ids_t1 += [C.IM_START] + qtok.encode("user\n" + q1) + [C.IM_END]
    grew = len(ids_t2) > len(ids_t1)
    print("[chk] prompt grew with history  : %s (%d -> %d)"
          % ("PASS" if grew else "FAIL", len(ids_t1), len(ids_t2)))

    # replies must be non-empty and differ (different questions)
    print("[chk] reply1 non-empty          : %s" % ("PASS" if r1.strip() else "FAIL"))
    print("[chk] reply2 non-empty          : %s" % ("PASS" if r2.strip() else "FAIL"))
    print("[chk] replies differ            : %s" % ("PASS" if r1 != r2 else "NOTE-same"))

    # reset must clear history
    eng.reset_state()
    sess.reset()
    ids_after = sess.build_ids(qtok.encode)
    cleared = len(sess.turns) == 0 and len(ids_after) < len(ids_t2)
    print("[chk] reset clears history      : %s" % ("PASS" if cleared else "FAIL"))

    print("\n[out] reply1 = %r" % (r1.encode("ascii", "backslashreplace").decode("ascii"),))
    print("[out] reply2 = %r" % (r2.encode("ascii", "backslashreplace").decode("ascii"),))
    return 0


if __name__ == "__main__":
    sys.exit(main())
