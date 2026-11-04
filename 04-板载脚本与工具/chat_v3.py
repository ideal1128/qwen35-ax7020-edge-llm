"""chat_v3.py -- real interactive chat front-end for Qwen3.5-0.8B on AX7020.

Why this file exists
--------------------
The board's /mnt/qwen/real_chat launches a 6 KB stub chat_board.py that
slices hidden[:64], searches only the first 1024 embeddings for the LM head,
and hardcodes its "decode" to the string "token_<id>".  It never produced
readable text.  This is the replacement.

What it does differently
------------------------
1. Uses the REAL engine (qwen_real_engine.QwenRealEngine), 24 layers, all
   65536 vocab entries, NEON kernels.
2. Uses qwen_tokenizer.CTokenizer, which regex-isolates the ChatML control
   tokens and maps them to their reserved ids.  The plain C tokenizer cannot
   do this: encode("<|im_start|>") returns the six byte-tokens
   [27, 91, 315, 4732, 91, 29] instead of [65504].
3. Builds a proper Qwen3 ChatML prompt so the model knows it is answering a
   turn instead of continuing raw text.
4. Decodes with the real tokenizer, so output is readable UTF-8.
5. Keeps multi-turn history and trims it to fit the 2048-token context.

Usage
-----
    python3 chat_v3.py "your question"      one-shot
    python3 chat_v3.py                      interactive REPL
    python3 chat_v3.py --max 32 "..."       explicit cap on generated tokens
    python3 chat_v3.py --greedy "..."       turn sampling off (default is on)
    python3 chat_v3.py --sample 0.7,40,0.9  temperature,top_k,top_p
    python3 chat_v3.py --system "..." ...   custom system prompt
    python3 chat_v3.py --think              enable the model's reasoning block
    python3 chat_v3.py --think-hide         run reasoning but do not print it
    python3 chat_v3.py --ctx 8192           resize the KV pool (default 4096)

Interactive commands
--------------------
    /clear  /new  /reset    drop the conversation and the KV cache
    /think                  show whether reasoning is on
    /think on | off         enable / disable the reasoning block
    /think show | hide      print / suppress the reasoning text
    /stats                  turns, tokens in KV, think state
    /help                   this list
    /quit  /exit  /q        leave (also Ctrl-D)

Decoding defaults
-----------------
There is NO hard token cap by default: generation stops when the model emits
<|im_end|>, bounded only by what the KV pool can still hold.  Sampling is ON
by default (T=0.7, top_k=40, top_p=0.9) because greedy decoding was measured
repeating itself (11 duplicate ids out of 48).

Reasoning is OFF by default.  With --think the prompt ends exactly like the
upstream chat_template.jinja generation prompt for enable_thinking=true, i.e.
`<|im_start|>assistant\n thinking\n`, and the model streams its reasoning
until it emits  (65528) and then answers.  Expect 1.5-4 minutes for a
thinking reply on this board: every reasoning token costs a full forward
pass, 670-860 ms.

Python 3.4.2 compatible: no f-strings, no subprocess.run, no type hints.
Pure ASCII source.
"""

import codecs
import io
import os
import sys
import time

sys.path.insert(0, "/mnt/qwen")
try:
    os.chdir("/mnt/qwen")
except OSError:
    pass

# Force CPU path: the PL bitstream on the board contains a gemv core that was
# measured to be non-deterministic, so it must not be used for inference.
os.environ["QWEN_CPU_ONLY"] = "1"

IM_START = 65504   # <|im_start|>
IM_END = 65505     # <|im_end|>
THINK_OPEN = 65527   #  thinking
THINK_CLOSE = 65528  # </think>

# Default KV-pool size in tokens.  qwen_real_engine sizes the pool from its
# own PAGED_NUM_BLOCKS; load_engine() overrides that from --ctx and the
# Session then reads the REAL capacity back off the engine, so this constant
# is only the fallback.
DEFAULT_CTX = 4096
MAX_CTX = DEFAULT_CTX   # kept for anything that still imports the old name


# ---------------------------------------------------------------------------
# stdout: the serial console may not be UTF-8.  Re-wrap it so Chinese output
# cannot raise UnicodeEncodeError mid-generation.
#
# NOTE: do NOT use codecs.getwriter() here.  That wrapper forwards unknown
# attributes to the underlying binary stream, so sys.stdout.encoding becomes
# FileIO.encoding -- which does not exist -- and Python 3.4's input() raises
#     AttributeError: '_io.FileIO' object has no attribute 'encoding'
# as soon as the interactive prompt is printed.  io.TextIOWrapper is a real
# text stream and exposes .encoding properly.
# ---------------------------------------------------------------------------
_ORIG_STDOUT = sys.stdout     # keep a reference: if the old wrapper is
                              # collected it would close the shared buffer


def _utf8_stdout():
    global _ORIG_STDOUT
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                          encoding="utf-8",
                                          errors="replace",
                                          line_buffering=True)
    except Exception:
        sys.stdout = _ORIG_STDOUT


class CpuOnlyHw(object):
    """Stand-in for the FPGA hardware class; every call is a no-op."""

    def __init__(self, *a, **k):
        self.mock = True

    def reset_state(self, *a, **k):
        pass

    def linear_attn_step(self, *a, **k):
        raise RuntimeError("PL path disabled")

    def close(self, *a, **k):
        pass


# ---------------------------------------------------------------------------
# Optional sampling.
#
# The engine's forward_token() always returns the argmax, computed inside the
# C kernel.  To sample we re-scan the 128 MiB bf16 embedding table from the
# final normed hidden state that forward_token() leaves in eng._final_ptr.
# That second scan costs roughly one lm_head pass (~115 ms), so sampling is
# off by default and greedy stays the fast path.
# ---------------------------------------------------------------------------
class Sampler(object):
    LIB = "/mnt/qwen/libqwen_sampling.so"

    def __init__(self, temperature=0.7, top_k=40, top_p=0.9, k=64):
        import ctypes
        self.ctypes = ctypes
        self.lib = ctypes.CDLL(self.LIB)
        self.lib.lm_head_topk.restype = ctypes.c_int
        self.lib.lm_head_topk.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float)]
        self.lib.sample_from_topk.restype = ctypes.c_int
        self.lib.sample_from_topk.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float,
            ctypes.POINTER(ctypes.c_uint)]
        self.k = int(k)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.ids = (ctypes.c_int * self.k)()
        self.scores = (ctypes.c_float * self.k)()
        # session-level RNG state; must persist across tokens
        self.rng = ctypes.c_uint(0x9E3779B9)

    def pick(self, eng):
        """Re-scan the LM head and sample one id from the current state.

        IMPORTANT: forward_token_c() runs the final RMSNorm inside C and does
        NOT write it back to eng._final_ptr, so reading _final_ptr directly
        gives stale data from the warmup (verified: produced incoherent
        output).  Recompute the norm into _final_ptr first -- it is cheap
        (~0.7 ms) compared with the 115 ms top-k scan.
        """
        eng._k_rmsnorm(eng._final_ptr, eng._h_ptr, eng._final_w_ptr, 1024, 1e-6)
        n = self.lib.lm_head_topk(eng._final_ptr, eng._embed_ptr, 65536, 1024,
                                  self.k, self.ids, self.scores)
        if n <= 0:
            return -1
        return self.lib.sample_from_topk(self.ids, self.scores, n,
                                         self.temperature, self.top_k,
                                         self.top_p,
                                         self.ctypes.byref(self.rng))


def load_engine(ctx=DEFAULT_CTX):
    import qwen_real_engine as E
    E.FastBoardHw = CpuOnlyHw

    # Size the KV pool and the RoPE tables before the engine is built.  Both
    # must cover ctx: the C attention path computes cos_table + pos*128 with no
    # bounds check, so a short table is silent memory corruption, not an error.
    ctx = int(ctx)
    if ctx < 256:
        ctx = 256
    blocks = (ctx + E.PAGED_BLOCK_SIZE - 1) // E.PAGED_BLOCK_SIZE
    E.PAGED_NUM_BLOCKS = blocks
    E.ROPE_MAX_SEQ = blocks * E.PAGED_BLOCK_SIZE
    print("[ctx] requesting %d tokens -> %d blocks, rope table %d"
          % (ctx, blocks, E.ROPE_MAX_SEQ))

    from qwen_real_engine import QwenRealEngine

    t = time.time()
    eng = QwenRealEngine()
    print("[init] engine ready in %.3f s" % (time.time() - t))

    # swap in the special-token aware tokenizer
    import qwen_tokenizer as T
    qtok = T.CTokenizer()
    if not qtok.tok_ptr:
        raise RuntimeError("qwen_tokenizer could not load libtokenizer.so "
                           "or tokenizer_64k.bin")
    eng.tok = qtok

    # sanity: the control tokens must now resolve to their reserved ids
    probe = qtok.encode("<|im_start|>")
    if probe != [IM_START]:
        print("[warn] <|im_start|> encodes to %s, expected [%d]" % (probe, IM_START))
    probe2 = qtok.encode("<|im_end|>")
    if probe2 != [IM_END]:
        print("[warn] <|im_end|> encodes to %s, expected [%d]" % (probe2, IM_END))
    return eng, qtok


# ---------------------------------------------------------------------------
# ChatML assembly
# ---------------------------------------------------------------------------
class Session(object):
    """Conversation state.

    The engine keeps its KV cache between calls, so history must NOT be
    re-fed.  Re-feeding was the original behaviour and it was both slow
    (turn 2 spent 26 s re-prefilling 38 tokens) and wrong (the duplicated
    tokens landed in KV a second time).  We now track how many tokens are
    already in KV and feed only the delta.
    """

    def __init__(self, system_prompt=None, max_ctx=DEFAULT_CTX, reserve=256,
                 think=False, show_think=True):
        self.system = system_prompt or "You are a helpful assistant."
        self.turns = []          # list of (role, text), for /stats only
        self.max_ctx = int(max_ctx)
        self.reserve = int(reserve)
        self.think = bool(think)
        self.show_think = bool(show_think)
        self.started = False     # has the system prompt been fed yet?
        self.fed = 0             # tokens already in the engine KV

    def reset(self):
        self.turns = []
        self.started = False
        self.fed = 0

    def add(self, role, text):
        self.turns.append((role, text))

    def _assistant_header(self, enc):
        """Reproduce chat_template.jinja lines 147-154 EXACTLY.

            {%- if add_generation_prompt %}
                {{- '<|im_start|>assistant\\n' }}
                {%- if enable_thinking is defined and enable_thinking is true %}
                    {{- ' thinking\\n' }}
                {%- else %}
                    {{- ' thinking\\n\\n</think>\\n\\n' }}
                {%- endif %}
            {%- endif %}

        The previous version emitted `assistant\\n` + `` + `\\n\\n` for the
        no-think case: it opened the block with  but never emitted the
        opening ` thinking\\n\\n`.  That is off-distribution -- the model saw a
        close marker for a block that was never opened -- and it also left the
        generation loop with no  handling at all.
        """
        ids = [IM_START]
        ids.extend(enc("assistant\n"))
        ids.append(THINK_OPEN)
        if self.think:
            ids.extend(enc("\n"))          # ' thinking\n'
        else:
            ids.extend(enc("\n\n"))        # ' thinking\n\n'
            ids.append(THINK_CLOSE)        # '</think>'
            ids.extend(enc("\n\n"))        # '\n\n'
        return ids

    def turn_ids(self, enc, user_text):
        """Return ONLY the tokens that still need to be fed for this turn.

        First turn: system prompt + user turn + assistant header.
        Later turns: the <|im_end|> that closed the previous reply (we break
        out of the generation loop before feeding it), then the new user
        turn, then the assistant header.
        """
        if not self.started:
            ids = [IM_START]
            ids.extend(enc("system\n" + self.system))
            ids.append(IM_END)
        else:
            ids = [IM_END]
        ids.append(IM_START)
        ids.extend(enc("user\n" + user_text))
        ids.append(IM_END)
        ids.extend(self._assistant_header(enc))
        return ids

    def would_overflow(self, delta_len, reserve_reply):
        """True when this turn does not fit in the KV pool.

        `reserve_reply` is how many tokens we insist on keeping free for the
        answer.  The old caller passed the *budget* here and the budget had
        already been clamped to leave that room, so the condition was
        identically false and this guard never fired -- which is why the pool
        used to fail with an IndexError inside allocate_slot() instead of
        resetting cleanly.
        """
        return (self.fed + delta_len + reserve_reply) > (self.max_ctx - 8)

    def describe(self):
        return "%d turns, %d/%d tok in KV, think=%s%s" % (
            len(self.turns), self.fed, self.max_ctx,
            "on" if self.think else "off",
            "" if (not self.think or self.show_think) else " (hidden)")


def generate(eng, qtok, sess, user_text, max_new=0, verbose=True, sampler=None):
    sess.add("user", user_text)
    delta = sess.turn_ids(qtok.encode, user_text)

    # ------------------------------------------------------------------
    # Capacity arithmetic.
    #
    # The pool holds sess.max_ctx tokens.  Prefill stores len(delta)-1 of
    # them; then each loop iteration feeds one more token and consumes one
    # more slot.  With `budget` steps the peak occupancy is
    #     fed + len(delta) + (budget - 1)
    # so budget must stay <= max_ctx - fed - len(delta) + 1.  SLACK keeps a
    # few slots spare rather than sitting exactly on the edge.
    # ------------------------------------------------------------------
    SLACK = 8

    def _room(fed):
        return sess.max_ctx - fed - len(delta) - SLACK

    # Reset FIRST when the turn would not fit.  The old code clamped the
    # budget to a minimum of 32 and then called would_overflow() with the
    # already-clamped budget as the "reserve", so the guard was identically
    # false and the pool overran into allocate_slot()'s
    # "Out of physical memory blocks" RuntimeError.
    if _room(sess.fed) <= 0 or sess.would_overflow(len(delta), sess.reserve):
        if verbose:
            print("[ctx] %d tok in KV + %d new + %d reserve > %d -- "
                  "clearing history"
                  % (sess.fed, len(delta), sess.reserve, sess.max_ctx))
        eng.reset_state()
        sess.reset()
        sess.add("user", user_text)
        delta = sess.turn_ids(qtok.encode, user_text)

    room = _room(sess.fed)
    if room <= 0:
        print("[ctx] the prompt alone is %d tokens and the pool holds %d; "
              "nothing was generated.  Use --ctx to enlarge the pool."
              % (len(delta), sess.max_ctx))
        sess.add("assistant", "")
        return "", []

    if max_new > 0:
        budget = min(max_new, room)
        cap_reason = "max" if max_new <= room else "ctx"
    else:
        budget = room
        cap_reason = "ctx"

    if verbose:
        print("[turn] feeding %d new tokens (%s), reply budget %d"
              % (len(delta), sess.describe(), budget))

    # prefill: everything except the last id
    t0 = time.time()
    try:
        for i, tid in enumerate(delta[:-1]):
            eng.forward_token(tid)
            if verbose and (i + 1) % 8 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[prefill] interrupted")
        sess.add("assistant", "")
        return "", []
    t_prefill = time.time() - t0
    sess.fed += len(delta) - 1
    sess.started = True
    if verbose and len(delta) > 8:
        sys.stdout.write("\n")
    if verbose:
        print("[prefill] %.1f ms for %d tokens (%.1f ms/tok)"
              % (t_prefill * 1000.0, len(delta) - 1,
                 (t_prefill * 1000.0) / max(1, len(delta) - 1)))

    cur = delta[-1]
    out_ids = []
    text_parts = []        # the answer the user reads
    think_parts = []       # the reasoning channel
    in_reasoning = bool(sess.think)
    t_gen = time.time()
    first_ms = None
    first_answer_s = None
    stopped_by_eos = False
    interrupted = False
    n_fwd = 0
    n_think = 0

    if in_reasoning and sess.show_think:
        sys.stdout.write("[think] ")
        sys.stdout.flush()

    try:
        for step in range(budget):
            t = time.time()
            nxt = eng.forward_token(cur)
            n_fwd += 1
            sess.fed += 1
            if sampler is not None:
                picked = sampler.pick(eng)
                if picked >= 0:
                    nxt = picked
            dt = time.time() - t
            if first_ms is None:
                first_ms = dt * 1000.0
            cur = nxt
            if nxt == IM_END:
                stopped_by_eos = True
                break

            # ---- the two reasoning markers ------------------------------
            # These are real generated tokens and belong in out_ids, but they
            # are NOT text: qtok.decode() would turn them into the literal
            # characters "" / "".  They only switch channel.
            if nxt == THINK_OPEN:
                in_reasoning = True
                out_ids.append(nxt)
                if sess.show_think:
                    sys.stdout.write("\n[think] ")
                    sys.stdout.flush()
                continue
            if nxt == THINK_CLOSE:
                in_reasoning = False
                out_ids.append(nxt)
                if first_answer_s is None:
                    first_answer_s = time.time() - t_gen
                if sess.show_think:
                    sys.stdout.write("\n[answer] ")
                    sys.stdout.flush()
                continue

            out_ids.append(nxt)
            piece = qtok.decode(nxt)
            if in_reasoning:
                think_parts.append(piece)
                n_think += 1
                if sess.show_think:
                    sys.stdout.write(piece)
                    sys.stdout.flush()
                elif n_think % 16 == 0:
                    # hidden reasoning still needs a heartbeat, otherwise the
                    # console looks dead for minutes
                    sys.stdout.write(".")
                    sys.stdout.flush()
            else:
                text_parts.append(piece)
                if first_answer_s is None:
                    first_answer_s = time.time() - t_gen
                sys.stdout.write(piece)
                sys.stdout.flush()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[gen] interrupted by Ctrl-C")

    t_total = time.time() - t_gen
    print()
    answer = "".join(text_parts)
    reasoning = "".join(think_parts)
    if not answer and reasoning:
        # The model never closed its reasoning block before the budget ran
        # out.  Keep the reasoning as the reply so history is not empty, and
        # say so loudly: silently returning "" looked like a hang.
        answer = reasoning
        print("[warn] no  was emitted before the budget ran out; the "
              "reply above is ALL reasoning.  Raise --max or --ctx.")
    sess.add("assistant", answer)

    if verbose:
        # NOTE: the loop always runs one extra forward pass that produces
        # <|im_end|> and is then discarded by the `break` above, so that pass
        # is included in t_total but is NOT in out_ids.  Dividing by
        # len(out_ids) therefore inflates the reported average by roughly
        # forward_ms / n (12% for an 8-token answer).  Divide by the number of
        # forward passes actually executed instead.
        n = max(1, n_fwd)
        print("[gen] %d tokens in %.1f ms (%d fwd, first %.1f ms, "
              "avg %.1f ms/tok = %.3f tok/s)"
              % (len(out_ids), t_total * 1000.0, n_fwd, first_ms or 0.0,
                 t_total * 1000.0 / n, n / t_total if t_total > 0 else 0.0))
        print("[gen] unique ids %d/%d" % (len(set(out_ids)), len(out_ids)))
        if sess.think:
            print("[gen] reasoning: %d tok, %.1f s before the first answer token"
                  % (n_think, first_answer_s or 0.0))
        if interrupted:
            print("[gen] stop: interrupted (Ctrl-C) -- answer may be cut")
        elif stopped_by_eos:
            print("[gen] stop: <|im_end|> (natural end of answer, not truncated)")
        else:
            print("[gen] stop: %s budget reached (%d tokens) -- answer may be cut"
                  % (cap_reason, budget))
        # the serial console mangles non-ASCII, so also show the raw bytes
        print("[gen] ids = %s" % (out_ids,))
        print("[gen] utf8 = %s"
              % (" ".join("%02X" % b for b in answer.encode("utf-8"))))
        print("[gen] ascii-safe = %r"
              % (answer.encode("ascii", "backslashreplace").decode("ascii"),))
    return answer, out_ids


def main():
    _utf8_stdout()

    argv = sys.argv[1:]
    max_new = 0                      # 0 = no artificial cap, stop at <|im_end|>
    system = "You are a helpful assistant."
    prompt = None
    ctx = DEFAULT_CTX
    reserve = 256
    think = False
    show_think = True
    # Sampling is ON by default: greedy decoding repeats itself on long
    # answers (measured 11 duplicate ids out of 48).  --greedy turns it off.
    sample_spec = (0.7, 40, 0.9)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--max" and i + 1 < len(argv):
            max_new = int(argv[i + 1])
            i += 2
        elif a == "--greedy":
            sample_spec = None
            i += 1
        elif a == "--system" and i + 1 < len(argv):
            system = argv[i + 1]
            i += 2
        elif a == "--sample" and i + 1 < len(argv):
            # --sample TEMP,TOPK,TOPP   e.g.  --sample 0.7,40,0.9
            p = argv[i + 1].split(",")
            t = float(p[0]) if len(p) > 0 and p[0] else 0.7
            tk = int(p[1]) if len(p) > 1 and p[1] else 40
            tp = float(p[2]) if len(p) > 2 and p[2] else 0.9
            sample_spec = (t, tk, tp)
            i += 2
        elif a == "--cps" and i + 1 < len(argv):
            # build the prompt from decimal code points so the command line
            # stays pure ASCII -- the serial console mangles UTF-8 arguments
            prompt = "".join(chr(int(x)) for x in argv[i + 1].split(","))
            i += 2
        elif a == "--ctx" and i + 1 < len(argv):
            ctx = int(argv[i + 1])
            i += 2
        elif a == "--reserve" and i + 1 < len(argv):
            reserve = int(argv[i + 1])
            i += 2
        elif a == "--think":
            think = True
            i += 1
        elif a == "--no-think":
            think = False
            i += 1
        elif a == "--think-show":
            show_think = True
            i += 1
        elif a == "--think-hide":
            show_think = False
            i += 1
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            prompt = a
            i += 1

    eng, qtok = load_engine(ctx)

    # Read the REAL pool capacity back off the engine.  Session must agree
    # with it exactly, otherwise the reply budget walks off the end of the
    # pool and allocate_slot() raises "Out of physical memory blocks".
    pool_cap = eng.paged_attn.num_blocks * eng.paged_attn.block_size
    sess = Session(system_prompt=system, max_ctx=pool_cap, reserve=reserve,
                   think=think, show_think=show_think)
    eng.reset_state()

    print("=" * 66)
    print("  Qwen3.5-0.8B on ALINX AX7020  (CPU path, 24 layers, 64K vocab)")
    print("  ChatML | %d ctx | no hard cap (stops at <|im_end|>)" % pool_cap)
    print("  reasoning: %s%s"
          % ("ON" if think else "off",
             "" if (not think or show_think) else " (hidden)"))
    if sample_spec is None:
        print("  decoding: greedy (may repeat on long answers)")
    else:
        print("  decoding: sampling T=%.2f top_k=%d top_p=%.2f"
              % sample_spec)
    print("=" * 66)

    sampler = None
    if sample_spec is not None:
        try:
            sampler = Sampler(temperature=sample_spec[0],
                              top_k=sample_spec[1],
                              top_p=sample_spec[2])
            print("[sample] temperature=%.2f top_k=%d top_p=%.2f "
                  "(costs an extra lm_head pass, ~115 ms/token)"
                  % sample_spec)
        except Exception as ex:
            print("[sample] unavailable (%r), falling back to greedy" % (ex,))

    print("[note] the first generated token includes weight staging;")
    print("[note] on a cold page cache it can take ~29 s. Be patient.")

    if prompt is not None:
        generate(eng, qtok, sess, prompt, max_new=max_new, sampler=sampler)
        return 0

    # interactive
    print("\nType your message and press Enter.")
    print("Commands: /clear  /think on|off  /think show|hide  /stats  /help  /quit")
    print("Ctrl-C stops a reply without leaving.  Ctrl-D exits.\n")
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line in ("/clear", "/new", "/reset"):
            eng.reset_state()
            sess.reset()
            print("[clear] conversation and KV cache cleared")
            continue
        if line == "/help":
            print("  /clear  /new  /reset   drop the conversation and the KV cache")
            print("  /think                  show the reasoning state")
            print("  /think on | off         enable / disable the reasoning block")
            print("  /think show | hide      print / suppress the reasoning text")
            print("  /stats                  turns, tokens in KV, think state")
            print("  /quit  /exit  /q        leave (also Ctrl-D)")
            continue
        if line == "/stats":
            print("[stats] %s" % sess.describe())
            continue
        if line == "/think" or line.startswith("/think "):
            arg = line[6:].strip().lower()
            if arg == "":
                print("[think] reasoning is %s, text is %s"
                      % ("on" if sess.think else "off",
                         "shown" if sess.show_think else "hidden"))
            elif arg in ("on", "true", "yes", "1"):
                sess.think = True
                print("[think] reasoning ON for the next turn "
                      "(expect 1.5-4 min per reply on this board)")
            elif arg in ("off", "false", "no", "0"):
                sess.think = False
                print("[think] reasoning off for the next turn")
            elif arg == "show":
                sess.show_think = True
                print("[think] reasoning text will be printed")
            elif arg == "hide":
                sess.show_think = False
                print("[think] reasoning text will be hidden (dots as heartbeat)")
            else:
                print("[think] usage: /think [on|off|show|hide]")
            continue
        try:
            generate(eng, qtok, sess, line, max_new=max_new, sampler=sampler)
        except Exception as ex:
            print("[error] %r" % (ex,))
    print("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
