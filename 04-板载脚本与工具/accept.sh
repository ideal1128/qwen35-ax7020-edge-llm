#!/bin/sh
# accept.sh -- one-command health check for the Qwen deployment on AX7020.
#
# Usage:  cd /mnt/qwen && ./accept.sh
# Last line is always  RESULT: PASS  or  RESULT: FAIL.
#
# The functional end-to-end prompt test lives in accept_prompts.sh.
# Pure ASCII.  POSIX sh.  No /usr/bin/time, no python dependencies.

Q=/mnt/qwen
FAIL=0
SKIP=0

pass() { echo "  [PASS] $*"; }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }
skip() { echo "  [SKIP] $*"; SKIP=$((SKIP + 1)); }

echo "=== Qwen3.5-0.8B on AX7020 -- acceptance ==="
echo "date: $(date)"
echo

# ---------------------------------------------------------------- 1. mount
echo "[1] data partition"
if mountpoint -q "$Q" 2>/dev/null; then
    pass "$Q mounted"
else
    fail "$Q not mounted"
fi
if touch "$Q/.acc_probe" 2>/dev/null; then
    rm -f "$Q/.acc_probe"
    pass "$Q writable"
else
    fail "$Q not writable"
fi

# --------------------------------------------------------------- 2. files
echo
echo "[2] required files"
check_size() {
    f="$Q/$1"; want="$2"; tol="$3"
    if [ ! -f "$f" ]; then
        fail "$1 missing"
        return
    fi
    got=$(wc -c < "$f" | tr -d ' ')
    lo=$((want - tol)); hi=$((want + tol))
    if [ "$got" -ge "$lo" ] && [ "$got" -le "$hi" ]; then
        pass "$1 ($got bytes)"
    else
        fail "$1 size $got, expected ~$want (+-$tol)"
    fi
}
check_size qwen35_weights_64k.bin          643362048 1048576
check_size embedding_64k.bin               134217728 1048576
check_size tokenizer_64k.bin                  734858   65536

for f in qwen35_weights_64k_manifest.json libqwen_kernels.so \
         libtokenizer.so qwen_real_engine.py qwen_tokenizer.py \
         chat_v3.py real_chat_v3; do
    if [ -f "$Q/$f" ]; then
        pass "$f present"
    else
        fail "$f missing"
    fi
done

# ---------------------------------------------------------------- 3. memory
echo
echo "[3] memory"
AVAIL=0
if [ -r /proc/meminfo ]; then
    MA=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)
    if [ -n "$MA" ]; then
        AVAIL=$((MA / 1024))
    else
        MF=$(awk '/^MemFree:/ {print $2}' /proc/meminfo 2>/dev/null)
        CA=$(awk '/^Cached:/ {print $2}' /proc/meminfo 2>/dev/null)
        [ -z "$MF" ] && MF=0
        [ -z "$CA" ] && CA=0
        AVAIL=$(( (MF + CA) / 1024 ))
    fi
fi
if [ "$AVAIL" -ge 800 ]; then
    pass "${AVAIL} MB available (need >= 800)"
else
    fail "${AVAIL} MB available (need >= 800)"
fi

# ----------------------------------------------------------------- 4. disk
echo
echo "[4] disk"
FREE_MB=$(df -m "$Q" 2>/dev/null | awk 'NR==2 {print $4}')
[ -z "$FREE_MB" ] && FREE_MB=0
if [ "$FREE_MB" -ge 1024 ]; then
    pass "${FREE_MB} MB free"
else
    fail "${FREE_MB} MB free (need >= 1024)"
fi

# ------------------------------------------------------------------ 5. cpu
echo
echo "[5] cpu"
FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)
if [ -n "$FREQ" ]; then
    pass "cpu0 at $((FREQ / 1000)) MHz"
else
    skip "cpufreq not readable"
fi
NCPU=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null)
pass "$NCPU core(s) online"

# ------------------------------------------------------------- 6. checksum
echo
echo "[6] weight checksum"
if [ -f "$Q/expected.md5" ]; then
    WANT=$(tr -d ' \r\n' < "$Q/expected.md5")
    GOT=$(head -c 1048576 "$Q/qwen35_weights_64k.bin" 2>/dev/null | md5sum | cut -d' ' -f1)
    if [ "$WANT" = "$GOT" ]; then
        pass "first 1 MB md5 matches"
    else
        fail "first 1 MB md5 $GOT != $WANT"
    fi
else
    skip "expected.md5 not present (record it once with: head -c 1048576 qwen35_weights_64k.bin | md5sum)"
fi

# ------------------------------------------------------- 7. python imports
echo
echo "[7] python import"
if python3 -c "import sys; sys.path.insert(0,'$Q'); import qwen_real_engine" 2>/dev/null; then
    pass "qwen_real_engine imports"
else
    fail "qwen_real_engine fails to import"
fi
if python3 -c "import sys; sys.path.insert(0,'$Q'); import qwen_tokenizer" 2>/dev/null; then
    pass "qwen_tokenizer imports"
else
    fail "qwen_tokenizer fails to import"
fi

# --------------------------------------------------------------- 8. result
echo
echo "=============================================="
echo "  failures: $FAIL   skipped: $SKIP"
if [ "$FAIL" -eq 0 ]; then
    echo "  RESULT: PASS"
    exit 0
else
    echo "  RESULT: FAIL"
    exit 1
fi
