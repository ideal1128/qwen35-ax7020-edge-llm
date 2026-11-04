#!/bin/sh
# crlf_test.sh -- verify startup.sh self-heals from CRLF line endings.
# Runs on the board.  Pure POSIX sh.

cd /mnt/qwen || exit 1

echo "=== 1. current /sd/startup.sh line endings ==="
if od -c /sd/startup.sh 2>/dev/null | head -3 | grep -q '\\r'; then
    echo "    /sd/startup.sh HAS CR (self-heal needed)"
else
    echo "    /sd/startup.sh is LF-only"
fi

echo "=== 2. build a genuinely CRLF copy ==="
awk '{printf "%s\r\n", $0}' startup.sh > crlf.sh
if od -c crlf.sh | head -3 | grep -q '\\r'; then
    echo "    crlf.sh HAS CR (good, test is valid)"
else
    echo "    crlf.sh is LF-only -- TEST INVALID"
    rm -f crlf.sh
    exit 1
fi

echo "=== 3. run the CRLF copy ==="
OUT=$(sh crlf.sh 2>&1)
echo "$OUT" | head -8
echo "--- rc=$? ---"

echo "=== 4. did it self-heal? ==="
echo "$OUT" | grep -q 'readiness:' && echo "    banner printed: PASS" || echo "    banner printed: FAIL"
echo "$OUT" | grep -q 'not found' && echo "    shell errors: FAIL" || echo "    shell errors: none"

rm -f crlf.sh /tmp/_startup_lf.sh
echo "=== done ==="
