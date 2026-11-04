#!/bin/sh
# repl_test.sh -- verify chat_v3.py interactive REPL works when stdin is a pipe.
#
# The demo runs the REPL, not --cps one-shot mode, and until now only the
# one-shot path was tested.  Piping stdin exercises the same input()/EOF/
# /quit logic without needing an interactive serial terminal.
#
# The Chinese prompt is emitted as raw UTF-8 octal escapes so this file stays
# pure ASCII:
#   U+4F60 ni  = E4 BD A0 = \344\275\240
#   U+597D hao = E5 A5 BD = \345\245\275
#   U+4F60 U+662F U+8C01 "ni shi shui" = E4 BD A0 E6 98 AF E8 B0 81

cd /mnt/qwen || exit 1

echo "=== 1. ASCII prompt through the REPL ==="
printf 'hello\n/quit\n' | timeout 200 python3 -u chat_v3.py --max 6 2>&1 \
    | grep -E '\[prompt\]|\[gen\]|you>|bye|rror' | head -8

echo
echo "=== 2. Chinese prompt through the REPL (UTF-8 octal escapes) ==="
printf '\344\275\240\345\245\275\n/quit\n' | timeout 200 python3 -u chat_v3.py --max 8 2>&1 \
    | grep -E '\[prompt\]|\[gen\] (ids|ascii)|rror' | head -8

echo
echo "=== 3. EOF without /quit must exit cleanly ==="
printf '' | timeout 60 python3 -u chat_v3.py --max 4 2>&1 | tail -3
echo "    (no traceback above means clean EOF handling)"

echo
echo "=== 4. /reset must be accepted ==="
printf '/reset\n/quit\n' | timeout 200 python3 -u chat_v3.py --max 4 2>&1 \
    | grep -E 'reset|bye|rror' | head -4

echo
echo "=== repl_test done ==="
