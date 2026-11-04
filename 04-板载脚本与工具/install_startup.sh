#!/bin/sh
# install_startup.sh -- install startup.sh onto the FAT32 boot partition.
#
# Run this ON THE BOARD:
#     cd /mnt/qwen && ./install_startup.sh
#
# Why a script: startup.sh must be LF-only.  /bin/sh cannot recover from a
# CRLF copy, because it fails to parse the first multi-line construct before
# any recovery code could run (measured -- see crlf_test.sh).  So CR is
# stripped here, at install time, where we still can.

set -e

SRC=/mnt/qwen/startup.sh
DST=/sd/startup.sh
LOG=/mnt/qwen/boot.log

say() { echo "[install_startup] $*"; }

if [ ! -f "$SRC" ]; then
    say "ERROR: $SRC missing"
    exit 1
fi

if ! mountpoint -q /sd 2>/dev/null; then
    say "/sd not mounted, trying to mount /dev/mmcblk0p1"
    mkdir -p /sd
    mount /dev/mmcblk0p1 /sd 2>/dev/null || true
fi

if ! mountpoint -q /sd 2>/dev/null; then
    say "ERROR: /sd not mounted (need the FAT32 boot partition)"
    exit 1
fi

# strip CR while copying -- this is the whole point of the script
tr -d '\r' < "$SRC" > "$DST"
chmod +x "$DST" 2>/dev/null || true

# verify the result really is LF-only
if od -c "$DST" 2>/dev/null | grep -q '\\r'; then
    say "ERROR: CR survived, $DST is still CRLF"
    exit 1
fi

# syntax check the installed copy
if ! sh -n "$DST" 2>/dev/null; then
    say "ERROR: installed copy has a syntax error"
    exit 1
fi

say "installed $DST ($(wc -c < "$DST" | tr -d ' ') bytes, LF-only, syntax OK)"

# is the rc.local hook in place?
if grep -q '/sd/startup.sh' /etc/rc.local 2>/dev/null; then
    say "rc.local hook present"
else
    say "WARNING: /etc/rc.local does not reference /sd/startup.sh"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] install_startup.sh installed $DST" >> "$LOG" 2>/dev/null

say "done. reboot to test, or run 'sh /sd/startup.sh' now."
