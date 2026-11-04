#!/bin/sh
# startup.sh -- boot-time readiness check for the Qwen deployment on AX7020.
#
# Placed on the FAT32 boot partition as /sd/startup.sh; /etc/rc.local runs it.
#
# LINE ENDINGS: this file MUST be LF-only.  A CRLF copy cannot self-heal,
# because /bin/sh fails to parse the first multi-line construct (case/if)
# before any recovery code could run -- measured on the board, see
# crlf_test.sh.  Always install with install_startup.sh, which strips CR.
#
# DESIGN NOTE: this script deliberately does NOT launch the interactive chat.
# Binding chat_v3.py to /dev/ttyPS0 would fight the login shell for the only
# console on the board, and a wedged foreground process there would make the
# board unreachable (serial is the only transport -- eth0 is NO-CARRIER).
# Instead it verifies the deployment, writes a log, and prints a banner so
# whoever logs in knows the system is ready and what to run.

LOG=/mnt/qwen/boot.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG" 2>/dev/null
}

# --- make sure the data partition is there -------------------------------
if ! mountpoint -q /mnt/qwen 2>/dev/null; then
    mkdir -p /mnt/qwen 2>/dev/null
    mount /dev/mmcblk0p3 /mnt/qwen 2>/dev/null
fi

if ! mountpoint -q /mnt/qwen 2>/dev/null; then
    log "FATAL: /mnt/qwen not mounted"
    exit 0
fi

log "startup.sh begin (kernel $(uname -r))"

# --- quiet the i2c timeout spam ------------------------------------------
dmesg -n 1 2>/dev/null

export QWEN_CPU_ONLY=1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

# --- readiness check -----------------------------------------------------
STATUS="unknown"
if [ -x /mnt/qwen/accept.sh ]; then
    if (cd /mnt/qwen && ./accept.sh > /mnt/qwen/accept_last.log 2>&1); then
        STATUS="PASS"
    else
        STATUS="FAIL"
    fi
    log "accept.sh -> $STATUS"
else
    log "accept.sh missing, skipping readiness check"
fi

# --- tell whoever logs in ------------------------------------------------
{
    echo ""
    echo "=============================================================="
    echo "  Qwen3.5-0.8B on ALINX AX7020"
    echo "  readiness: $STATUS   (log: /mnt/qwen/boot.log)"
    echo ""
    echo "  start a chat with:   cd /mnt/qwen && ./real_chat_v3"
    echo "  health check with:   cd /mnt/qwen && ./accept.sh"
    echo "=============================================================="
} > /dev/ttyPS0 2>/dev/null

log "startup.sh done, status=$STATUS"
exit 0
