#!/bin/sh
# bfp4_test.sh -- run the model with 4-bit mantissas simulated, then restore.
#
# libqwen_kernels_bfp4.so was built with -DBFP4_SIM, which rounds every weight
# mantissa to a multiple of 16.  That is numerically the same as storing 4-bit
# mantissas with the block exponent bumped by 4, so this measures the accuracy
# cost of a real 4-bit weight file (which would halve bytes read per token).
#
# The restore is unconditional: even if the model crashes or produces garbage,
# the 8-bit library goes back before this script exits.

cd /mnt/qwen || exit 1

NORMAL=libqwen_kernels.so.tmp8
cp libqwen_kernels.so "$NORMAL" || { echo "ABORT: cannot back up current .so"; exit 1; }

restore() {
    cp "$NORMAL" libqwen_kernels.so
    rm -f "$NORMAL"
    echo "restored 8-bit: $(md5sum libqwen_kernels.so | cut -d' ' -f1)"
}
trap restore EXIT INT TERM

echo "8-bit reference .so md5: $(md5sum "$NORMAL" | cut -d' ' -f1)"
echo "4-bit probe     .so md5: $(md5sum libqwen_kernels_bfp4.so | cut -d' ' -f1)"
echo ""

cp libqwen_kernels_bfp4.so libqwen_kernels.so

echo "=== run 1: short prompt, greedy (ids must match the 8-bit reference) ==="
timeout 280 ./real_chat_v3 --cps 20320,22909 --greedy --max 16 2>&1 \
    | grep -E 'prefill|gen\] ids|gen\] [0-9]+ tokens|ascii-safe|Traceback|Error'

echo ""
echo "=== run 2: longer free-form answer, greedy (does the text stay coherent?) ==="
timeout 280 ./real_chat_v3 --cps 35831,32469,26159,20160,20160,21487 2>&1 \
    | grep -E 'gen\] [0-9]+ tokens|ascii-safe|gen\] stop'
