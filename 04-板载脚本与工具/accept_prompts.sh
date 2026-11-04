#!/bin/bash
# On-board acceptance test: fixed prompts, capture output + wall time.
# ASCII only -- the board's terminal locale is C and mangles multibyte text.
cd /mnt/qwen || exit 1
export OMP_NUM_THREADS=2
mountpoint -q /mnt/qwen || mount /dev/mmcblk0p3 /mnt/qwen

LOG=/mnt/qwen/acceptance_log.txt
{
echo "===== ALINX AX7020 / Qwen3.5-0.8B on-board acceptance ====="
date '+%Y-%m-%d %H:%M:%S'
echo "kernel: $(uname -r)"
echo "python: $(python3 -V 2>&1)"
echo "cores : $(nproc)"
echo "weights: $(ls -l qwen35_weights_64k.bin | awk '{print $5}') bytes"
echo "embed  : $(ls -l embedding_64k.bin | awk '{print $5}') bytes"
echo

i=0
while IFS= read -r p; do
    [ -z "$p" ] && continue
    i=$((i+1))
    echo "----- [$i] prompt: $p"
    t0=$(date +%s%N)
    python3 qwen_real_engine.py "$p" 2>/dev/null | sed -n '/^Assistant:/,$p'
    t1=$(date +%s%N)
    echo "----- elapsed: $(( (t1-t0)/1000000 )) ms"
    echo
done <<'EOF'
The capital of France is
hello
1 2 3 4 5 6 7 8
def fibonacci(n):
import numpy as np
EOF
echo "===== end ====="
} 2>&1 | tee "$LOG"
