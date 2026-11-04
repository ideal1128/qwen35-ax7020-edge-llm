#!/bin/bash
# 扫预取距离/局部性，找 gemv 的最优配置
cd /mnt/qwen || exit 1
for cfg in "256 0" "512 0" "1024 0" "2048 0" "512 1" "512 3"; do
  set -- $cfg
  d=$1; l=$2
  gcc -std=c99 -O3 -mcpu=cortex-a9 -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard \
      -fopenmp -DPFD=$d -DPFL=$l -shared -fPIC qwen_kernels_v5.c \
      -o libqwen_kernels.so -lm || exit 1
  printf 'PFD=%-5s PFL=%s  ' "$d" "$l"
  OMP_NUM_THREADS=2 N_TOK=2 python3 bench.py 2>&1 | grep -E 'RESULT tps|lm_head' | tr '\n' ' '
  echo
done
