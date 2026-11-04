#!/bin/bash
# 更细的预取距离扫描
cd /mnt/qwen || exit 1
for d in 32 64 96 128 192; do
  gcc -std=c99 -O3 -mcpu=cortex-a9 -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard \
      -fopenmp -DPFD=$d -DPFL=0 -shared -fPIC qwen_kernels_v7.c \
      -o libqwen_kernels.so -lm || exit 1
  printf 'PFD=%-5s ' "$d"
  OMP_NUM_THREADS=2 ENGINE=qwen_real_engine_m2a N_TOK=2 python3 bench.py 2>&1 \
      | grep -E 'RESULT tps|lm_head' | tr '\n' ' '
  echo
done
