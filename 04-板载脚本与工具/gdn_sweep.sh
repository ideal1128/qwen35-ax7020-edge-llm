#!/bin/bash
# 扫 GDN state 预取距离（单位：行，每行 512 B）
cd /mnt/qwen || exit 1
for g in 2 4 8 16; do
  gcc -std=c99 -O3 -mcpu=cortex-a9 -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard \
      -fopenmp -DPFD=96 -DPFL=0 -DGDPF=$g -shared -fPIC qwen_kernels_v8.c \
      -o libqwen_kernels.so -lm || exit 1
  printf 'GDPF=%-3s ' "$g"
  OMP_NUM_THREADS=2 ENGINE=qwen_real_engine_m2a python3 profile_kernels.py 2>&1 \
      | grep -E 'gdn_recurrent|wall' | tr '\n' ' '
  echo
done
