#!/bin/bash
# 比较不同预取距离下的 PS 侧读带宽
cd /mnt/qwen || exit 1
for pf in 0 256 512 1024 2048; do
  if [ "$pf" -eq 0 ]; then
    gcc -std=gnu99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -fopenmp membw.c -o m0 -lm || exit 1
  else
    gcc -std=gnu99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -fopenmp -DPF=$pf membw.c -o "m$pf" -lm || exit 1
  fi
done
for f in m0 m256 m512 m1024 m2048; do
  printf '%s  ' "$f"
  OMP_NUM_THREADS=2 ./"$f" qwen35_weights_64k.bin 2 | tail -1
done
