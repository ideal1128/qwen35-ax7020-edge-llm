"""List the real BFP8 (M,K) shapes and multiplicities, for the batch-GEMV bench."""
import collections
import json

W = '/mnt/qwen/'
M = json.load(open(W + 'qwen35_weights_64k_manifest.json'))
c = collections.Counter()
for name, v in M['tensors'].items():
    s = v['shape']
    if len(s) != 2:
        continue
    m, k = s
    if k % 64:
        continue
    if v['byte_size'] != m * k + m * k // 64:
        continue
    c[(m, k)] += 1

print('distinct BFP8 shapes: %d, tensors: %d' % (len(c), sum(c.values())))
tot = 0
for (m, k), n in sorted(c.items(), key=lambda x: -x[0][0] * x[0][1]):
    print('  M=%-5d K=%-5d x%-3d %9.3f MB  total %9.3f MB'
          % (m, k, n, m * k / 1e6, n * m * k / 1e6))
    tot += n * m * k
print('total mantissa: %.1f MB' % (tot / 1e6))
