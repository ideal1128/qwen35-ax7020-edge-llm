"""MTP inventory: is there a draft model / MTP head on the board at all?"""
import collections
import json
import os

W = '/mnt/qwen/'
M = json.load(open(W + 'qwen35_weights_64k_manifest.json'))
names = sorted(M['tensors'])
print('manifest tensors: %d' % len(names))

hit = [n for n in names if 'mtp' in n.lower() or 'draft' in n.lower()
       or 'eagle' in n.lower()]
print('mtp/draft/eagle tensors: %d' % len(hit))
for n in hit[:40]:
    info = M['tensors'][n]
    print('   %-60s %-14s %d' % (n, info['shape'], info['byte_size']))

c = collections.Counter(n.split('.')[0] for n in names)
print('top-level prefixes: %s' % dict(c))
c2 = collections.Counter('.'.join(n.split('.')[:2]) for n in names)
print('layer groups: %s' % dict(sorted(c2.items())))

for p in ('config.json', 'model_config.json', 'qwen35_config.json'):
    if os.path.exists(W + p):
        print('--- %s ---' % p)
        print(open(W + p).read()[:1200])

print('--- files ---')
for f in sorted(os.listdir(W)):
    full = W + f
    if os.path.isfile(full) and f.endswith(('.bin', '.json', '.so', '.c', '.py')):
        print('  %-44s %12d' % (f, os.path.getsize(full)))
