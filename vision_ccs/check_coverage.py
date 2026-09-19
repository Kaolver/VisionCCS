#!/usr/bin/env python3
"""How many of our VQAv2 images exist in the Snellius COCO mirror?

Run on Snellius (login or int node). Needs only the stdlib, so no venv:
    python3 check_coverage.py
"""
import json, os, sys
from collections import Counter

DIRS = [
    '/scratch-nvme/ml-datasets/coco/train/data',
    '/scratch-nvme/ml-datasets/coco/validation/data',
    '/scratch-nvme/ml-datasets/coco/test/data',
]
CATS = ['object_detection', 'attribute_recognition', 'spatial_recognition']

d = json.load(open('vqav2_mapped.json'))

# Index each directory once; 100k stat() calls over a network FS is slow.
index = {}
for p in DIRS:
    try:
        names = set(os.listdir(p))
    except OSError as e:
        print(f"  ! {p}: {e}")
        names = set()
    index[p] = names
    print(f"  {p}: {len(names)} files")

print()
found = Counter()
missing_per_cat = Counter()
total_per_cat = Counter()
missing_ids = set()

for c in CATS:
    for item in d[c]:
        total_per_cat[c] += 1
        fn = f"{item['image_id']:012d}.jpg"
        for p in DIRS:
            if fn in index[p]:
                found[p] += 1
                break
        else:
            missing_per_cat[c] += 1
            missing_ids.add(item['image_id'])

print("=== per-category sample coverage ===")
for c in CATS:
    t, m = total_per_cat[c], missing_per_cat[c]
    print(f"  {c:26s} {t-m:5d}/{t:5d} found  ({100*(t-m)/t:5.1f}%)")

t = sum(total_per_cat.values()); m = sum(missing_per_cat.values())
print(f"\n  {'TOTAL':26s} {t-m:5d}/{t:5d} found  ({100*(t-m)/t:5.1f}%)")

print("\n=== which directory supplied them ===")
for p, n in found.items():
    print(f"  {p}: {n}")

uniq = {i['image_id'] for c in CATS for i in d[c]}
print(f"\nunique images needed : {len(uniq)}")
print(f"unique images missing: {len(missing_ids)}")

if missing_ids:
    with open('missing_image_ids.txt', 'w') as f:
        for i in sorted(missing_ids):
            f.write(f"{i}\n")
    print("wrote missing_image_ids.txt")
