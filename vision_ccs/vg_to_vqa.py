"""Convert the Visual Genome question files into the vqav2_mapped.json schema.

vg/{train,val,test}.jsonl (from the martin branch) are yes/no questions about
Visual Genome annotations on COCO images, balanced 50/50 per category. Mapping
them onto the vqav2_mapped.json schema lets every script that takes --vqa-json
run on VG unchanged:

    python vg_to_vqa.py --vg-dir ../vg --out ./vg_mapped.json
    python zero_shot.py --vqa-json ./vg_mapped.json \\
        --categories object attribute spatial --tag _vg

The three splits are pooled by default, which is the same population
linear_ccs.py (martin) draws its own seeded split from, so a zero-shot number
here is comparable to CCS on VG.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

SPLITS = ('train', 'val', 'test')
CATEGORIES = ('object', 'attribute', 'spatial')


def convert(path):
    """VG records -> the item schema build_pairs() expects.

    image_id is the COCO id, not the Visual Genome id: the images on Snellius
    are COCO files named by COCO id, which find_image() zero-pads.
    """
    out = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append({'question_id': r['question_id'],
                    'image_id': int(r['coco_id']),
                    'vg_image_id': r['image_id'],
                    'question': r['question'].strip(),
                    'answer': r['answer'].strip().lower(),
                    'category': r['category'],
                    'generation_type': r.get('generation_type'),
                    'split': r.get('split')})
    return out


def write_atomic(path, text):
    """Write via a temp file + rename, so parallel jobs that regenerate the
    same file never let a reader see it half-written."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vg-dir', default='../vg',
                    help='directory holding {train,val,test}.jsonl')
    ap.add_argument('--splits', nargs='+', default=list(SPLITS), choices=SPLITS,
                    help='which splits to pool (default: all, as linear_ccs.py does)')
    ap.add_argument('--out', default='./vg_mapped.json')
    args = ap.parse_args()

    vg_dir = Path(args.vg_dir)
    records = []
    for split in args.splits:
        path = vg_dir / f'{split}.jsonl'
        if not path.exists():
            print(f'ERROR: {path} not found', file=sys.stderr)
            return 1
        records += convert(path)

    # question_id is the join key downstream; the splits are disjoint by image,
    # so a clash means the input files changed shape.
    qids = [r['question_id'] for r in records]
    if len(set(qids)) != len(qids):
        print('ERROR: duplicate question_id across the pooled splits', file=sys.stderr)
        return 1
    bad = sorted({r['answer'] for r in records} - {'yes', 'no'})
    if bad:
        print(f'ERROR: non yes/no answers {bad}', file=sys.stderr)
        return 1

    mapped = {c: [] for c in CATEGORIES}
    for i, r in enumerate(records):
        if r['category'] in mapped:
            mapped[r['category']].append({**r, 'index': i})

    for c, items in mapped.items():
        n_yes = sum(1 for it in items if it['answer'] == 'yes')
        print(f'{c:10s} {len(items):6d} items  {n_yes / max(len(items), 1):.1%} yes  '
              f'{len(set(it["image_id"] for it in items)):5d} images')

    write_atomic(args.out, json.dumps(mapped, indent=1))
    print(f'\nWrote {args.out} (splits pooled: {", ".join(args.splits)})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
