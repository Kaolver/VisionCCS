"""Convert POPE annotations into the vqav2_mapped.json schema.

POPE (Li et al. 2023) is yes/no object-existence questions -- the contrast-pair
shape this repo already builds -- so it connects the hallucination framing to an
actual hallucination benchmark. It is also balanced 50/50 by construction, and
its taxonomy is given rather than inferred by an LLM categorisation step.

Download coco_pope_{random,popular,adversarial}.json from
github.com/RUCAIBox/POPE, then:

    python pope_to_vqa.py --pope-dir ./pope --out ./pope_mapped.json

and point the rest of the pipeline at it:

    python extract.py --vqa-json ./pope_mapped.json \\
        --categories pope_random pope_popular pope_adversarial \\
        --image-dirs /path/to/coco/val2014

The files are JSON Lines despite the .json extension, so both are accepted.
"""

import argparse
import json
import sys
from pathlib import Path

SPLITS = ('random', 'popular', 'adversarial')


def read_records(path):
    """Read a POPE annotation file (JSON Lines, or a JSON list)."""
    text = Path(path).read_text(encoding='utf-8').strip()
    if not text:
        return []
    if text[0] == '[':
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def convert(records, category):
    """POPE records -> the item schema build_pairs() expects.

    image_id stays the POPE filename string: find_image() only zero-pads
    integers, so casting to int would produce '000000310196.jpg' instead of
    'COCO_val2014_000000310196.jpg'.
    """
    out, seen, dropped = [], set(), 0
    for i, r in enumerate(records):
        answer = str(r.get('label', r.get('answer', ''))).strip().lower()
        question = str(r.get('text', r.get('question', ''))).strip()
        image = r.get('image', r.get('image_id'))
        if answer not in ('yes', 'no') or not question or not image:
            dropped += 1
            continue
        # question_id is the join key downstream and POPE restarts numbering per
        # split, so namespace by split and fail loudly on a within-split clash.
        split = category.replace('pope_', '')
        offset = SPLITS.index(split) * 1_000_000 if split in SPLITS else 0
        qid = int(r.get('question_id', i)) + offset
        if qid in seen:
            raise ValueError(f'duplicate question_id {qid} in {category}')
        seen.add(qid)
        out.append({'question_id': qid, 'image_id': str(image),
                    'question': question, 'answer': answer,
                    'category': category, 'index': i})
    return out, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pope-dir', default='./pope',
                    help='directory holding coco_pope_{random,popular,adversarial}.json')
    ap.add_argument('--prefix', default='coco_pope_',
                    help='filename prefix (use gqa_pope_ / aokvqa_pope_ for the '
                         'other POPE sources)')
    ap.add_argument('--splits', nargs='+', default=list(SPLITS), choices=SPLITS)
    ap.add_argument('--out', default='./pope_mapped.json')
    args = ap.parse_args()

    pope_dir = Path(args.pope_dir)
    if not pope_dir.is_dir():
        print(f'ERROR: {pope_dir} is not a directory. Download the POPE '
              f'annotations from github.com/RUCAIBox/POPE first.', file=sys.stderr)
        return 1

    mapped, any_found = {}, False
    for split in args.splits:
        path = pope_dir / f'{args.prefix}{split}.json'
        if not path.exists():
            print(f'[skip] {path} not found')
            continue
        any_found = True
        items, dropped = convert(read_records(path), f'pope_{split}')
        mapped[f'pope_{split}'] = items
        n_yes = sum(1 for it in items if it['answer'] == 'yes')
        print(f'pope_{split:12s} {len(items):6d} items  '
              f'{n_yes / max(len(items), 1):.1%} yes  '
              f'{len(set(it["image_id"] for it in items)):5d} images  '
              f'{dropped} dropped')

    if not any_found:
        print(f'ERROR: no POPE files matching {args.prefix}*.json in {pope_dir}',
              file=sys.stderr)
        return 1

    Path(args.out).write_text(json.dumps(mapped, indent=1))
    print(f'\nWrote {args.out} with categories {list(mapped)}')
    print('Next: point --vqa-json at it and pass --categories '
          + ' '.join(mapped) + '\n'
          'Note POPE uses COCO val2014 filenames, so --image-dirs must contain '
          'the val2014 images.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
