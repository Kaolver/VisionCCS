"""Collect zero-shot results into one table (markdown + CSV).

Reads the per-category npz files run_zeroshot_all.sh writes, not only the
summary JSONs: zero_shot.py saves each category's npz as soon as it finishes but
the summary only at the very end, so a job killed by its time limit still
contributes the categories it completed. Accuracies are recomputed with the
same functions zero_shot.py uses.

    python collect_zeroshot.py --out-dir ./zeroshot_report

Writes <out-dir>/zeroshot_table.md and <out-dir>/zeroshot_table.csv.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from zero_shot import calibrated_accuracy

MODELS = {'llava': 'LLaVA-1.5-7B', 'qwen2': 'Qwen2-VL-7B', 'qwen2_5': 'Qwen2.5-VL-7B'}
DATASETS = {
    'vqa': ('VQAv2', ['object_detection', 'attribute_recognition', 'spatial_recognition']),
    'vg': ('Visual Genome', ['object', 'attribute', 'spatial']),
}
# tag suffix run_zeroshot_all.sh appends after _{dataset}
VARIANTS = {
    'noinstr': ('_noinstr', 'Prompt matched to CCS (no "Answer yes or no.")'),
    'instr': ('', 'With "Answer yes or no." instruction'),
}
SHORT = {'object_detection': 'object', 'attribute_recognition': 'attribute',
         'spatial_recognition': 'spatial'}
COLUMNS = ['object', 'attribute', 'spatial']


def score(npz_path):
    d = np.load(npz_path)
    labels = d['labels'].astype(int)
    margin = d['yes_logit'] - d['no_logit']
    cal, _ = calibrated_accuracy(margin, labels)
    return {'n': int(len(labels)),
            'raw_acc': float(((margin > 0).astype(int) == labels).mean()),
            'calibrated_acc': cal,
            'predicted_yes_rate': float((margin > 0).mean()),
            'true_yes_rate': float(labels.mean())}


def collect(out_dir):
    rows, missing = [], []
    for variant, (suffix, _) in VARIANTS.items():
        for dataset, (_, cats) in DATASETS.items():
            for model in MODELS:
                tag = f'_{dataset}{suffix}'
                summary_path = out_dir / f'zeroshot_{model}{tag}_summary.json'
                summary = (json.loads(summary_path.read_text())
                           if summary_path.exists() else {})
                for cat in cats:
                    npz = out_dir / f'zeroshot_{model}{tag}_{cat}.npz'
                    if not npz.exists():
                        missing.append(f'{model} / {dataset} / {variant} / {cat}')
                        continue
                    rows.append({'variant': variant, 'dataset': dataset, 'model': model,
                                 'category': SHORT.get(cat, cat),
                                 'skipped': summary.get(cat, {}).get('skipped', ''),
                                 **score(npz)})
    return rows, missing


def pct(x):
    return f'{100 * x:.1f}'


def markdown(rows, missing):
    lines = []
    for variant, (_, title) in VARIANTS.items():
        sel = [r for r in rows if r['variant'] == variant]
        if not sel:
            continue
        lines += [f'### {title}', '',
                  '| dataset | model | object | attribute | spatial | all | says yes | n | skipped |',
                  '|---|---|---|---|---|---|---|---|---|']
        for dataset, (dname, _) in DATASETS.items():
            for model, mname in MODELS.items():
                cells = {r['category']: r for r in sel
                         if r['dataset'] == dataset and r['model'] == model}
                if not cells:
                    continue
                per_cat = [f"{pct(cells[c]['calibrated_acc'])} ({pct(cells[c]['raw_acc'])})"
                           if c in cells else '-' for c in COLUMNS]
                # each category is calibrated on its own, as zero_shot.py does,
                # so 'all' is the n-weighted mean of the per-category numbers
                n = sum(r['n'] for r in cells.values())
                w = lambda k: sum(r[k] * r['n'] for r in cells.values()) / n
                skipped = [r['skipped'] for r in cells.values()]
                skipped = sum(skipped) if all(s != '' for s in skipped) else '?'
                done = '' if len(cells) == len(COLUMNS) else ' (partial)'
                lines.append(f'| {dname} | {mname} | ' + ' | '.join(per_cat) +
                             f" | {pct(w('calibrated_acc'))} ({pct(w('raw_acc'))}){done}"
                             f" | {pct(w('predicted_yes_rate'))}% | {n} | {skipped} |")
        lines.append('')
    lines += ['Cells are calibrated accuracy % (raw accuracy %). Raw: the model\'s own '
              'Yes/No preference (yes-logit > no-logit). Calibrated (Burns et al. 2022): '
              'the half of items with the largest yes-no margin are predicted yes, which '
              'removes the model\'s yes/no bias. "says yes" is how often the uncalibrated '
              'model answers yes; the data are 50/50.', '']
    if missing:
        lines += [f'Not finished yet ({len(missing)}): ' + '; '.join(missing), '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default='./zeroshot_report')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rows, missing = collect(out_dir)
    if not rows:
        print(f'No zero-shot npz files in {out_dir}', file=sys.stderr)
        return 1

    with open(out_dir / 'zeroshot_table.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    md = markdown(rows, missing)
    (out_dir / 'zeroshot_table.md').write_text(md, encoding='utf-8')
    print(md)
    print(f'Wrote {out_dir / "zeroshot_table.md"} and {out_dir / "zeroshot_table.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
