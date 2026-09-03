"""Item-level comparison of CCS predictions against zero-shot baseline."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from zero_shot import calibrate


def load_zeroshot(zs_dir, model_tag, category):
    f = Path(zs_dir) / f'zeroshot_{model_tag}_{category}.npz'
    if not f.exists():
        return None
    d = np.load(f)
    return {'margin': d['yes_logit'] - d['no_logit'], 'labels': d['labels'].astype(int)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results_json')
    ap.add_argument('--zeroshot-dir', default='./zeroshot')
    ap.add_argument('--split', default='ungrouped', choices=['ungrouped', 'grouped'])
    args = ap.parse_args()

    res = json.loads(Path(args.results_json).read_text())
    rows, totals = [], {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'n': 0, 'agree': 0}

    for cell, c in res.get('cells', {}).items():
        model_tag, category = cell.split('/', 1)
        zs = load_zeroshot(args.zeroshot_dir, model_tag, category)
        if zs is None:
            print(f'[skip] no zero-shot file for {cell}')
            continue

        for run, v in c.get('runs', {}).items():
            if not run.startswith(args.split + '/'):
                continue
            te = v.get('test_idx')
            pred = v.get('ccs_test_pred')
            if te is None or pred is None:
                print(f'[skip] {cell} {run}: no per-item predictions')
                continue

            te = np.asarray(te)
            ccs = np.asarray(pred)
            if te.max() >= len(zs['labels']):
                print(f'[skip] {cell} {run}: test_idx out of range for zero-shot rows')
                continue

            y = zs['labels'][te]
            m = zs['margin'][te]
            zsp = calibrate(m)

            ccs_ok, zs_ok = (ccs == y), (zsp == y)
            a = int((ccs_ok & zs_ok).sum())
            b = int((ccs_ok & ~zs_ok).sum())
            cc = int((~ccs_ok & zs_ok).sum())
            d = int((~ccs_ok & ~zs_ok).sum())
            agree = int((ccs == zsp).sum())
            n = len(y)
            for k, val in (('a', a), ('b', b), ('c', cc), ('d', d),
                           ('n', n), ('agree', agree)):
                totals[k] += val
            rows.append((cell, run, n, ccs_ok.mean(), zs_ok.mean(),
                         agree / n, b, cc))

    if not rows:
        print('\nNothing to compare.')
        return 1

    print(f"\nPer-run ({args.split} splits), zero-shot restricted to CCS test rows:")
    hdr = (f"  {'cell':28s} {'seed':>6s} {'n':>5s} {'CCS':>7s} {'zero-shot':>10s} "
           f"{'agree':>7s} {'CCS+':>5s} {'CCS-':>5s}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for cell, run, n, ca, za, ag, b, cc in rows:
        print(f"  {cell:28s} {run.split('seed')[-1]:>6s} {n:5d} {ca:6.1%} "
              f"{za:9.1%} {ag:6.1%} {b:5d} {cc:5d}")

    n = totals['n']
    print(f"\nPooled over {len(rows)} runs, {n} item-predictions:")
    print(f"  CCS accuracy        {(totals['a'] + totals['b']) / n:.1%}")
    print(f"  zero-shot accuracy  {(totals['a'] + totals['c']) / n:.1%}")
    print(f"  agreement rate      {totals['agree'] / n:.1%}")
    print(f"\n  {'':22s} {'zero-shot right':>16s} {'zero-shot wrong':>16s}")
    print(f"  {'CCS right':22s} {totals['a']:16d} {totals['b']:16d}   <- CCS adds")
    print(f"  {'CCS wrong':22s} {totals['c']:16d} {totals['d']:16d}")
    print(f"\n  CCS adds {totals['b']} items, loses {totals['c']}  "
          f"-> net {totals['b'] - totals['c']:+d} ({(totals['b']-totals['c'])/n:+.2%})")

    b, cc = totals['b'], totals['c']
    if b + cc > 0:
        chi2 = (abs(b - cc) - 1) ** 2 / (b + cc)
        print(f"  McNemar chi2 = {chi2:.2f} on the {b + cc} discordant pairs "
              f"({'significant at p<0.05' if chi2 > 3.84 else 'not significant'})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
