"""Compare label-free restart selection criteria."""

import argparse
import json
import sys

import numpy as np

CRITERIA = {
    'loss (Burns)':      ('loss', +1),
    'consistency_err':   ('consistency_err', +1),
    'confidence':        ('confidence', +1),
    'saturated_low':     ('saturated', +1),
    'saturated_high':    ('saturated', -1),
}


def load_runs(path):
    """Yield (cell_key, run_key, restarts) for every run with restart records."""
    with open(path) as f:
        res = json.load(f)
    out = []
    for cell, c in res.get('cells', {}).items():
        for run, v in c.get('runs', {}).items():
            r = v.get('ccs', {}).get('restarts') or []
            if r and 'test_acc_flipped' in r[0]:
                out.append((cell, run, r))
    return out


def select(restarts, key, direction):
    """Index of the restart this criterion would choose."""
    vals = np.array([r[key] for r in restarts], dtype=float) * direction
    return int(np.argmin(vals))


def _rule_picks(runs, key, direction):
    return np.array([rs[select(rs, key, direction)]['test_acc_flipped']
                     for _, _, rs in runs])


def print_by_cell(runs):
    """Per (cell, split) breakdown of loss-selection vs consistency-selection
    vs oracle. This is the shape the results table needs: the aggregate hides
    that the gain is concentrated in the cells where a restart went degenerate.
    """
    from collections import OrderedDict
    groups = OrderedDict()
    for cell, run, rs in runs:
        groups.setdefault((cell, run.split('/')[0]), []).append((cell, run, rs))

    print('\nPer-cell breakdown (mean +/- std over seeds):')
    hdr = (f"  {'cell':30s} {'split':10s} {'loss(Burns)':>14s} "
           f"{'consistency':>14s} {'oracle':>8s} {'gain':>7s}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for (cell, split), rr in groups.items():
        burns = _rule_picks(rr, 'loss', +1)
        cons = _rule_picks(rr, 'consistency_err', +1)
        orac = np.array([max(r['test_acc_flipped'] for r in rs) for _, _, rs in rr])
        print(f"  {cell:30s} {split:10s} {burns.mean():6.1%}+/-{burns.std():5.1%} "
              f"{cons.mean():6.1%}+/-{cons.std():5.1%} {orac.mean():7.1%} "
              f"{cons.mean() - burns.mean():+6.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results_json')
    ap.add_argument('--by-cell', action='store_true',
                    help='break the table down per model/category/split')
    ap.add_argument('--combine', action='store_true',
                    help='also try loss+consistency_err after per-run z-scoring')
    args = ap.parse_args()

    runs = load_runs(args.results_json)
    if not runs:
        print(f'No restart records with accuracies found in {args.results_json}.\n'
              'Re-run reanalysis.py with a version that scores every restart.')
        return 1

    n_restarts = len(runs[0][2])
    print(f'{len(runs)} runs x {n_restarts} restarts = {len(runs)*n_restarts} probes\n')

    allr = [r for _, _, rs in runs for r in rs]
    acc_all = np.array([r['test_acc_flipped'] for r in allr])
    print('Correlation with test accuracy (pooled over every restart):')
    seen = set()
    for name, (key, _) in CRITERIA.items():
        if key not in allr[0] or key in seen:
            continue
        seen.add(key)
        name = key
        v = np.array([r[key] for r in allr], dtype=float)
        if key == 'loss':
            v = np.log(np.maximum(v, 1e-12))
            name = name + ' [log]'
        rho = np.corrcoef(v, acc_all)[0, 1] if v.std() > 0 else float('nan')
        print(f'  {name:22s} {rho:+.3f}')

    print('\nAccuracy of the restart each rule selects (mean over runs):')
    oracle = np.array([max(r['test_acc_flipped'] for r in rs) for _, _, rs in runs])
    worst = np.array([min(r['test_acc_flipped'] for r in rs) for _, _, rs in runs])
    rand = np.array([np.mean([r['test_acc_flipped'] for r in rs]) for _, _, rs in runs])

    rows = []
    for name, (key, direction) in CRITERIA.items():
        if key not in runs[0][2][0]:
            continue
        picked = np.array([rs[select(rs, key, direction)]['test_acc_flipped']
                           for _, _, rs in runs])
        rows.append((name, picked))

    if args.combine:
        picked = []
        for _, _, rs in runs:
            def z(k):
                v = np.array([r[k] for r in rs], dtype=float)
                return (v - v.mean()) / (v.std() + 1e-12)
            score = z('loss') + z('consistency_err')
            picked.append(rs[int(np.argmin(score))]['test_acc_flipped'])
        rows.append(('loss+consistency (z)', np.array(picked)))

    baseline = dict(rows)['loss (Burns)']
    print(f"  {'rule':24s} {'mean':>7s} {'std':>7s} {'regret':>8s} {'vs Burns':>9s}")
    print('  ' + '-' * 60)
    for name, picked in sorted(rows, key=lambda t: -t[1].mean()):
        print(f'  {name:24s} {picked.mean():6.1%} {picked.std():6.1%} '
              f'{(oracle - picked).mean():7.1%} {picked.mean()-baseline.mean():+8.1%}')
    print('  ' + '-' * 60)
    print(f"  {'random restart':24s} {rand.mean():6.1%} {rand.std():6.1%} "
          f"{(oracle - rand).mean():7.1%} {rand.mean()-baseline.mean():+8.1%}")
    print(f"  {'ORACLE (uses labels)':24s} {oracle.mean():6.1%} {oracle.std():6.1%} "
          f"{0.0:7.1%} {oracle.mean()-baseline.mean():+8.1%}")
    print(f"  {'worst restart':24s} {worst.mean():6.1%} {worst.std():6.1%} "
          f"{(oracle - worst).mean():7.1%} {worst.mean()-baseline.mean():+8.1%}")

    if args.by_cell:
        print_by_cell(runs)

    print('\nWorst single run (largest gap between best and selected restart):')
    gaps = [(max(r['test_acc_flipped'] for r in rs)
             - rs[select(rs, 'loss', +1)]['test_acc_flipped'], cell, run, rs)
            for cell, run, rs in runs]
    gap, cell, run, rs = max(gaps, key=lambda t: t[0])
    print(f'  {cell}  {run}   loss-selection gave up {gap:.1%}')
    hdr = f"    {'#':>2s} {'loss':>10s} {'consist':>8s} {'conf':>7s} {'sat':>6s} {'acc':>7s}"
    print(hdr)
    for i, r in enumerate(sorted(rs, key=lambda z: z['loss'])):
        mark = '  <- picked by loss' if i == 0 else ''
        print(f"    {i:2d} {r['loss']:10.2e} {r.get('consistency_err', float('nan')):8.4f} "
              f"{r.get('confidence', float('nan')):7.4f} {r.get('saturated', float('nan')):6.2f} "
              f"{r['test_acc_flipped']:7.1%}{mark}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
