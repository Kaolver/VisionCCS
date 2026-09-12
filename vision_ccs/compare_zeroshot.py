"""Item-level comparison of CCS predictions against zero-shot baseline."""

# ============================================================================
# NOTE (review): ROLE OF THIS FILE
# Item-level comparison of CCS vs zero-shot on the SAME test rows. Needs a
# reanalysis results JSON whose runs contain 'test_idx' and 'ccs_test_pred'
# (written by reanalysis.run_cell) and the zeroshot_<model>_<cat>.npz files.
# Row alignment assumes both extractors skipped the same missing images and
# that zero_shot.py ran without --limit.
# ============================================================================
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from zero_shot import calibrate


def load_zeroshot(zs_dir, model_tag, category, tag=''):
    # file written by zero_shot.py: one row per item that had an image on disk
    f = Path(zs_dir) / f'zeroshot_{model_tag}{tag}_{category}.npz'
    if not f.exists():
        return None
    # margin > 0  <=>  the model puts more next-token mass on 'Yes' than on 'No'.
    # Only the DIFFERENCE matters for a two-way decision, so we keep just that.
    d = np.load(f)
    out = {'margin': d['yes_logit'] - d['no_logit'],
           'labels': d['labels'].astype(int),
           'question_ids': d['question_ids'] if 'question_ids' in d else None,
           'file': f.name}
    return out


def join_on_question_id(zs, qids, labels, min_coverage=0.9):
    """Line CCS test rows up with zero-shot rows by question_id.

    Returns (keep, zs_rows, status), so ccs_pred[keep] and zs_margin[zs_rows]
    are item-for-item aligned, or (None, None, reason) if the join can't be
    trusted. A positional join would be valid only if zero_shot.py and the
    extractor skipped identical items in identical order; they skip
    independently, so one divergent skip shifts every row silently.

    Unmatched items are dropped and the coverage reported; below min_coverage
    the two artefacts probably aren't the same extraction, so refuse. The label
    cross-check catches the rest.
    """
    if zs['question_ids'] is None:
        return None, None, ('zero-shot file predates question_id logging -- '
                            're-run zero_shot.py to enable the join')
    if qids is None:
        return None, None, ('results JSON has no test_question_ids -- re-run '
                            'reanalysis.py to enable the join')

    index = {int(q): i for i, q in enumerate(zs['question_ids'])}
    keep, rows = [], []
    for k, q in enumerate(qids):
        i = index.get(int(q))
        if i is not None:
            keep.append(k)
            rows.append(i)

    coverage = len(keep) / max(len(qids), 1)
    if coverage < min_coverage:
        return None, None, (f'only {len(keep)}/{len(qids)} ({coverage:.1%}) CCS '
                            f'test items found in zero-shot -- below '
                            f'{min_coverage:.0%}, refusing')

    keep, rows = np.asarray(keep, dtype=int), np.asarray(rows, dtype=int)
    if labels is not None:
        joined = zs['labels'][rows]
        bad = int((joined != np.asarray(labels)[keep]).sum())
        if bad:
            return None, None, (f'label mismatch on {bad}/{len(rows)} joined rows '
                                '-- the two artefacts disagree about the data')
    note = '' if coverage == 1.0 else f' ({len(qids) - len(keep)} items dropped)'
    return keep, rows, f'joined on question_id{note}'


# ============================================================================
# NOTE (review): builds the 2x2 contingency table (both right / only CCS /
# only zero-shot / both wrong), pooled over runs, and a McNemar chi-square
# with continuity correction on the discordant cells b and c.
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results_json')
    ap.add_argument('--zeroshot-dir', default='./zeroshot')
    ap.add_argument('--split', default='ungrouped', choices=['ungrouped', 'grouped'])
    ap.add_argument('--tag', default='',
                    help="zero-shot variant suffix, e.g. '_noinstr' for the "
                         'prompt-matched run written by run_zeroshot.sh')
    ap.add_argument('--min-coverage', type=float, default=0.9,
                    help='refuse a run if fewer than this fraction of its CCS '
                         'test items are present in the zero-shot artefact')
    args = ap.parse_args()

    res = json.loads(Path(args.results_json).read_text())
    # a,b,c,d are the four cells of the 2x2 'who was right' table, pooled over runs:
    #   a = both right   b = only CCS right   c = only zero-shot right   d = both wrong
    # n = items counted, agree = items where CCS and zero-shot gave the SAME answer
    rows, totals = [], {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'n': 0, 'agree': 0}

    for cell, c in res.get('cells', {}).items():
        # cell keys look like 'qwen2/object_detection' (see reanalysis.main)
        model_tag, category = cell.split('/', 1)
        zs = load_zeroshot(args.zeroshot_dir, model_tag, category, args.tag)
        if zs is None:
            print(f'[skip] no zero-shot file for {cell} (tag={args.tag!r})')
            continue

        for run, v in c.get('runs', {}).items():
            # run keys look like 'ungrouped/seed42'; keep only the requested split kind
            if not run.startswith(args.split + '/'):
                continue
            # ccs_test_pred = ORIENTED 0/1 predictions for the test rows;
            # test_question_ids names those rows (reanalysis.run_cell)
            pred = v.get('ccs_test_pred')
            if pred is None:
                print(f'[skip] {cell} {run}: no per-item predictions')
                continue

            keep, zs_rows, status = join_on_question_id(
                zs, v.get('test_question_ids'), v.get('test_labels'),
                args.min_coverage)
            if keep is None:
                print(f'[skip] {cell} {run}: {status}')
                continue

            ccs = np.asarray(pred)
            if len(ccs) != len(v.get('test_question_ids', [])):
                print(f'[skip] {cell} {run}: {len(ccs)} predictions for '
                      f'{len(v.get("test_question_ids", []))} test items')
                continue
            ccs = ccs[keep]

            # restrict zero-shot to the SAME test rows CCS was scored on
            y = zs['labels'][zs_rows]
            m = zs['margin'][zs_rows]
            # calibrated zero-shot: top half of THESE test rows by margin ->
            # yes. Calibrating on the subset keeps the 50/50 rate exact and
            # avoids using items CCS never saw.
            zsp = calibrate(m)

            # boolean masks: which items each method got right
            ccs_ok, zs_ok = (ccs == y), (zsp == y)
            # the 2x2 table for this run (& = elementwise AND, ~ = NOT)
            a = int((ccs_ok & zs_ok).sum())
            b = int((ccs_ok & ~zs_ok).sum())
            cc = int((~ccs_ok & zs_ok).sum())
            d = int((~ccs_ok & ~zs_ok).sum())
            # agreement counts identical PREDICTIONS, right or wrong; two methods can have
            # equal accuracy yet disagree on many items, which is the point of this script
            agree = int((ccs == zsp).sum())
            n = len(y)
            for k, val in (('a', a), ('b', b), ('c', cc), ('d', d),
                           ('n', n), ('agree', agree)):
                totals[k] += val
            # per-run record: accuracy of each method, agreement rate, and the two
            # discordant counts (b = CCS-only right, cc = zero-shot-only right)
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
    # CCS right = a + b; zero-shot right = a + c (read the 2x2 table row/column-wise)
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
        # McNemar's test on paired predictions: only the DISCORDANT cells b and c carry
        # information about which method is better. Statistic with Edwards' continuity
        # correction: chi2 = (|b - c| - 1)^2 / (b + c), ~ chi-square with 1 d.f. under
        # H0 'both methods err equally often'. 3.84 is the 5% critical value (two-sided).
        chi2 = (abs(b - cc) - 1) ** 2 / (b + cc)
        print(f"  McNemar chi2 = {chi2:.2f} on the {b + cc} discordant pairs "
              f"({'significant at p<0.05' if chi2 > 3.84 else 'not significant'})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
