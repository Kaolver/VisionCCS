"""Report tables and the layer figure from the final layer sweeps.

Reads every sweep_*.json that final_jobs/ wrote (layer_sweep.py output) and
writes, into --out-dir:

  final_tables.md     headline table, per-category table, shuffled-image
                      control, and the layer curves as numbers
  final_results.csv   every (model, dataset, category, layer, seed) row
  layer_curves.png    accuracy vs layer per model and dataset (if matplotlib)

    python final_summary.py --results-dir ./final_results --out-dir ./final_report

Every method in a row is scored on the SAME test questions (grouped 60/40
split, one per seed): zero-shot comes from layer_sweep --zeroshot-dir, joined
on question_id, so no number here compares different question sets.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

MODELS = {'llava': 'LLaVA-1.5-7B', 'qwen2': 'Qwen2-VL-7B', 'qwen2_5': 'Qwen2.5-VL-7B'}
DATASETS = {'vqa': 'VQAv2', 'vg': 'Visual Genome'}
SHORT = {'object_detection': 'object', 'attribute_recognition': 'attribute',
         'spatial_recognition': 'spatial'}
VG_CATEGORIES = {'object', 'attribute', 'spatial'}
COLUMNS = ['object', 'attribute', 'spatial']
POSITION = 'final'          # the token CCS has always been read at


def load_rows(results_dir):
    """One row per (sweep file, category, layer, seed)."""
    rows = []
    for f in sorted(Path(results_dir).glob('sweep_*.json')):
        d = json.loads(f.read_text())
        shuffled = bool(d['config'].get('shuffled'))
        for key, cell in d['cells'].items():
            model, cat = key.split('/', 1)
            dataset = 'vg' if cat in VG_CATEGORIES else 'vqa'
            for gk, seeds in cell['grid'].items():
                pos, layer = gk.split('/L')
                if pos != POSITION:
                    continue
                for seed, r in seeds.items():
                    zs = r.get('zeroshot', {})
                    matched = [v for t, v in zs.items() if t.endswith('_noinstr')]
                    instr = [v for t, v in zs.items() if not t.endswith('_noinstr')]
                    lr = r.get('logreg', {})
                    rows.append({
                        'model': model, 'dataset': dataset, 'shuffled': shuffled,
                        'category': SHORT.get(cat, cat), 'layer': int(layer),
                        'seed': int(seed), 'n_test': r['n_test'],
                        'ccs': r['ccs']['flipped_acc'],
                        'val_consistency_err': r['ccs'].get('val_consistency_err', np.nan),
                        'sup_probe': r['sup_probe']['raw_acc'],
                        'logreg': lr.get('raw_acc', np.nan),
                        'logreg_C': lr.get('C', np.nan),
                        'logreg_converged': lr.get('converged', ''),
                        'zs_matched': matched[0]['cal_acc'] if matched else np.nan,
                        'zs_instr': instr[0]['cal_acc'] if instr else np.nan,
                        'zs_matched_raw': matched[0]['raw_acc'] if matched else np.nan,
                    })
    return rows


def cell_means(rows):
    """(model, dataset, shuffled, category, layer) -> metric means over seeds."""
    groups = {}
    for r in rows:
        k = (r['model'], r['dataset'], r['shuffled'], r['category'], r['layer'])
        groups.setdefault(k, []).append(r)
    out = {}
    for k, rs in groups.items():
        m = {f: float(np.nanmean([r[f] for r in rs])) if any(
                 np.isfinite(r[f]) for r in rs) else np.nan
             for f in ('ccs', 'sup_probe', 'logreg', 'zs_matched', 'zs_instr',
                       'zs_matched_raw', 'val_consistency_err')}
        m['ccs_sd'] = float(np.std([r['ccs'] for r in rs]))
        m['n_test'] = int(np.mean([r['n_test'] for r in rs]))
        m['seeds'] = len(rs)
        out[k] = m
    return out


def pct(x, sd=None):
    if x is None or not np.isfinite(x):
        return '-'
    return f'{100 * x:.1f}' + (f' ± {100 * sd:.1f}' if sd is not None and sd > 0 else '')


def diff(a, b):
    return '-' if not (np.isfinite(a) and np.isfinite(b)) else f'{100 * (a - b):+.1f}'


def summarise(cells):
    """Per (model, dataset, shuffled): last layer, label-free pick, oracle."""
    combos = sorted({k[:3] for k in cells})
    out = {}
    for model, dataset, shuffled in combos:
        cats = [c for c in COLUMNS if any(k[:4] == (model, dataset, shuffled, c)
                                          for k in cells)]
        per_cat = {}
        for c in cats:
            layers = sorted(k[4] for k in cells if k[:4] == (model, dataset, shuffled, c))
            byl = {l: cells[(model, dataset, shuffled, c, l)] for l in layers}
            last = layers[-1]
            # label-free choice: lowest held-out-train consistency error; the
            # oracle (highest test accuracy) is shown only to price that choice
            finite = [l for l in layers if np.isfinite(byl[l]['val_consistency_err'])]
            pick = min(finite, key=lambda l: byl[l]['val_consistency_err']) if finite else last
            oracle = max(layers, key=lambda l: byl[l]['ccs'])
            per_cat[c] = {'last': last, 'pick': pick, 'oracle': oracle, 'byl': byl}
        out[(model, dataset, shuffled)] = per_cat
    return out


def avg(per_cat, which, field):
    vals = [pc['byl'][pc[which]][field] for pc in per_cat.values()]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if len(vals) == len(per_cat) and vals else np.nan


def tables(summary):
    L = []
    real = {k: v for k, v in summary.items() if not k[2]}

    L += ['## 1. Headline: zero-shot vs CCS vs supervised (same test questions)', '',
          'Accuracy %, mean of the three categories. CCS at the last layer is the '
          'setup of the original report; "picked layer" is chosen per category '
          'without labels (lowest consistency error on a held-out slice of train).', '',
          '| model | dataset | zero-shot (CCS prompt) | zero-shot (+instruction) '
          '| CCS last layer | CCS picked layer | CCS − zero-shot | logistic regression '
          '(last layer) | best logistic regression (any layer) |',
          '|---|---|---|---|---|---|---|---|---|']
    for (model, dataset, _), pc in sorted(real.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        zs = avg(pc, 'last', 'zs_matched')
        ccs_last = avg(pc, 'last', 'ccs')
        ccs_pick = avg(pc, 'pick', 'ccs')
        lr_last = avg(pc, 'last', 'logreg')
        lr_best = np.mean([np.nanmax([v['logreg'] for v in p['byl'].values()])
                           for p in pc.values()])
        picks = '/'.join(f"L{p['pick']}" for p in pc.values())
        L.append(f'| {MODELS[model]} | {DATASETS[dataset]} | {pct(zs)} | '
                 f"{pct(avg(pc, 'last', 'zs_instr'))} | {pct(ccs_last)} | "
                 f'{pct(ccs_pick)} ({picks}) | {diff(max(ccs_last, ccs_pick), zs)} | '
                 f'{pct(lr_last)} | {pct(lr_best)} |')
    L += ['', '"CCS − zero-shot" uses the better of the two CCS columns, so it is '
          'generous to CCS. Logistic regression chooses its regularisation on a '
          'held-out slice of train; "best (any layer)" uses test labels to pick '
          'the layer and is an upper bound, not a result.', '']

    L += ['## 2. Per category, last layer', '',
          '| model | dataset | category | n test | zero-shot | CCS | supervised probe '
          '| logistic regression |', '|---|---|---|---|---|---|---|---|']
    for (model, dataset, _), pc in sorted(real.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        for c, p in pc.items():
            m = p['byl'][p['last']]
            L.append(f"| {MODELS[model]} | {DATASETS[dataset]} | {c} | {m['n_test']} | "
                     f"{pct(m['zs_matched'])} | {pct(m['ccs'], m['ccs_sd'])} | "
                     f"{pct(m['sup_probe'])} | {pct(m['logreg'])} |")
    L.append('')

    shuf = {k: v for k, v in summary.items() if k[2]}
    if shuf:
        L += ['## 3. Shuffled-image control', '',
              'Each question is paired with a different image. If accuracy stays '
              'high, the probe is reading a language prior, not the image.', '',
              '| model | dataset | CCS last layer: real → shuffled | CCS picked layer: '
              'real → shuffled | logistic regression: real → shuffled |',
              '|---|---|---|---|---|']
        for (model, dataset, _), pcs in sorted(shuf.items()):
            pcr = summary.get((model, dataset, False))
            if not pcr:
                continue
            L.append(f"| {MODELS[model]} | {DATASETS[dataset]} | "
                     f"{pct(avg(pcr, 'last', 'ccs'))} → {pct(avg(pcs, 'last', 'ccs'))} | "
                     f"{pct(avg(pcr, 'pick', 'ccs'))} → {pct(avg(pcs, 'pick', 'ccs'))} | "
                     f"{pct(avg(pcr, 'last', 'logreg'))} → {pct(avg(pcs, 'last', 'logreg'))} |")
        L.append('')

    L += ['## 4. Accuracy by layer (mean of the three categories)', '']
    for (model, dataset, shuffled), pc in sorted(summary.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        layers = sorted(next(iter(pc.values()))['byl'])
        L += [f"### {MODELS[model]}, {DATASETS[dataset]}{' (shuffled images)' if shuffled else ''}",
              '', '| layer | CCS | supervised probe | logistic regression |', '|---|---|---|---|']
        for l in layers:
            row = {f: np.mean([p['byl'][l][f] for p in pc.values() if l in p['byl']])
                   for f in ('ccs', 'sup_probe', 'logreg')}
            L.append(f"| {l} | {pct(row['ccs'])} | {pct(row['sup_probe'])} | {pct(row['logreg'])} |")
        zs = avg(pc, 'last', 'zs_matched')
        if np.isfinite(zs):
            L.append(f'\nZero-shot (CCS prompt, same test questions): {pct(zs)}')
        L.append('')
    return '\n'.join(L)


def figure(summary, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available; skipping the figure')
        return False
    real = sorted({(m, d) for (m, d, s) in summary if not s},
                  key=lambda md: (list(DATASETS).index(md[1]), list(MODELS).index(md[0])))
    if not real:
        return False
    ds = [d for d in DATASETS if any(md[1] == d for md in real)]
    ms = [m for m in MODELS if any(md[0] == m for md in real)]
    fig, axes = plt.subplots(len(ds), len(ms), figsize=(4.2 * len(ms), 3.4 * len(ds)),
                             squeeze=False, sharey=True)
    for i, d in enumerate(ds):
        for j, m in enumerate(ms):
            ax = axes[i][j]
            pc = summary.get((m, d, False))
            if not pc:
                ax.axis('off')
                continue
            layers = sorted(next(iter(pc.values()))['byl'])
            def curve(f, which=pc):
                return [np.mean([p['byl'][l][f] for p in which.values() if l in p['byl']])
                        for l in layers]
            ax.plot(layers, curve('ccs'), 'o-', label='CCS (unsupervised)')
            ax.plot(layers, curve('logreg'), 's-', label='logistic regression (supervised)')
            zs = avg(pc, 'last', 'zs_matched')
            if np.isfinite(zs):
                ax.axhline(zs, color='k', ls='--', lw=1, label='zero-shot (calibrated)')
            sh = summary.get((m, d, True))
            if sh:
                ax.plot(layers, curve('ccs', sh), 'x:', label='CCS, shuffled images')
            ax.axhline(0.5, color='grey', lw=0.5)
            ax.set_title(f'{MODELS[m]}, {DATASETS[d]}', fontsize=10)
            ax.set_xlabel('layer')
            if j == 0:
                ax.set_ylabel('accuracy')
            ax.set_ylim(0.45, 1.0)
            ax.grid(alpha=0.3)
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default='./final_results')
    ap.add_argument('--out-dir', default='./final_report')
    args = ap.parse_args()
    # the tables use ± and →; never let a non-UTF-8 console abort the run
    sys.stdout.reconfigure(errors='replace')

    rows = load_rows(args.results_dir)
    if not rows:
        print(f'No sweep_*.json results in {args.results_dir}', file=sys.stderr)
        return 1
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    with open(out / 'final_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary = summarise(cell_means(rows))
    md = tables(summary)
    (out / 'final_tables.md').write_text(md, encoding='utf-8')
    print(md)
    made = figure(summary, out / 'layer_curves.png')
    print(f"\nWrote {out / 'final_tables.md'}, {out / 'final_results.csv'}"
          + (f", {out / 'layer_curves.png'}" if made else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
