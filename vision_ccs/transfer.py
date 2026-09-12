"""Cross-category transfer: fit a probe on one category, evaluate on another.

A category-specific classifier does not transfer; a truth direction should.
Every number so far was fit and evaluated inside one category, which cannot tell
the two apart.

The diagonal is the within-category number. Read the off-diagonal: near the
diagonal means the direction is category-general, near chance means CCS fit the
category. CRC-TPC transfers too -- if it drops with CCS, the structure was in
the contrast pairs.

    python transfer.py --cache-dir ./caches_v3 --model qwen2 --layer 14
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from reanalysis import (CATEGORIES, MODEL_TAGS, build_pairs, find_cache,
                        load_cache, align_pairs, pairs_field, make_split,
                        normalize, score_report, train_ccs, train_pca_tpc,
                        train_kmeans_diff, train_supervised_probe)


def load_v3(cache_dir, model_tag, category, layer, position, templates):
    """Load one (layer, position) slice out of an extract.py cache."""
    from layer_sweep import find_cache_v3, load_cache_v3, cell_arrays
    path = find_cache_v3(cache_dir, model_tag, category, False, templates)
    if path is None:
        return None
    c = load_cache_v3(path)
    li = c['layers'].index(layer) if layer in c['layers'] else len(c['layers']) - 1
    pi = c['positions'].index(position) if position in c['positions'] else 0
    pos, neg = cell_arrays(c, li, pi)
    return {'pos': pos, 'neg': neg, 'labels': c['labels'],
            'image_ids': c['image_ids'], 'source': path.name,
            'layer': c['layers'][li], 'position': c['positions'][pi]}


def load_v1(cache_dir, vqa_json, model_tag, category, image_dirs):
    """Load a v1 (final-layer) cache and recover its image_ids for grouping."""
    found = find_cache(cache_dir, model_tag, category)
    if found is None:
        return None
    path, _, kind = found
    pos, neg, labels = load_cache(path)
    pairs = build_pairs(vqa_json, category, mode='ccs' if kind == 'ccs' else 'supervised')
    kept, status = align_pairs(pairs, labels, image_dirs)
    return {'pos': pos, 'neg': neg, 'labels': labels,
            'image_ids': pairs_field(kept, 'image_id'), 'source': path.name,
            'layer': -1, 'position': 'final', 'alignment': status}


def transfer_cell(fit, ev, cfg, seed, norm_scheme, grouped, same_category):
    """Fit on `fit`'s train split, evaluate on `ev`'s test split.

    Each category is normalized with its own statistics, as Burns does; pushing
    the fit category's mean onto the eval category would measure distribution
    shift rather than the direction's generality.

    Same category means the same split on both sides, or train rows leak.
    """
    fg = fit['image_ids'] if grouped else None
    eg = ev['image_ids'] if grouped else None
    tr, te_same = make_split(len(fit['labels']), seed, cfg['train_frac'], groups=fg)
    if same_category:
        te = te_same
    else:
        _, te = make_split(len(ev['labels']), seed, cfg['train_frac'], groups=eg)

    y_tr, y_te = fit['labels'][tr], ev['labels'][te]
    p_tr, n_tr, _, _ = normalize(fit['pos'][tr], fit['neg'][tr],
                                 fit['pos'][tr], fit['neg'][tr],
                                 norm_scheme, cfg['var_normalize'],
                                 cluster_k=cfg.get('cluster_k', 8), seed=seed)
    _, _, p_te, n_te = normalize(ev['pos'][te], ev['neg'][te],
                                 ev['pos'][te], ev['neg'][te],
                                 norm_scheme, cfg['var_normalize'],
                                 cluster_k=cfg.get('cluster_k', 8), seed=seed)

    out = {'n_fit': int(len(tr)), 'n_eval': int(len(te))}
    s, meta = train_ccs(p_tr, n_tr, p_te, n_te, cfg, seed, y_tr=y_tr, y_te=y_te)
    meta.pop('restarts', None)
    out['ccs'] = {**score_report(s, y_te), **meta}
    for name, fn in (('crc_tpc', train_pca_tpc), ('kmeans_diff', train_kmeans_diff)):
        s, m = fn(p_tr, n_tr, p_te, n_te, seed=seed)
        out[name] = {**score_report(s, y_te), **m}
    s, _ = train_supervised_probe(p_tr, n_tr, p_te, n_te, y_tr, cfg, seed)
    out['sup_probe'] = score_report(s, y_te)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', default='./caches_v3')
    ap.add_argument('--cache-version', default='v3', choices=['v1', 'v3'],
                    help="v3 = extract.py all-layer caches; v1 = the original "
                         "final-layer hidden_states_cache_* caches")
    ap.add_argument('--vqa-json', default='./vqav2_mapped.json')
    ap.add_argument('--model', default='qwen2', choices=MODEL_TAGS)
    ap.add_argument('--categories', nargs='+', default=CATEGORIES)
    ap.add_argument('--templates', nargs='+', default=['plain'])
    ap.add_argument('--layer', type=int, default=-1,
                    help='v3 only: layer index to slice (default: last stored)')
    ap.add_argument('--position', default='answer',
                    help='v3 only: token position to slice')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 1, 2])
    ap.add_argument('--grouped', action='store_true', default=True)
    ap.add_argument('--ungrouped', dest='grouped', action='store_false')
    ap.add_argument('--norm', default='per_split',
                    choices=['per_split', 'train_stats', 'cluster'])
    ap.add_argument('--cluster-k', type=int, default=8)
    ap.add_argument('--no-var-normalize', action='store_true')
    ap.add_argument('--selection', default='val_consistency',
                    choices=['loss', 'val_consistency', 'test_consistency'])
    ap.add_argument('--weight-norm', default='none', choices=['none', 'unit'])
    ap.add_argument('--train-frac', type=float, default=0.6)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--ntries', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--image-dirs', nargs='+', default=[
        '/scratch-nvme/ml-datasets/coco/train/data',
        '/scratch-nvme/ml-datasets/coco/val/data'])
    ap.add_argument('--out', default='./transfer.json')
    args = ap.parse_args()

    cfg = {'train_frac': args.train_frac, 'epochs': args.epochs,
           'ntries': args.ntries, 'lr': args.lr, 'weight_decay': args.weight_decay,
           'var_normalize': not args.no_var_normalize,
           'selection': args.selection, 'val_frac': args.val_frac,
           'weight_norm': args.weight_norm, 'cluster_k': args.cluster_k,
           'skip_logreg': True, 'skip_baselines': True}

    caches = {}
    for cat in args.categories:
        if args.cache_version == 'v3':
            c = load_v3(args.cache_dir, args.model, cat, args.layer,
                        args.position, args.templates)
        else:
            c = load_v1(args.cache_dir, args.vqa_json, args.model, cat,
                        args.image_dirs)
        if c is None:
            print(f'[skip] no cache for {args.model}/{cat} in {args.cache_dir}')
            continue
        if args.grouped and c['image_ids'] is None:
            print(f'[warn] {cat}: no image_ids -> falling back to ungrouped')
        caches[cat] = c
        print(f'{cat:24s} {c["source"]}  n={len(c["labels"])}  '
              f'layer={c["layer"]} pos={c["position"]}')

    if len(caches) < 2:
        print('\nNeed at least two categories with caches to measure transfer.')
        return 1

    grouped = args.grouped and all(c['image_ids'] is not None for c in caches.values())
    results = {'config': {**cfg, 'norm': args.norm, 'model': args.model,
                          'layer': args.layer, 'position': args.position,
                          'templates': args.templates, 'grouped': grouped,
                          'seeds': args.seeds}, 'cells': {}}

    cats = list(caches)
    for fit_cat in cats:
        for eval_cat in cats:
            runs = {}
            for seed in args.seeds:
                runs[str(seed)] = transfer_cell(
                    caches[fit_cat], caches[eval_cat], cfg, seed, args.norm,
                    grouped, fit_cat == eval_cat)
            results['cells'][f'{fit_cat}->{eval_cat}'] = runs
            m = np.mean([r['ccs']['flipped_acc'] for r in runs.values()])
            print(f'  fit {fit_cat:24s} -> eval {eval_cat:24s}  CCS {m:.1%}')

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f'\nWrote {args.out}')
    _print_matrix(results, cats, 'ccs')
    _print_matrix(results, cats, 'crc_tpc')
    _print_matrix(results, cats, 'sup_probe')
    _summarise(results, cats)
    return 0


def _cellval(results, fit_cat, eval_cat, method):
    runs = results['cells'].get(f'{fit_cat}->{eval_cat}')
    if not runs:
        return float('nan')
    key = 'raw_acc' if method == 'sup_probe' else 'flipped_acc'
    return float(np.mean([r[method][key] for r in runs.values()]))


def _print_matrix(results, cats, method):
    print('\n' + '=' * 78)
    print(f'{method.upper()} transfer matrix   (rows = fit on, cols = evaluated on)')
    print('=' * 78)
    corner = 'fit \\ eval'
    print(f'  {corner:24s}' + ''.join(f'{c[:14]:>16s}' for c in cats))
    for fit_cat in cats:
        row = ''.join(f'{_cellval(results, fit_cat, c, method):>15.1%} ' for c in cats)
        print(f'  {fit_cat:24s}{row}')


def _summarise(results, cats):
    """Diagonal vs off-diagonal."""
    print('\n' + '=' * 78)
    print(f"  {'method':14s} {'within-category':>17s} {'transferred':>13s} {'drop':>8s}")
    print('  ' + '-' * 56)
    for method in ('ccs', 'crc_tpc', 'sup_probe'):
        diag = [_cellval(results, c, c, method) for c in cats]
        off = [_cellval(results, a, b, method) for a in cats for b in cats if a != b]
        diag = [v for v in diag if v == v]
        off = [v for v in off if v == v]
        if not diag or not off:
            continue
        d, o = float(np.mean(diag)), float(np.mean(off))
        print(f'  {method:14s} {d:16.1%} {o:12.1%} {o - d:+7.1%}')
    print('\n  A transferred score near chance means the probe learned the')
    print('  category, not a truth direction. Compare CCS against CRC-TPC: if')
    print('  they drop together, the structure was in the contrast pairs.')


if __name__ == '__main__':
    sys.exit(main())
