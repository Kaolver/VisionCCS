"""Phase 2.4: CCS accuracy as a function of LAYER and token POSITION.

Reads the Phase 1 caches from extract.py, shaped (n, n_layers, n_positions, d),
and runs the existing matched-protocol analysis at every (layer, position) cell.

Why this is the experiment that matters
---------------------------------------
Everything before this probed hidden_states[-1] -- the final layer, whose job is
producing next-token logits, and which zero-shot IS once pushed through the LM
head. Probing it and comparing against zero-shot largely compares the readout
layer with itself, and the measured 90.9% item-level agreement is close to what
that design guarantees. CCS's claim is about INTERNAL representations, and the
probing literature puts truth-related structure in the middle-to-late layers.
If CCS ever exceeds zero-shot, that is where it happens.

Reporting rule (pre-committed)
------------------------------
The FULL curve is reported, never the best layer. At n_test ~ 450 with +/-3pp
CIs, taking the max over ~15 layers buys several points from noise alone -- the
exact selection bias this project's finding #4 is about, turned on ourselves. If
a single layer must be named, it is chosen by the label-free val_consistency
criterion or on a held-out category, never by test accuracy. --pick-layer does
that honestly and reports what the choice cost against the oracle layer.

Two conveniences over the v1 path: image_ids are stored in the cache, so grouped
splits need no alignment replay; and the shuffled-image control is just another
cache (--shuffled), so it sweeps identically.

    python layer_sweep.py --cache-dir ./caches_v3 --model qwen2
"""

# ============================================================================
# NOTE (review): ROLE OF THIS FILE
# Runs the reanalysis protocol at every (layer, position) cell of an extract.py
# v3 cache. Reuses make_split / normalize / train_* from reanalysis.py.
# NOTE POTENTIAL MISMATCH: default --selection here is 'val_consistency',
# whereas reanalysis.py defaults to 'loss' (the original CCS rule).
# ============================================================================
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from reanalysis import (CATEGORIES, make_split, normalize, score_report,
                        train_ccs, train_supervised_probe, train_logreg)


def find_cache_v3(cache_dir, model_tag, category, shuffled=False, templates=None,
                  distractor='none'):
    """Mirror the suffix extract.py builds, in the same order."""
    suffix = '_shuffled' if shuffled else ''
    if distractor != 'none':
        suffix += f'_distract-{distractor}'
    if templates and list(templates) != ['plain']:
        suffix += '_t' + '-'.join(templates)
    p = Path(cache_dir) / f'hs_{model_tag}_{category}{suffix}.npz'
    return p if p.exists() else None


def load_cache_v3(path):
    """Load a Phase 1 cache. Arrays stay float16 until a cell is sliced out."""
    # np.load is lazy for .npz: arrays are only read when indexed below
    d = np.load(path)
    need = ('pos_hiddens', 'neg_hiddens', 'labels', 'layers', 'positions')
    missing = [k for k in need if k not in d]
    if missing:
        raise ValueError(f'{path}: missing keys {missing} -- is this a v3 cache?')
    pos, neg = d['pos_hiddens'], d['neg_hiddens']
    # v3 layout is (n items, n layers kept, n positions, hidden dim d)
    if pos.ndim != 4:
        raise ValueError(f'{path}: expected (n, layers, positions, d), got {pos.shape}')
    if pos.shape != neg.shape:
        raise ValueError(f'{path}: pos {pos.shape} != neg {neg.shape}')
    labels = d['labels'].astype(int)
    if len(labels) != pos.shape[0]:
        raise ValueError(f'{path}: {len(labels)} labels for {pos.shape[0]} rows')
    return {'pos': pos, 'neg': neg, 'labels': labels,
            'layers': [int(x) for x in d['layers']],
            'positions': [str(x) for x in d['positions']],
            'image_ids': d['image_ids'] if 'image_ids' in d else None,
            'question_ids': d['question_ids'] if 'question_ids' in d else None,
            'template_ids': d['template_ids'] if 'template_ids' in d else None,
            'templates': ([str(x) for x in d['templates']]
                          if 'templates' in d else ['plain']),
            'distractor_labels': (d['distractor_labels'].astype(int)
                                  if 'distractor_labels' in d else None),
            'distractor': str(d['distractor']) if 'distractor' in d else 'none'}


def split_groups(cache, group_by):
    """Grouping key for make_split, or None for an ungrouped split.

    Multi-template caches hold one row per (item, template), so an ungrouped
    split can put an item in train and its paraphrase in test. Grouping by
    image_id also keeps template siblings together, so it is the safe default.
    """
    if group_by == 'none':
        return None
    key = {'image': 'image_ids', 'question': 'question_ids'}[group_by]
    g = cache.get(key)
    if g is None:
        raise ValueError(f'cache has no {key}; cannot group by {group_by}')
    return g


def cell_arrays(cache, li, pi):
    """Slice one (layer, position) out as float32 (n, d) for pos and neg."""
    # select one layer index li and one position index pi -> (n, d) float32;
    # the float16 -> float32 cast happens only for the cell being analysed
    return (cache['pos'][:, li, pi, :].astype(np.float32),
            cache['neg'][:, li, pi, :].astype(np.float32))


# ============================================================================
# NOTE (review): one seed at one cell. 'restarts' is dropped from the CCS meta
# to keep the sweep JSON small, so select_criteria.py cannot be run on a
# layer-sweep output.
# ============================================================================
def run_one(pos, neg, labels, image_ids, cfg, seed, grouped, norm_scheme,
            skip_logreg):
    # identical protocol to reanalysis.run_cell, minus the controls
    groups = image_ids if grouped else None

def run_one(pos, neg, labels, image_ids, cfg, seed, groups, norm_scheme,
            skip_logreg, distractor_labels=None):
    tr, te = make_split(len(labels), seed, cfg['train_frac'], groups=groups)
    y_tr, y_te = labels[tr], labels[te]
    p_tr, n_tr, p_te, n_te = normalize(pos[tr], neg[tr], pos[te], neg[te],
                                       norm_scheme, cfg['var_normalize'],
                                       cluster_k=cfg.get('cluster_k', 8), seed=seed)
    out = {'n_train': int(len(tr)), 'n_test': int(len(te))}
    if image_ids is not None:
        # share of test rows whose IMAGE also occurs in train (0 for a grouped split)
        out['leaked_test_frac'] = float(np.isin(image_ids[te], image_ids[tr]).mean())

    s, meta = train_ccs(p_tr, n_tr, p_te, n_te, cfg, seed, y_tr=y_tr, y_te=y_te)
    # consequence: select_criteria.py cannot be run on a layer-sweep JSON.
    # train_ccs still carries the selected restart's val-consistency, which
    # --pick-layer needs.
    meta.pop('restarts', None)          # keep the sweep JSON small
    out['ccs'] = {**score_report(s, y_te), **meta}

    # Same CCS predictions, scored against the banner. Chance is the only honest
    # outcome; above it, the probe locked onto the banner rather than truth.
    if distractor_labels is not None:
        d_te = np.asarray(distractor_labels)[te]
        if 0 < d_te.sum() < len(d_te):
            r = score_report(s, d_te)
            out['ccs_vs_distractor'] = {'flipped_acc': r['flipped_acc'],
                                        'flipped_auroc': r['flipped_auroc'],
                                        'n_test': r['n_test']}

    s, _ = train_supervised_probe(p_tr, n_tr, p_te, n_te, y_tr, cfg, seed)
    out['sup_probe'] = score_report(s, y_te)
    if not skip_logreg:
        try:
            s, m = train_logreg(p_tr, n_tr, p_te, n_te, y_tr, seed)
            out['logreg'] = {**score_report(s, y_te), **m}
        except ImportError:
            out['logreg'] = {'error': 'sklearn unavailable'}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', default='./caches_v3')
    ap.add_argument('--model', default='qwen2')
    ap.add_argument('--categories', nargs='+', default=CATEGORIES)
    ap.add_argument('--shuffled', action='store_true',
                    help='use the shuffled-image control caches')
    ap.add_argument('--distractor', default='none', choices=['none', 'banner'],
                    help='load the banner-distractor caches instead')
    ap.add_argument('--positions', nargs='+', default=None,
                    help='subset of stored positions (default: all)')
    ap.add_argument('--layers', nargs='+', type=int, default=None,
                    help='subset of stored layer indices (default: all)')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 1, 2])
    ap.add_argument('--grouped', action='store_true',
                    help='shorthand for --group-by image')
    ap.add_argument('--group-by', default=None,
                    choices=['none', 'image', 'question'],
                    help='split unit. Defaults to image when the cache holds more '
                         'than one template (paraphrase siblings must not straddle '
                         'the split), otherwise none.')
    ap.add_argument('--templates', nargs='+', default=['plain'],
                    help='template set identifying the cache to load')
    ap.add_argument('--cluster-k', type=int, default=8)
    ap.add_argument('--selection', default='val_consistency',
                    choices=['loss', 'val_consistency', 'test_consistency'])
    ap.add_argument('--weight-norm', default='none', choices=['none', 'unit'])
    ap.add_argument('--norm', default='per_split',
                    choices=['per_split', 'train_stats', 'cluster'])
    ap.add_argument('--no-var-normalize', action='store_true')
    ap.add_argument('--skip-logreg', action='store_true')
    ap.add_argument('--train-frac', type=float, default=0.6)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--ntries', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--pick-layer', action='store_true',
                    help='additionally report the layer chosen label-free (lowest '
                         'val consistency) and what that choice cost vs the oracle')
    ap.add_argument('--pick-criterion', default='val_consistency_err',
                    choices=['val_consistency_err', 'consistency_err'],
                    help='val_* is label-free AND inductive (default); '
                         'consistency_err is label-free but transductive')
    ap.add_argument('--out', default='./layer_sweep.json')
    args = ap.parse_args()

    cfg = {'train_frac': args.train_frac, 'epochs': args.epochs,
           'ntries': args.ntries, 'lr': args.lr, 'weight_decay': args.weight_decay,
           'var_normalize': not args.no_var_normalize,
           'selection': args.selection, 'val_frac': args.val_frac,
           'weight_norm': args.weight_norm, 'skip_logreg': args.skip_logreg,
           'cluster_k': args.cluster_k}

    results = {'config': {**cfg, 'norm': args.norm, 'grouped': args.grouped,
                          'group_by': args.group_by, 'templates': args.templates,
                          'shuffled': args.shuffled, 'seeds': args.seeds,
                          'model': args.model}, 'cells': {}}

    for category in args.categories:
        path = find_cache_v3(args.cache_dir, args.model, category, args.shuffled,
                             args.templates, args.distractor)
        if path is None:
            print(f'[skip] no v3 cache for {args.model}/{category} '
                  f'{"(shuffled)" if args.shuffled else ""} in {args.cache_dir}')
            continue
        cache = load_cache_v3(path)
        layers, positions = cache['layers'], cache['positions']

        # Resolved once per cache so a multi-template cache can't silently fall
        # back to an ungrouped split.
        group_by = args.group_by
        if group_by is None:
            group_by = 'image' if len(cache['templates']) > 1 else (
                'image' if args.grouped else 'none')
        elif args.grouped and group_by == 'none':
            raise SystemExit('--grouped and --group-by none contradict each other')
        if len(cache['templates']) > 1 and group_by == 'none':
            raise SystemExit(
                f'{path.name} holds {len(cache["templates"])} templates per item; '
                'an ungrouped split would put paraphrases of the same item on '
                'both sides. Pass --group-by image (or question).')
        groups = split_groups(cache, group_by)
        print(f'  templates {cache["templates"]}  split unit: {group_by}')

        # map requested layer NUMBERS / position NAMES to indices into the cache axes
        li_sel = [i for i, l in enumerate(layers)
                  if args.layers is None or l in args.layers]
        pi_sel = [i for i, p in enumerate(positions)
                  if args.positions is None or p in args.positions]

        print(f'\n{"=" * 78}')
        print(f'{args.model} / {category}{"  [SHUFFLED CONTROL]" if args.shuffled else ""}')
        print(f'  {path.name}  n={len(cache["labels"])}  d={cache["pos"].shape[-1]}')
        print(f'  layers {[layers[i] for i in li_sel]}')
        print(f'  positions {[positions[i] for i in pi_sel]}')
        print('=' * 78)

        # grid entries are keyed 'position/L<layer>' -> {seed: run_one result}
        key = f'{args.model}/{category}'
        results['cells'][key] = {'file': path.name, 'n': int(len(cache['labels'])),
                                 'layers': layers, 'positions': positions,
                                 'templates': cache['templates'],
                                 'group_by': group_by, 'grid': {}}

        for pi in pi_sel:
            pos_name = positions[pi]
            print(f'\n  position = {pos_name}')
            dcol = ' vs banner' if cache['distractor_labels'] is not None else ''
            print(f"    {'layer':>6s} {'CCS':>16s} {'sup_probe':>16s} "
                  f"{'loss':>10s}{dcol}")
            for li in li_sel:
                P, N = cell_arrays(cache, li, pi)
                # per cell: repeat over seeds and report mean +/- std
                # (std = spread over seeds)
                accs, sups, losses, dis = [], [], [], []
                for seed in args.seeds:
                    r = run_one(P, N, cache['labels'], cache['image_ids'], cfg,
                                seed, groups, args.norm, args.skip_logreg,
                                cache['distractor_labels'])
                    accs.append(r['ccs']['flipped_acc'])
                    sups.append(r['sup_probe']['raw_acc'])
                    losses.append(r['ccs']['best_loss'])
                    if 'ccs_vs_distractor' in r:
                        dis.append(r['ccs_vs_distractor']['flipped_acc'])
                    results['cells'][key]['grid'].setdefault(
                        f'{pos_name}/L{layers[li]}', {})[str(seed)] = r
                a, s_ = np.array(accs), np.array(sups)
                extra = f' {np.mean(dis):9.1%}' if dis else ''
                print(f'    {layers[li]:6d} {a.mean():7.1%}+/-{a.std():5.1%} '
                      f'{s_.mean():7.1%}+/-{s_.std():5.1%} '
                      f'{np.mean(losses):10.2e}{extra}')

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f'\nWrote {args.out}')

    if args.pick_layer:
        _report_picked_layer(results, args.pick_criterion)
    return 0


# ============================================================================
# NOTE (review): a previous revision ranked cells by 'consistency_err', which
# train_ccs measures on the TEST inputs -- label-free but TRANSDUCTIVE, and
# at odds with the module docstring's promise of 'val_consistency'. The
# criterion is now a parameter defaulting to 'val_consistency_err', the
# held-out train-slice value, which is label-free AND inductive.
# ============================================================================
def _report_picked_layer(results, criterion='val_consistency_err'):
    """Name a layer WITHOUT looking at test accuracy, and price the choice.

    Default criterion is val consistency, measured on the held-out train slice:
    label-free AND inductive. `consistency_err` is also label-free but
    transductive -- it needs the test inputs to make the choice.

    The oracle row prices the honest choice; it is not a result.
    """
    print('\n' + '=' * 88)
    print(f'LABEL-FREE LAYER CHOICE (selected on {criterion}, not accuracy)')
    print('=' * 88)
    print(f"  {'cell':34s} {'picked':>10s} {'its acc':>9s} {'oracle':>10s} {'cost':>7s}")
    for key, c in results['cells'].items():
        rows = []
        for gk, seeds in c['grid'].items():
            acc = np.mean([s['ccs']['flipped_acc'] for s in seeds.values()])
            con = np.mean([s['ccs'].get(criterion, np.nan) for s in seeds.values()])
            rows.append((gk, acc, con))
        if not rows:
            continue
        # r[2] == r[2] is the idiom for 'not NaN' (NaN is the only value != itself)
        finite = [r for r in rows if r[2] == r[2]]
        if not finite:
            print(f'  {key:34s} (no {criterion} values recorded -- '
                  f'was --val-frac set to 0?)')
            continue
        # label-free pick: lowest consistency error;  oracle: highest accuracy (labels);
        # cost = picked accuracy - oracle accuracy (<= 0, how much honesty costs)
        picked = min(finite, key=lambda r: r[2])
        oracle = max(rows, key=lambda r: r[1])
        print(f'  {key:34s} {picked[0]:>10s} {picked[1]:8.1%} '
              f'{oracle[0]:>10s} {picked[1] - oracle[1]:+6.1%}')
    print('\n  The full curve above is the result; this row exists so a single')
    print('  number can be quoted without selecting on the test set.')


if __name__ == '__main__':
    sys.exit(main())
