"""Re-analysis of cached VisionCCS hidden states."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

CATEGORIES = ['object_detection', 'attribute_recognition', 'spatial_recognition']
MODEL_TAGS = ['llava', 'qwen2', 'qwen2_5']

_CCS_RE = r'^cache_{cat}_(\d+)_{tag}_ccs_aligned\.npz$'
_SUP_RE = r'^cache_{cat}_(\d+)_supervised_contrast_{tag}\.npz$'


def find_cache(cache_dir, category, model_tag):
    """Locate cache file for a given category and model tag."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return None
    for pattern, kind in ((_CCS_RE, 'ccs'), (_SUP_RE, 'supervised')):
        rx = re.compile(pattern.format(cat=re.escape(category), tag=re.escape(model_tag)))
        hits = [(p, int(rx.match(p.name).group(1))) for p in cache_dir.iterdir()
                if rx.match(p.name)]
        if hits:
            path, n = max(hits, key=lambda t: t[1])
            return path, n, kind
    return None


def load_cache(path):
    """Load cached hidden states and labels from npz file."""
    data = np.load(path)
    pos = data['pos_hiddens'].astype(np.float32)
    neg = data['neg_hiddens'].astype(np.float32)
    labels = data['labels'].astype(int)
    if not (len(pos) == len(neg) == len(labels)):
        raise ValueError(f'{path}: ragged arrays {pos.shape} {neg.shape} {labels.shape}')
    return pos, neg, labels


def build_pairs(vqa_json, category, mode, seed=42):
    """Reconstruct pair ordering from VQA dataset."""
    with open(vqa_json, 'r') as f:
        vqa_data = json.load(f)[category]

    if mode == 'supervised':
        samples = vqa_data[:len(vqa_data)]
    elif mode == 'ccs':
        n_samples = len(vqa_data)
        rng = np.random.default_rng(seed)
        yes_items = [it for it in vqa_data if it['answer'] == 'yes']
        no_items = [it for it in vqa_data if it['answer'] != 'yes']
        n_per_class = min(len(yes_items), len(no_items), n_samples // 2)
        yes_sel = [yes_items[i] for i in rng.permutation(len(yes_items))[:n_per_class]]
        no_sel = [no_items[i] for i in rng.permutation(len(no_items))[:n_per_class]]
        samples = yes_sel + no_sel
        samples = [samples[i] for i in rng.permutation(len(samples))]
    else:
        raise ValueError(f'unknown mode {mode!r}')

    return [{'image_id': it['image_id'],
             'question_id': it['question_id'],
             'label': 1 if it['answer'] == 'yes' else 0}
            for it in samples]


def _image_exists(image_id, image_dirs):
    """Replay find_image() from vision_ccs.py: is this image on disk?"""
    name = f'{image_id:012d}.jpg' if isinstance(image_id, int) else image_id
    return any((Path(d) / name).exists() for d in image_dirs)


def align_pairs(pairs, labels, image_dirs=None):
    """Attach image_ids to cached rows, verified against the label sequence.

    Extraction appends in pair order but SKIPS pairs whose image was missing on
    disk. Those skips are scattered through the shuffled order, not clustered at
    the tail, so a prefix match usually fails once anything was dropped.

    When image_dirs is given we replay the same existence check the extractor
    ran, which reproduces the skip set exactly and recovers the alignment. The
    label sequence still has to match afterwards -- we never guess.
    """
    pair_labels = np.array([p['label'] for p in pairs], dtype=int)

    if len(pair_labels) == len(labels) and np.array_equal(pair_labels, labels):
        return np.array([p['image_id'] for p in pairs]), 'exact'

    if len(labels) < len(pair_labels) and np.array_equal(pair_labels[:len(labels)], labels):
        return np.array([p['image_id'] for p in pairs[:len(labels)]]), 'prefix'

    if image_dirs:
        kept = [p for p in pairs if _image_exists(p['image_id'], image_dirs)]
        kept_labels = np.array([p['label'] for p in kept], dtype=int)
        if len(kept_labels) == len(labels) and np.array_equal(kept_labels, labels):
            n_skip = len(pair_labels) - len(kept_labels)
            return (np.array([p['image_id'] for p in kept]),
                    f'recovered ({n_skip} rows dropped for missing images)')
        return None, (f'MISMATCH after replay (pairs={len(pair_labels)}, '
                      f'on-disk={len(kept_labels)}, cached={len(labels)})')

    return None, (f'MISMATCH (pairs={len(pair_labels)}, cached={len(labels)}) '
                  f'-- pass --image-dirs to recover')


def make_split(n, seed, train_frac, groups=None):
    """Split dataset randomly or grouped by image."""
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(n)
        cut = int(round(n * train_frac))
        return np.sort(perm[:cut]), np.sort(perm[cut:])

    uniq = np.unique(groups)
    gperm = rng.permutation(len(uniq))
    target = int(round(n * train_frac))
    train_groups, count = set(), 0
    for gi in gperm:
        if count >= target:
            break
        g = uniq[gi]
        train_groups.add(g)
        count += int((groups == g).sum())
    mask = np.array([g in train_groups for g in groups])
    return np.where(mask)[0], np.where(~mask)[0]


def normalize(pos_tr, neg_tr, pos_te, neg_te, scheme, var_normalize):
    """Normalize activation arrays."""
    eps = 1e-8

    def stats(x):
        mu = x.mean(axis=0, keepdims=True)
        sd = (x.std(axis=0, keepdims=True) + eps) if var_normalize else np.float32(1.0)
        return mu, sd

    if scheme == 'per_split':
        out = []
        for x in (pos_tr, neg_tr, pos_te, neg_te):
            mu, sd = stats(x)
            out.append((x - mu) / sd)
        return out
    if scheme == 'train_stats':
        pmu, psd = stats(pos_tr)
        nmu, nsd = stats(neg_tr)
        return [(pos_tr - pmu) / psd, (neg_tr - nmu) / nsd,
                (pos_te - pmu) / psd, (neg_te - nmu) / nsd]
    raise ValueError(f'unknown scheme {scheme!r}')


def _randomized_pca(X, k, seed=0, oversample=10, n_iter=4):
    """Top-k principal directions of X via randomized SVD."""
    rng = np.random.default_rng(seed)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    Q, _ = np.linalg.qr(Xc @ rng.normal(size=(Xc.shape[1], min(k + oversample, Xc.shape[1]))))
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(Xc.T @ Q)
        Q, _ = np.linalg.qr(Xc @ Q)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    return mu, Vt[:k].T


def pca_reduce(pos_tr, neg_tr, pos_te, neg_te, k, seed=0):
    """Project activations onto top-k principal components fit on train."""
    mu, W = _randomized_pca(np.concatenate([pos_tr, neg_tr], axis=0), k, seed=seed)
    return tuple(((x - mu) @ W).astype(np.float32)
                 for x in (pos_tr, neg_tr, pos_te, neg_te))


def gaussian_control(pos_tr, neg_tr, pos_te, neg_te, seed=0):
    """Generate Gaussian noise control matching input shapes."""
    rng = np.random.default_rng(seed)
    return tuple(rng.standard_normal(x.shape).astype(np.float32)
                 for x in (pos_tr, neg_tr, pos_te, neg_te))


def auroc(scores, y):
    """Compute rank-based AUROC with tie correction."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return float('nan')

    order = np.argsort(scores, kind='mergesort')
    s = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def score_report(scores, y):
    """Compute accuracy and AUROC metrics."""
    y = np.asarray(y).astype(int)
    preds = (np.asarray(scores) > 0.5).astype(int)
    raw_acc = float((preds == y).mean())
    a = auroc(scores, y)
    pos_m, neg_m = y == 1, y == 0
    flipped = preds if raw_acc >= 0.5 else 1 - preds
    return {
        'raw_acc': raw_acc,
        'flipped_acc': max(raw_acc, 1 - raw_acc),
        'auroc': a,
        'flipped_auroc': max(a, 1 - a) if a == a else float('nan'),
        'acc_pos': float((flipped[pos_m] == y[pos_m]).mean()) if pos_m.any() else float('nan'),
        'acc_neg': float((flipped[neg_m] == y[neg_m]).mean()) if neg_m.any() else float('nan'),
        'n_test': int(len(y)),
        'ci95': 1.96 * float(np.sqrt(max(raw_acc, 1 - raw_acc) * (1 - max(raw_acc, 1 - raw_acc)) / len(y))),
    }


def _probe_and_opt(torch, nn, optim, d, lr, wd, device):
    probe = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid()).to(device)
    return probe, optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)


def probe_diagnostics(p_pos, p_neg):
    """Compute consistency, confidence, and saturation diagnostics."""
    p_pos = np.asarray(p_pos, dtype=float).ravel()
    p_neg = np.asarray(p_neg, dtype=float).ravel()
    both = np.concatenate([p_pos, p_neg])
    return {
        'consistency_err': float(np.abs(p_pos + p_neg - 1.0).mean()),
        'confidence': float(np.minimum(p_pos, p_neg).mean()),
        'saturated': float(((both > 0.99) | (both < 0.01)).mean()),
    }


def train_ccs(pos_tr, neg_tr, pos_te, neg_te, cfg, seed, y_tr=None, y_te=None):
    """Train unsupervised CCS probe."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import copy

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xp = torch.tensor(pos_tr, dtype=torch.float32, device=device)
    Xn = torch.tensor(neg_tr, dtype=torch.float32, device=device)
    Tp = torch.tensor(pos_te, dtype=torch.float32, device=device)
    Tn = torch.tensor(neg_te, dtype=torch.float32, device=device)

    best_loss, best_probe, restarts = float('inf'), None, []
    for t in range(cfg['ntries']):
        torch.manual_seed(seed * 1000 + t)
        probe, opt = _probe_and_opt(torch, nn, optim, Xp.shape[1],
                                    cfg['lr'], cfg['weight_decay'], device)
        perm = torch.randperm(len(Xp), device=device)
        xp, xn = Xp[perm], Xn[perm]
        for _ in range(cfg['epochs']):
            p_pos, p_neg = probe(xp), probe(xn)
            loss = ((p_pos - (1 - p_neg)) ** 2).mean() + (torch.min(p_pos, p_neg) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        fl = float(loss.detach().cpu())

        with torch.no_grad():
            tr_pos, tr_neg = probe(Xp).squeeze(-1), probe(Xn).squeeze(-1)
            te_pos, te_neg = probe(Tp).squeeze(-1), probe(Tn).squeeze(-1)
            s_tr = (0.5 * (tr_pos + (1 - tr_neg))).cpu().numpy()
            s_te = (0.5 * (te_pos + (1 - te_neg))).cpu().numpy()
            rec = {'restart': t, 'loss': fl,
                   **probe_diagnostics(te_pos.cpu().numpy(), te_neg.cpu().numpy())}
            if y_te is not None:
                r = score_report(s_te, y_te)
                rec['test_acc_raw'] = r['raw_acc']
                rec['test_acc_flipped'] = r['flipped_acc']
                rec['test_auroc_flipped'] = r['flipped_auroc']
            if y_tr is not None:
                rec['train_acc_flipped'] = score_report(s_tr, y_tr)['flipped_acc']
            restarts.append(rec)

        if fl < best_loss:
            best_loss, best_probe = fl, copy.deepcopy(probe)

    with torch.no_grad():
        bp, bn = best_probe(Tp).squeeze(-1), best_probe(Tn).squeeze(-1)
        scores = (0.5 * (bp + (1 - bn))).cpu().numpy()
        diag = probe_diagnostics(bp.cpu().numpy(), bn.cpu().numpy())
        train_acc = None
        if y_tr is not None:
            s_tr = (0.5 * (best_probe(Xp).squeeze(-1)
                           + (1 - best_probe(Xn).squeeze(-1)))).cpu().numpy()
            train_acc = score_report(s_tr, y_tr)['flipped_acc']

    return scores, {'best_loss': best_loss, 'restarts': restarts,
                    'train_acc_flipped': train_acc, **diag}


def train_supervised_probe(pos_tr, neg_tr, pos_te, neg_te, y_tr, cfg, seed):
    """Train supervised baseline probe."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import copy

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xp = torch.tensor(pos_tr, dtype=torch.float32, device=device)
    Xn = torch.tensor(neg_tr, dtype=torch.float32, device=device)
    Y = torch.tensor(y_tr, dtype=torch.float32, device=device)
    crit = nn.BCELoss()

    best_loss, best_probe = float('inf'), None
    for t in range(cfg['ntries']):
        torch.manual_seed(seed * 1000 + t)
        probe, opt = _probe_and_opt(torch, nn, optim, Xp.shape[1],
                                    cfg['lr'], cfg['weight_decay'], device)
        for _ in range(cfg['epochs']):
            avg = 0.5 * (probe(Xp).squeeze(-1) + (1 - probe(Xn).squeeze(-1)))
            loss = crit(avg.clamp(1e-6, 1 - 1e-6), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        fl = float(loss.detach().cpu())
        if fl < best_loss:
            best_loss, best_probe = fl, copy.deepcopy(probe)

    with torch.no_grad():
        tp = torch.tensor(pos_te, dtype=torch.float32, device=device)
        tn = torch.tensor(neg_te, dtype=torch.float32, device=device)
        scores = (0.5 * (best_probe(tp) + (1 - best_probe(tn)))).squeeze(-1).cpu().numpy()
    return scores, {'best_loss': best_loss}


def train_logreg(pos_tr, neg_tr, pos_te, neg_te, y_tr, seed):
    """Train logistic regression baseline with parameter sweep on train slice."""
    from sklearn.linear_model import LogisticRegression

    x_tr, x_te = neg_tr - pos_tr, neg_te - pos_te
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_tr))
    cut = int(round(0.8 * len(x_tr)))
    fit_i, val_i = perm[:cut], perm[cut:]

    best = (-1.0, 1.0)
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(class_weight='balanced', max_iter=1000, C=C)
        m.fit(x_tr[fit_i], y_tr[fit_i])
        s = m.score(x_tr[val_i], y_tr[val_i])
        if s > best[0]:
            best = (s, C)
    val_acc, C = best

    m = LogisticRegression(class_weight='balanced', max_iter=1000, C=C)
    m.fit(x_tr, y_tr)
    n_iter = int(np.max(m.n_iter_))
    return m.predict_proba(x_te)[:, 1], {'C': C, 'val_acc': val_acc,
                                         'n_iter': n_iter, 'converged': n_iter < 1000}


def run_cell(pos, neg, labels, image_ids, cfg, seed, grouped, norm_scheme,
             controls=False, pca_k=50):
    """Evaluate all methods on a single configuration cell."""
    groups = image_ids if grouped else None
    tr, te = make_split(len(labels), seed, cfg['train_frac'], groups=groups)
    y_tr, y_te = labels[tr], labels[te]

    out = {'grouped': bool(grouped), 'n_train': int(len(tr)), 'n_test': int(len(te)),
           'test_yes_frac': float(y_te.mean())}
    out['leaked_test_rows'] = (int(np.isin(image_ids[te], image_ids[tr]).sum())
                               if image_ids is not None else None)
    if out['leaked_test_rows'] is not None:
        out['leaked_test_frac'] = out['leaked_test_rows'] / max(len(te), 1)

    def methods(p_tr, n_tr, p_te, n_te, tag_into):
        s, meta = train_ccs(p_tr, n_tr, p_te, n_te, cfg, seed, y_tr=y_tr, y_te=y_te)
        tag_into['ccs'] = {**score_report(s, y_te), **meta}
        s, meta = train_supervised_probe(p_tr, n_tr, p_te, n_te, y_tr, cfg, seed)
        tag_into['sup_probe'] = {**score_report(s, y_te), **meta}
        try:
            s, meta = train_logreg(p_tr, n_tr, p_te, n_te, y_tr, seed)
            tag_into['logreg'] = {**score_report(s, y_te), **meta}
        except ImportError:
            tag_into['logreg'] = {'error': 'sklearn unavailable'}

    raw = normalize(pos[tr], neg[tr], pos[te], neg[te], norm_scheme, cfg['var_normalize'])
    methods(*raw, out)

    if controls:
        out['controls'] = {}
        g = gaussian_control(*raw, seed=seed)
        gn = normalize(*g, scheme=norm_scheme, var_normalize=cfg['var_normalize'])
        out['controls']['gaussian'] = {}
        methods(*gn, out['controls']['gaussian'])

        pc = pca_reduce(pos[tr], neg[tr], pos[te], neg[te], pca_k, seed=seed)
        pcn = normalize(*pc, scheme=norm_scheme, var_normalize=cfg['var_normalize'])
        out['controls'][f'pca{pca_k}'] = {}
        methods(*pcn, out['controls'][f'pca{pca_k}'])

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir', default='./hidden_states_cache_final')
    ap.add_argument('--vqa-json', default='./vqav2_mapped.json')
    ap.add_argument('--models', nargs='+', default=MODEL_TAGS)
    ap.add_argument('--categories', nargs='+', default=CATEGORIES)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42],
                    help='Random seeds to run')
    ap.add_argument('--splits', nargs='+', default=['ungrouped', 'grouped'],
                    choices=['ungrouped', 'grouped'],
                    help='Split configurations')
    ap.add_argument('--controls', action='store_true',
                    help='Run Gaussian noise and PCA controls')
    ap.add_argument('--pca-k', type=int, default=50)
    ap.add_argument('--norm', default='per_split', choices=['per_split', 'train_stats'])
    ap.add_argument('--train-frac', type=float, default=0.6)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--ntries', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--no-var-normalize', action='store_true')
    ap.add_argument('--image-dirs', nargs='+', default=[
        '/scratch-nvme/ml-datasets/coco/train/data',
        '/scratch-nvme/ml-datasets/coco/val/data'],
        help='replayed to recover which pairs the extractor skipped, which is what enables the grouped split (A5)')
    ap.add_argument('--out', default='./reanalysis_results.json')
    args = ap.parse_args()

    cfg = {'train_frac': args.train_frac, 'epochs': args.epochs, 'ntries': args.ntries,
           'lr': args.lr, 'weight_decay': args.weight_decay,
           'var_normalize': not args.no_var_normalize}

    results = {'config': {**cfg, 'norm': args.norm, 'splits': args.splits,
                          'controls': args.controls, 'pca_k': args.pca_k,
                          'seeds': args.seeds, 'cache_dir': args.cache_dir}, 'cells': {}}

    for model_tag in args.models:
        for category in args.categories:
            found = find_cache(args.cache_dir, category, model_tag)
            if found is None:
                print(f'[skip] no cache for {model_tag}/{category} in {args.cache_dir}')
                continue
            path, _, kind = found
            pos, neg, labels = load_cache(path)

            pairs = build_pairs(args.vqa_json, category,
                                mode='ccs' if kind == 'ccs' else 'supervised')
            image_ids, status = align_pairs(pairs, labels, args.image_dirs)

            print(f'\n=== {model_tag} / {category} ===')
            print(f'  cache {path.name}')
            print(f'  n={len(labels)}  d={pos.shape[1]}  yes_frac={labels.mean():.3f}  '
                  f'image_id alignment: {status}')

            splits = list(args.splits)
            if image_ids is None and 'grouped' in splits:
                print('  [warning] alignment failed -> skipping grouped split')
                splits = [s for s in splits if s != 'grouped']

            key = f'{model_tag}/{category}'
            results['cells'][key] = {'cache': path.name, 'kind': kind,
                                     'n': int(len(labels)), 'd': int(pos.shape[1]),
                                     'alignment': status, 'runs': {}}

            for split_kind in splits:
                for seed in args.seeds:
                    cell = run_cell(pos, neg, labels, image_ids, cfg, seed,
                                    split_kind == 'grouped', args.norm,
                                    controls=args.controls, pca_k=args.pca_k)
                    results['cells'][key]['runs'][f'{split_kind}/seed{seed}'] = cell
                    _print_cell(split_kind, seed, cell, args.pca_k)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f'\nWrote {args.out}')
    _print_summary(results, args.pca_k)


def _fmt(m, key='raw_acc'):
    return 'n/a' if 'error' in m else f'{m[key]:.1%}'


def _print_cell(split_kind, seed, cell, pca_k):
    leak = ('' if cell.get('leaked_test_frac') is None
            else f"  leak={cell['leaked_test_frac']:.1%}")
    print(f"  [{split_kind:9s} seed {seed}] n_test={cell['n_test']}{leak}")
    c = cell['ccs']
    print(f"      CCS        raw {c['raw_acc']:.1%}  flip {c['flipped_acc']:.1%}"
          f"  auroc {c['flipped_auroc']:.3f}  loss {c['best_loss']:.2e}"
          f"  train {c['train_acc_flipped']:.1%}"
          f"  sat {c['saturated']:.0%}  consist {c['consistency_err']:.3f}")
    print(f"      sup_probe  raw {_fmt(cell['sup_probe'])}"
          f"      logreg  raw {_fmt(cell['logreg'])}"
          + ('' if 'error' in cell['logreg']
             else f"  (C={cell['logreg']['C']}, converged={cell['logreg']['converged']})"))
    for name in ('gaussian', f'pca{pca_k}'):
        ctl = cell.get('controls', {}).get(name)
        if ctl:
            print(f"      [{name:9s}] CCS flip {ctl['ccs']['flipped_acc']:.1%}"
                  f"  loss {ctl['ccs']['best_loss']:.2e}"
                  f"  sup_probe {_fmt(ctl['sup_probe'])}"
                  f"  logreg {_fmt(ctl['logreg'])}")


def _print_summary(results, pca_k):
    """Print comparison summary table."""
    print('\n' + '=' * 100)
    print('SUMMARY  (mean +/- std over seeds; CCS shown as flipped acc, '
          'baselines as raw acc)')
    print('=' * 100)
    hdr = f"{'cell':34s} {'split':10s} {'CCS':>15s} {'sup_probe':>15s} {'logreg':>15s} {'CCS auroc':>12s}"
    print(hdr)
    print('-' * len(hdr))

    def agg(runs, path):
        vals = []
        for r in runs:
            node = r
            for p in path[:-1]:
                node = node.get(p, {})
            if 'error' in node or path[-1] not in node:
                return None
            vals.append(node[path[-1]])
        return (float(np.mean(vals)), float(np.std(vals))) if vals else None

    def cellstr(a, pct=True):
        if a is None:
            return f"{'n/a':>15s}"
        return f"{a[0]:.1%} +/-{a[1]:.1%}" if pct else f"{a[0]:.3f} +/-{a[1]:.3f}"

    for key, c in results['cells'].items():
        for split_kind in ('ungrouped', 'grouped'):
            runs = [v for k, v in c['runs'].items() if k.startswith(split_kind + '/')]
            if not runs:
                continue
            print(f"{key:34s} {split_kind:10s} "
                  f"{cellstr(agg(runs, ['ccs', 'flipped_acc'])):>15s} "
                  f"{cellstr(agg(runs, ['sup_probe', 'raw_acc'])):>15s} "
                  f"{cellstr(agg(runs, ['logreg', 'raw_acc'])):>15s} "
                  f"{cellstr(agg(runs, ['ccs', 'flipped_auroc']), pct=False):>12s}")

    xs, ys = [], []
    for c in results['cells'].values():
        for r in c['runs'].values():
            for rec in r['ccs'].get('restarts', []):
                if 'test_acc_flipped' in rec:
                    xs.append(rec['loss'])
                    ys.append(rec['test_acc_flipped'])
    if len(xs) > 2 and np.std(xs) > 0 and np.std(ys) > 0:
        rho = float(np.corrcoef(np.log(np.maximum(xs, 1e-12)), ys)[0, 1])
        print(f"\ncorr(log final loss, test acc) over {len(xs)} restarts: {rho:+.3f}")


if __name__ == '__main__':
    main()
