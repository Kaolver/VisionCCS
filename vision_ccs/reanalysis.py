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


def align_pairs(pairs, labels):
    """Attach image_ids to cached rows based on label sequence alignment."""
    pair_labels = np.array([p['label'] for p in pairs], dtype=int)

    if len(pair_labels) == len(labels) and np.array_equal(pair_labels, labels):
        return np.array([p['image_id'] for p in pairs]), 'exact'

    if len(labels) < len(pair_labels) and np.array_equal(pair_labels[:len(labels)], labels):
        return np.array([p['image_id'] for p in pairs[:len(labels)]]), 'prefix'

    return None, f'MISMATCH (pairs={len(pair_labels)}, cached={len(labels)})'


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


def train_ccs(pos_tr, neg_tr, pos_te, neg_te, cfg, seed):
    """Train unsupervised CCS probe."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import copy

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xp = torch.tensor(pos_tr, dtype=torch.float32, device=device)
    Xn = torch.tensor(neg_tr, dtype=torch.float32, device=device)

    best_loss, best_probe, losses = float('inf'), None, []
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
        losses.append(fl)
        if fl < best_loss:
            best_loss, best_probe = fl, copy.deepcopy(probe)

    with torch.no_grad():
        tp = torch.tensor(pos_te, dtype=torch.float32, device=device)
        tn = torch.tensor(neg_te, dtype=torch.float32, device=device)
        scores = (0.5 * (best_probe(tp) + (1 - best_probe(tn)))).squeeze(-1).cpu().numpy()
    return scores, {'best_loss': best_loss, 'restart_losses': losses}


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


def run_cell(pos, neg, labels, image_ids, cfg, seed, group_split, norm_scheme):
    groups = image_ids if (group_split and image_ids is not None) else None
    tr, te = make_split(len(labels), seed, cfg['train_frac'], groups=groups)
    p_tr, n_tr, p_te, n_te = normalize(pos[tr], neg[tr], pos[te], neg[te],
                                       norm_scheme, cfg['var_normalize'])
    y_tr, y_te = labels[tr], labels[te]

    out = {'n_train': int(len(tr)), 'n_test': int(len(te)),
           'test_yes_frac': float(y_te.mean())}
    if groups is not None:
        out['leaked_test_rows'] = 0
    else:
        out['leaked_test_rows'] = (int(np.isin(image_ids[te], image_ids[tr]).sum())
                                   if image_ids is not None else None)

    s, meta = train_ccs(p_tr, n_tr, p_te, n_te, cfg, seed)
    out['ccs'] = {**score_report(s, y_te), **meta}

    s, meta = train_supervised_probe(p_tr, n_tr, p_te, n_te, y_tr, cfg, seed)
    out['sup_probe'] = {**score_report(s, y_te), **meta}

    try:
        s, meta = train_logreg(p_tr, n_tr, p_te, n_te, y_tr, seed)
        out['logreg'] = {**score_report(s, y_te), **meta}
    except ImportError:
        out['logreg'] = {'error': 'sklearn unavailable'}

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir', default='./hidden_states_cache_final')
    ap.add_argument('--vqa-json', default='./vqav2_mapped.json')
    ap.add_argument('--models', nargs='+', default=MODEL_TAGS)
    ap.add_argument('--categories', nargs='+', default=CATEGORIES)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--group-split', action='store_true',
                    help='keep all questions about one image on one side')
    ap.add_argument('--norm', default='per_split', choices=['per_split', 'train_stats'])
    ap.add_argument('--train-frac', type=float, default=0.6)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--ntries', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--no-var-normalize', action='store_true')
    ap.add_argument('--out', default='./reanalysis_results.json')
    args = ap.parse_args()

    cfg = {'train_frac': args.train_frac, 'epochs': args.epochs, 'ntries': args.ntries,
           'lr': args.lr, 'weight_decay': args.weight_decay,
           'var_normalize': not args.no_var_normalize}

    results = {'config': {**cfg, 'norm': args.norm, 'group_split': args.group_split,
                          'seeds': args.seeds, 'cache_dir': args.cache_dir}, 'cells': {}}

    for model_tag in args.models:
        for category in args.categories:
            found = find_cache(args.cache_dir, category, model_tag)
            if found is None:
                print(f'[skip] no cache for {model_tag}/{category} in {args.cache_dir}')
                continue
            path, n_named, kind = found
            pos, neg, labels = load_cache(path)

            pairs = build_pairs(args.vqa_json, category,
                                mode='ccs' if kind == 'ccs' else 'supervised')
            image_ids, status = align_pairs(pairs, labels)

            print(f'\n=== {model_tag} / {category} ===')
            print(f'  cache {path.name}  n={len(labels)} d={pos.shape[1]} '
                  f'yes={labels.mean():.3f}')
            print(f'  image_id alignment: {status}')
            if image_ids is None and args.group_split:
                print('  [warning] group split requested but alignment failed -> '
                      'falling back to ungrouped for this cell')

            key = f'{model_tag}/{category}'
            results['cells'][key] = {'cache': path.name, 'kind': kind,
                                     'n': int(len(labels)), 'd': int(pos.shape[1]),
                                     'alignment': status, 'seeds': {}}

            for seed in args.seeds:
                cell = run_cell(pos, neg, labels, image_ids, cfg, seed,
                                args.group_split, args.norm)
                results['cells'][key]['seeds'][str(seed)] = cell
                lr_bit = ''
                if 'error' not in cell['logreg']:
                    lr_bit = (f"  logreg {cell['logreg']['raw_acc']:.1%} "
                              f"(C={cell['logreg']['C']}, "
                              f"conv={cell['logreg']['converged']})")
                print(f"  seed {seed}: "
                      f"CCS raw {cell['ccs']['raw_acc']:.1%} / "
                      f"flip {cell['ccs']['flipped_acc']:.1%} / "
                      f"auroc {cell['ccs']['flipped_auroc']:.3f}   "
                      f"sup_probe {cell['sup_probe']['raw_acc']:.1%}{lr_bit}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
