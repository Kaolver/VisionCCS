"""CCS, logistic regression and zero-shot as specified by Burns et al. (2022).

Where the paper and ccs/utils.py differ, the paper is followed:
  normalization  per prompt and split, centred, then average norm sqrt(d) (G.1)
  CCS            lr 0.01 (code default 1e-3); otherwise as in the code
  LR             on the concatenation (phi(x0), phi(x1)) (code: the difference)
"""
import numpy as np


def _torch():
    import torch
    return torch


def normalize(x, scheme='paper'):
    x = x - x.mean(0, keepdims=True)
    if scheme == 'mean':
        return x
    if scheme == 'var':
        return x / (x.std(0, keepdims=True) + 1e-8)
    if scheme == 'paper':
        return x * np.sqrt(x.shape[1]) / (np.linalg.norm(x, axis=1).mean() + 1e-8)
    raise ValueError(scheme)


def normalize_by_group(x, groups, scheme):
    out = np.empty_like(x, dtype=np.float32)
    for g in np.unique(groups):
        m = groups == g
        out[m] = normalize(x[m].astype(np.float32), scheme)
    return out


def train_ccs(x0, x1, ntries=10, epochs=1000, lr=1e-2, weight_decay=0.01,
              seed=0, device='cpu'):
    """All restarts as one (ntries, d) probe; AdamW is elementwise, so they stay independent."""
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    d = x0.shape[1]
    bound = 1.0 / np.sqrt(d)  # nn.Linear default init
    W = ((torch.rand(ntries, d, generator=g) * 2 - 1) * bound).to(device).requires_grad_()
    b = ((torch.rand(ntries, generator=g) * 2 - 1) * bound).to(device).requires_grad_()
    X0 = torch.as_tensor(x0, dtype=torch.float32, device=device)
    X1 = torch.as_tensor(x1, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        p0 = torch.sigmoid(X0 @ W.T + b)
        p1 = torch.sigmoid(X1 @ W.T + b)
        loss = (torch.min(p0, p1) ** 2).mean(0) + ((p0 - (1 - p1)) ** 2).mean(0)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    losses = loss.detach().cpu().numpy()
    best = int(np.argmin(losses))
    return W.detach().cpu().numpy(), b.detach().cpu().numpy(), losses, best


def ccs_scores(x0, x1, w, b):
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    return 0.5 * (sig(x1 @ w + b) + 1 - sig(x0 @ w + b))


def fit_logreg(X, y, C=1.0, max_iter=1000, device='cpu'):
    """sklearn LogisticRegression(class_weight='balanced', C=C) via torch L-BFGS."""
    torch = _torch()
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    n = len(y)
    n1 = float(yt.sum())
    s = torch.where(yt > 0.5, n / (2 * max(n1, 1)), n / (2 * max(n - n1, 1)))
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-6, tolerance_change=1e-9,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = X @ w + b
        ll = torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction='none')
        # sklearn's objective divided by C * n
        obj = (s * ll).sum() / n + (w @ w) / (2 * C * n)
        obj.backward()
        return obj

    opt.step(closure)
    return w.detach().cpu().numpy(), float(b.detach().cpu().numpy()[0])


def zero_shot(lp0, lp1, labels, templates):
    """(raw, calibrated) accuracy, averaged over prompts."""
    raw, cal = [], []
    for t in np.unique(templates):
        m = templates == t
        margin = lp1[m] - lp0[m]
        raw.append(np.mean((margin > 0) == labels[m]))
        # rank split rather than a median threshold, which is not 50/50 on ties
        top = np.zeros(len(margin), dtype=bool)
        top[np.argsort(margin, kind='stable')[len(margin) // 2:]] = True
        cal.append(np.mean(top == labels[m]))
    return float(np.mean(raw)), float(np.mean(cal))


def split_items(groups, seed, train_frac=0.6):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_tr = int(round(train_frac * len(uniq)))
    tr_groups = set(uniq[perm[:n_tr]].tolist())
    tr = np.array([g in tr_groups for g in groups])
    return tr, ~tr


def per_template_acc(pred, labels, templates):
    accs = [float(np.mean(pred[templates == t] == labels[templates == t]))
            for t in np.unique(templates)]
    return float(np.mean(accs)), float(np.std(accs)), accs


def evaluate_cell(h0, h1, labels, groups, templates, seed, cfg, device='cpu'):
    """CCS, LR and tuned-C LR for one (layer, position, seed). groups is the split unit."""
    tr, te = split_items(groups, seed, cfg.get('train_frac', 0.6))
    norm = cfg.get('norm', 'paper')
    x0tr = normalize_by_group(h0[tr], templates[tr], norm)
    x1tr = normalize_by_group(h1[tr], templates[tr], norm)
    x0te = normalize_by_group(h0[te], templates[te], norm)
    x1te = normalize_by_group(h1[te], templates[te], norm)
    ytr, yte, tte = labels[tr], labels[te], templates[te]
    res = {'n_train_rows': int(tr.sum()), 'n_test_rows': int(te.sum())}

    W, B, losses, best = train_ccs(x0tr, x1tr, cfg.get('ntries', 10), cfg.get('epochs', 1000),
                                   cfg.get('lr', 1e-2), cfg.get('weight_decay', 0.01),
                                   seed=seed, device=device)
    pred = ccs_scores(x0te, x1te, W[best], B[best]) > 0.5
    # max(acc, 1 - acc), one orientation per dataset shared by all prompts
    if np.mean(pred == yte) < 0.5:
        pred = ~pred
    m, sd, per = per_template_acc(pred, yte, tte)
    res['ccs'] = {'acc': m, 'sd_prompts': sd, 'per_prompt': per,
                  'loss': float(losses[best]), 'losses': losses.tolist()}

    if not cfg.get('skip_lr'):
        Xtr = np.concatenate([x0tr, x1tr], 1)
        Xte = np.concatenate([x0te, x1te], 1)
        w, b = fit_logreg(Xtr, ytr, 1.0, device=device)
        m, sd, _ = per_template_acc((Xte @ w + b) > 0, yte, tte)
        res['lr'] = {'acc': m, 'sd_prompts': sd, 'train_acc': float(np.mean(((Xtr @ w + b) > 0) == ytr))}

        # C is chosen on 20% of the train items, never on test
        g_tr = groups[tr]
        fit, val = split_items(g_tr, seed + 1000, 0.8)
        best_c, best_v = 1.0, -1.0
        for C in cfg.get('c_grid', (0.001, 0.01, 0.1, 1.0, 10.0)):
            w, b = fit_logreg(Xtr[fit], ytr[fit], C, device=device)
            v = float(np.mean(((Xtr[val] @ w + b) > 0) == ytr[val]))
            if v > best_v:
                best_c, best_v = C, v
        w, b = fit_logreg(Xtr, ytr, best_c, device=device)
        m, sd, _ = per_template_acc((Xte @ w + b) > 0, yte, tte)
        res['lr_tuned'] = {'acc': m, 'sd_prompts': sd, 'C': best_c, 'val_acc': best_v}
    return res, te
