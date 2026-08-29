"""Unit tests for reanalysis.py."""

import json
import sys
import tempfile
import pathlib

import numpy as np

import reanalysis as R

VQA = 'vqav2_mapped.json'

_ok = True


def check(name, cond, extra=''):
    global _ok
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {extra}' if extra else ''))
    _ok = _ok and bool(cond)


def _martin_load(cat, seed=42):
    vqa = json.load(open(VQA))[cat]
    n_samples = len(vqa)
    rng = np.random.default_rng(seed)
    yes = [i for i in vqa if i['answer'] == 'yes']
    no = [i for i in vqa if i['answer'] != 'yes']
    npc = min(len(yes), len(no), n_samples // 2)
    sel = ([yes[i] for i in rng.permutation(len(yes))[:npc]] +
           [no[i] for i in rng.permutation(len(no))[:npc]])
    return [sel[i] for i in rng.permutation(len(sel))]


def test_pair_reconstruction():
    for cat in R.CATEGORIES:
        mine, ref = R.build_pairs(VQA, cat, 'ccs'), _martin_load(cat)
        same = (len(mine) == len(ref) and
                all(m['question_id'] == r['question_id'] for m, r in zip(mine, ref)))
        check(f'build_pairs ccs order == vision_ccs.py ({cat})', same, f'n={len(mine)}')

    sup = R.build_pairs(VQA, 'object_detection', 'supervised')
    raw = json.load(open(VQA))['object_detection']
    check('build_pairs supervised == file prefix',
          [p['question_id'] for p in sup] == [i['question_id'] for i in raw])


def test_alignment():
    pairs = R.build_pairs(VQA, 'object_detection', 'ccs')
    lab = np.array([p['label'] for p in pairs])

    ids, st = R.align_pairs(pairs, lab)
    check('align exact', st == 'exact' and len(ids) == len(lab))

    ids, st = R.align_pairs(pairs, lab[:500])
    check('align prefix', st == 'prefix' and len(ids) == 500)

    bad = lab[:500].copy()
    bad[0] = 1 - bad[0]
    ids, st = R.align_pairs(pairs, bad)
    check('align mismatch rejection', ids is None and st.startswith('MISMATCH'), st)


def test_auroc():
    check('auroc perfect', R.auroc(np.array([.1, .2, .8, .9]), [0, 0, 1, 1]) == 1.0)
    check('auroc inverted', R.auroc(np.array([.9, .8, .2, .1]), [0, 0, 1, 1]) == 0.0)
    check('auroc all ties = 0.5', R.auroc(np.array([.5] * 4), [0, 0, 1, 1]) == 0.5)
    a = R.auroc(np.array([0.1, 0.4, 0.4, 0.6, 0.9]), [0, 0, 1, 1, 1])
    check('auroc tie-corrected', abs(a - (1 + 0.5 + 1 + 1 + 1 + 1) / 6) < 1e-12, f'{a:.4f}')
    check('auroc single-class -> nan', R.auroc(np.array([.1, .2]), [1, 1]) != R.auroc(np.array([.1, .2]), [1, 1]))


def test_splits():
    n = 1000
    tr, te = R.make_split(n, 42, 0.6)
    check('split sizes', len(tr) == 600 and len(te) == 400)
    check('split disjoint + complete', set(tr).isdisjoint(te) and len(set(tr) | set(te)) == n)
    check('split deterministic', np.array_equal(R.make_split(n, 42, 0.6)[0], tr))
    check('split seed-sensitive', not np.array_equal(R.make_split(n, 7, 0.6)[0], tr))

    g = np.repeat(np.arange(200), 5)
    tr, te = R.make_split(len(g), 1, 0.6, groups=g)
    check('group split: zero image leakage', len(set(g[tr]) & set(g[te])) == 0)
    check('group split: ~60% train', 0.5 < len(tr) / len(g) < 0.7, f'{len(tr)/len(g):.2f}')
    check('group split disjoint + complete',
          set(tr).isdisjoint(te) and len(set(tr) | set(te)) == len(g))


def test_normalize():
    rng = np.random.default_rng(0)
    ptr, ntr, pte, nte = (rng.normal(3, 2, (60, 8)).astype('f4') for _ in range(4))

    a, b, c, d = R.normalize(ptr, ntr, pte, nte, 'per_split', True)
    check('per_split centers every split', all(abs(x.mean()) < 1e-5 for x in (a, b, c, d)))
    check('per_split gives unit per-column std',
          all(abs(x.std(0).mean() - 1) < 1e-3 for x in (a, b, c, d)))

    a, _, c, _ = R.normalize(ptr, ntr, pte, nte, 'train_stats', True)
    check('train_stats centers train, not test', abs(a.mean()) < 1e-5 and abs(c.mean()) > 1e-3)

    a, _, _, _ = R.normalize(ptr, ntr, pte, nte, 'per_split', False)
    check('var_normalize=False preserves per-column std',
          np.abs(a.std(axis=0) - ptr.std(axis=0)).max() < 1e-5)
    check('var_normalize=False is pure centering',
          np.abs(a - (ptr - ptr.mean(axis=0, keepdims=True))).max() == 0.0)


def test_find_cache():
    with tempfile.TemporaryDirectory() as td:
        t = pathlib.Path(td)
        (t / 'cache_object_detection_1306_qwen2_ccs_aligned.npz').touch()
        (t / 'cache_object_detection_1323_supervised_contrast_qwen2.npz').touch()
        (t / 'cache_object_detection_1306_qwen2_5_ccs_aligned.npz').touch()

        p, n, k = R.find_cache(t, 'object_detection', 'qwen2')
        check('find_cache prefers ccs cache', k == 'ccs' and n == 1306, p.name)
        check('find_cache does not confuse qwen2 with qwen2_5',
              R.find_cache(t, 'object_detection', 'qwen2_5')[0].name
              .startswith('cache_object_detection_1306_qwen2_5'))
        check('find_cache missing -> None',
              R.find_cache(t, 'spatial_recognition', 'llava') is None)


def test_pca_control():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(400, 20)) @ rng.normal(size=(20, 300)) + 0.05 * rng.normal(size=(400, 300))
    _, W = R._randomized_pca(Z, 20, seed=1)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    sv = np.linalg.svd(W.T @ Vt[:20].T, compute_uv=False)
    check('randomized PCA recovers subspace', sv.min() > 0.999, f'min sv={sv.min():.6f}')

    ptr, ntr, pte, nte = (rng.normal(size=(120, 300)).astype('f4') for _ in range(4))
    a, b, c, d = R.pca_reduce(ptr, ntr, pte, nte, 50)
    check('pca_reduce shapes', all(x.shape == (120, 50) for x in (a, b, c, d)))
    check('pca_reduce dtype float32', all(x.dtype == np.float32 for x in (a, b, c, d)))
    a2, _, _, _ = R.pca_reduce(ptr, ntr, pte + 99.0, nte, 50)
    check('pca_reduce is fit on train only', np.abs(a - a2).max() == 0.0)


def test_gaussian_control():
    rng = np.random.default_rng(0)
    xs = tuple(rng.normal(size=(40, 60)).astype('f4') for _ in range(4))
    g = R.gaussian_control(*xs, seed=3)
    check('gaussian_control preserves shape/dtype',
          all(a.shape == b.shape and a.dtype == np.float32 for a, b in zip(g, xs)))
    check('gaussian_control replaces features', np.abs(g[0] - xs[0]).max() > 0.1)


def test_diagnostics():
    d = R.probe_diagnostics(np.array([0.999, 0.998]), np.array([0.001, 0.002]))
    check('consistent probe -> ~0 consistency error', d['consistency_err'] < 0.01,
          f"{d['consistency_err']:.4f}")
    check('saturation detected on both branches', d['saturated'] == 1.0)

    d = R.probe_diagnostics(np.array([0.9, 0.9]), np.array([0.9, 0.9]))
    check('inconsistent probe -> large error', abs(d['consistency_err'] - 0.8) < 1e-9)
    check('confidence term', abs(d['confidence'] - 0.9) < 1e-9)
    check('mid-range outputs not saturated', d['saturated'] == 0.0)

    d = R.probe_diagnostics(np.array([0.999, 0.999]), np.array([0.5, 0.5]))
    check('one-sided saturation detected', d['saturated'] == 0.5, f"{d['saturated']}")


def test_score_report():
    y = np.array([0, 0, 1, 1])
    r = R.score_report(np.array([0.9, 0.8, 0.2, 0.1]), y)
    check('score_report exposes raw and flipped', r['raw_acc'] == 0.0 and r['flipped_acc'] == 1.0)
    check('score_report class accs', r['acc_pos'] == 1.0 and r['acc_neg'] == 1.0)


if __name__ == '__main__':
    for fn in (test_pair_reconstruction, test_alignment, test_auroc, test_splits,
               test_normalize, test_find_cache, test_score_report,
               test_pca_control, test_gaussian_control, test_diagnostics):
        print(f'\n-- {fn.__name__} --')
        fn()
    print('\n' + ('ALL PASS' if _ok else 'FAILURES PRESENT'))
    sys.exit(0 if _ok else 1)
