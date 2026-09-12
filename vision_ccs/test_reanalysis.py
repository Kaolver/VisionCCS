"""Unit tests for reanalysis.py."""

# ============================================================================
# NOTE (review): ROLE OF THIS FILE
# Self-contained checks (no pytest). Run from vision_ccs/ with
# `python test_reanalysis.py`; needs vqav2_mapped.json in the cwd and numpy.
# Tests needing torch/sklearn are absent: train_* functions are untested.
# ============================================================================
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
    # minimal test runner: print PASS/FAIL, remember any failure in the global _ok
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {extra}' if extra else ''))
    _ok = _ok and bool(cond)


# ============================================================================
# NOTE (review): a verbatim copy of the balanced sampler in
# vision_ccs.load_vqa_data, used as the reference that build_pairs(mode='ccs')
# must reproduce order-for-order. If vision_ccs.py's sampler changes, update
# this copy or the test silently tests the wrong thing.
# ============================================================================
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

    kept, st = R.align_pairs(pairs, lab)
    check('align exact', st == 'exact' and len(kept) == len(lab))

    kept, st = R.align_pairs(pairs, lab[:500])
    check('align prefix', st == 'prefix' and len(kept) == 500)

    # flip one label so the checksum comparison must reject the alignment
    bad = lab[:500].copy()
    bad[0] = 1 - bad[0]
    kept, st = R.align_pairs(pairs, bad)
    check('align mismatch rejection', kept is None and st.startswith('MISMATCH'), st)

    kept, _ = R.align_pairs(pairs, lab)
    qids = R.pairs_field(kept, 'question_id')
    iids = R.pairs_field(kept, 'image_id')
    check('pairs_field returns both join keys',
          len(qids) == len(lab) and len(iids) == len(lab))
    check('question_ids are unique (usable as a join key)',
          len(set(int(q) for q in qids)) == len(qids))
    check('pairs_field(None) -> None', R.pairs_field(None, 'question_id') is None)


def test_auroc():
    check('auroc perfect', R.auroc(np.array([.1, .2, .8, .9]), [0, 0, 1, 1]) == 1.0)
    check('auroc inverted', R.auroc(np.array([.9, .8, .2, .1]), [0, 0, 1, 1]) == 0.0)
    check('auroc all ties = 0.5', R.auroc(np.array([.5] * 4), [0, 0, 1, 1]) == 0.5)
    a = R.auroc(np.array([0.1, 0.4, 0.4, 0.6, 0.9]), [0, 0, 1, 1, 1])
    # hand-computed: 2 negatives x 3 positives = 6 pairs; the .4 vs .4 tie counts 1/2
    check('auroc tie-corrected', abs(a - (1 + 0.5 + 1 + 1 + 1 + 1) / 6) < 1e-12, f'{a:.4f}')
    # x != x is only true for NaN, hence this is a 'returns nan' check
    check('auroc single-class -> nan', R.auroc(np.array([.1, .2]), [1, 1]) != R.auroc(np.array([.1, .2]), [1, 1]))


def test_splits():
    n = 1000
    tr, te = R.make_split(n, 42, 0.6)
    check('split sizes', len(tr) == 600 and len(te) == 400)
    check('split disjoint + complete', set(tr).isdisjoint(te) and len(set(tr) | set(te)) == n)
    check('split deterministic', np.array_equal(R.make_split(n, 42, 0.6)[0], tr))
    check('split seed-sensitive', not np.array_equal(R.make_split(n, 7, 0.6)[0], tr))

    # 200 images x 5 questions each -> grouped split must keep an image on one side
    g = np.repeat(np.arange(200), 5)
    tr, te = R.make_split(len(g), 1, 0.6, groups=g)
    check('group split: zero image leakage', len(set(g[tr]) & set(g[te])) == 0)
    check('group split: ~60% train', 0.5 < len(tr) / len(g) < 0.7, f'{len(tr)/len(g):.2f}')
    check('group split disjoint + complete',
          set(tr).isdisjoint(te) and len(set(tr) | set(te)) == len(g))


def test_normalize():
    rng = np.random.default_rng(0)
    # mean 3, std 2 so that centring and scaling are both visible in the checks
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


def test_cluster_norm():
    rng = np.random.default_rng(0)
    # Two clusters separated by a large offset: the distracting feature.
    d = 12
    off = np.zeros(d, dtype='f4'); off[0] = 50.0
    def make(n):
        g = rng.integers(0, 2, n)
        base = rng.normal(0, 1, (n, d)).astype('f4')
        return (base + np.outer(g, off)).astype('f4')
    ptr, ntr, pte, nte = make(200), make(200), make(80), make(80)

    a, b, c, e = R.cluster_normalize(ptr, ntr, pte, nte, 2, True, seed=0)
    check('cluster_normalize preserves shapes',
          a.shape == ptr.shape and c.shape == pte.shape)
    check('cluster_normalize suppresses the between-cluster offset',
          a[:, 0].std() < 2.0, f'std={a[:, 0].std():.3f} (raw {ptr[:, 0].std():.1f})')

    a2, _, _, _ = R.cluster_normalize(ptr, ntr, pte + 1000.0, nte, 2, True, seed=0)
    check('cluster_normalize is fit on train only',
          np.abs(a - a2).max() == 0.0)

    single, _, _, _ = R.cluster_normalize(ptr, ntr, pte, nte, 1, True, seed=0)
    plain = R.normalize(ptr, ntr, pte, nte, 'train_stats', True)[0]
    check('cluster k=1 reduces to train_stats normalization',
          np.abs(single - plain).max() < 1e-4,
          f'max diff {np.abs(single - plain).max():.2e}')

    check('normalize dispatches cluster scheme',
          R.normalize(ptr, ntr, pte, nte, 'cluster', True, cluster_k=2)[0].shape == ptr.shape)


def test_kmeans():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-8, 0.3, (60, 4)), rng.normal(8, 0.3, (60, 4))]).astype('f4')
    C, a = R._kmeans(X, 2, seed=0)
    check('kmeans separates two clean clusters',
          len(set(a[:60])) == 1 and len(set(a[60:])) == 1 and a[0] != a[60])
    check('kmeans deterministic from seed', np.array_equal(R._kmeans(X, 2, seed=0)[1], a))
    check('kmeans k>n is clamped', len(R._kmeans(X[:3], 10, seed=0)[0]) == 3)


def test_baselines():
    """A planted signal every baseline must find, and a null it must not."""
    rng = np.random.default_rng(2)
    n, d = 400, 30
    w = rng.normal(size=d); w /= np.linalg.norm(w)

    def synth(n):
        y = rng.integers(0, 2, n)
        noise_p = rng.normal(0, 1, (n, d))
        noise_n = rng.normal(0, 1, (n, d))
        delta = np.outer(2.0 * (y - 0.5), w) * 3.0
        return ((noise_p + delta / 2).astype('f4'),
                (noise_n - delta / 2).astype('f4'), y)

    ptr, ntr, ytr = synth(n)
    pte, nte, yte = synth(200)

    for name, fn in (('crc_tpc', R.train_pca_tpc), ('kmeans_diff', R.train_kmeans_diff)):
        s, _ = fn(ptr, ntr, pte, nte, seed=0)
        acc = R.score_report(s, yte)['flipped_acc']
        check(f'{name} recovers a planted direction', acc > 0.9, f'{acc:.1%}')

    s, _ = R.train_mean_diff(ptr, ntr, pte, nte, ytr, seed=0)
    check('mean_diff (supervised) recovers it too',
          R.score_report(s, yte)['flipped_acc'] > 0.9)

    # Null: no planted signal, so everything must sit near chance.
    q = [rng.normal(size=(n, d)).astype('f4') for _ in range(2)]
    r = [rng.normal(size=(200, d)).astype('f4') for _ in range(2)]
    ynull = rng.integers(0, 2, 200)
    for name, fn in (('crc_tpc', R.train_pca_tpc), ('kmeans_diff', R.train_kmeans_diff),
                     ('random_dir', R.train_random_dir)):
        s, _ = fn(q[0], q[1], r[0], r[1], seed=0)
        acc = R.score_report(s, ynull)['flipped_acc']
        check(f'{name} stays near chance on noise', acc < 0.62, f'{acc:.1%}')

    s, _ = R.train_random_dir(ptr, ntr, pte, nte, seed=0)
    check('random_dir returns scores in [0,1]', s.min() >= 0.0 and s.max() <= 1.0)

    s, m = R.train_mean_diff(ptr, ntr, pte, nte, np.ones(n, dtype=int), seed=0)
    check('mean_diff flags a single-class train set instead of dividing by zero',
          m.get('degenerate') is True)


def test_derange():
    from extract import _derange
    rng = np.random.default_rng(3)

    items = [int(v) for v in rng.integers(0, 50, 300)]
    perm = _derange(items, 7)
    check('derange is a permutation',
          sorted(perm.tolist()) == list(range(len(items))))
    fixed = sum(1 for i, j in enumerate(perm) if items[j] == items[i])
    check('derange leaves no item paired with its own image', fixed == 0, f'{fixed} fixed')
    check('derange deterministic from seed',
          np.array_equal(_derange(items, 7), perm))

    # Duplicated values must still yield a valid permutation.
    dupes = [0, 0, 0, 1, 1, 2, 3, 4, 5, 6]
    p2 = _derange(dupes, 1)
    check('derange handles duplicate values without collapsing',
          sorted(p2.tolist()) == list(range(len(dupes))))
    check('derange breaks all pairings when values allow',
          all(dupes[p2[i]] != dupes[i] for i in range(len(dupes))))

    # A majority value cannot be fully deranged; must not loop forever.
    heavy = [9] * 8 + [1, 2]
    p3 = _derange(heavy, 2)
    check('derange survives an undernageable majority value',
          sorted(p3.tolist()) == list(range(len(heavy))))


def test_banner_distractor():
    from PIL import Image
    from extract import add_banner, DISTRACTOR_WORDS
    from layer_sweep import find_cache_v3

    src = Image.new('RGB', (600, 400), (90, 120, 160))
    outs = [add_banner(src, w) for w in DISTRACTOR_WORDS]
    check('banner preserves image size', all(o.size == src.size for o in outs))
    check('banner actually modifies the image',
          all(o.tobytes() != src.tobytes() for o in outs))
    check('TRUE and FALSE banners are visually distinct',
          outs[0].tobytes() != outs[1].tobytes())
    check('banner survives a tiny image',
          add_banner(Image.new('RGB', (48, 32)), 'FALSE').size == (48, 32))

    # Writer and reader must agree on the suffix.
    with tempfile.TemporaryDirectory() as td:
        t = pathlib.Path(td)
        (t / 'hs_qwen2_object_detection_distract-banner.npz').touch()
        (t / 'hs_qwen2_object_detection_shuffled_distract-banner_tplain-qa.npz').touch()
        check('find_cache_v3 resolves the distractor suffix',
              find_cache_v3(t, 'qwen2', 'object_detection',
                            distractor='banner') is not None)
        check('find_cache_v3 composes shuffled + distractor + templates',
              find_cache_v3(t, 'qwen2', 'object_detection', shuffled=True,
                            templates=['plain', 'qa'], distractor='banner')
              is not None)
        check('find_cache_v3 does not match a plain cache when banner is asked',
              find_cache_v3(t, 'qwen2', 'object_detection') is None)


def test_prompts():
    import prompts as PR
    for name in PR.ALL_TEMPLATES:
        pos = PR.render(name, 'Is there a dog?', True)
        neg = PR.render(name, 'Is there a dog?', False)
        pw, nw = PR.answer_words(name)
        check(f'template {name}: pos/neg differ only in the answer word',
              pos[:-len(pw)] == neg[:-len(nw)] and pos.endswith(pw) and neg.endswith(nw),
              repr(pos))
        prefix = PR.zeroshot_prefix(name, 'Is there a dog?')
        check(f'template {name}: zero-shot prefix is the CCS prompt truncated',
              pos.startswith(prefix) and pw not in prefix, repr(prefix))
        check(f'template {name}: question mark not doubled', '??' not in pos, repr(pos))

    check('template_names default', PR.template_names(None) == ['plain'])
    check('template_names all', PR.template_names(['all']) == PR.ALL_TEMPLATES)
    try:
        PR.template_names(['nope'])
        check('template_names rejects unknown names', False)
    except ValueError:
        check('template_names rejects unknown names', True)


def test_zeroshot_join():
    from compare_zeroshot import join_on_question_id
    zs = {'question_ids': np.array([10, 11, 12, 13]),
          'labels': np.array([1, 0, 1, 0]),
          'margin': np.array([0.5, -0.5, 0.2, -0.2])}

    keep, rows, st = join_on_question_id(zs, [12, 10], [1, 1])
    check('join maps question_ids to rows regardless of order',
          rows is not None and list(rows) == [2, 0] and list(keep) == [0, 1], st)

    keep, rows, st = join_on_question_id(zs, [12, 99, 10], [1, 1, 1],
                                         min_coverage=0.5)
    check('join drops unmatched items and reports coverage',
          list(keep) == [0, 2] and list(rows) == [2, 0] and 'dropped' in st, st)

    keep, rows, st = join_on_question_id(zs, [12, 99], [1, 1], min_coverage=0.9)
    check('join refuses when coverage falls below the threshold',
          keep is None and 'refusing' in st, st)

    # Same length, same range, different items -- what a positional join misses.
    keep, rows, st = join_on_question_id(zs, [10, 11], [1, 1])
    check('join refuses on a label mismatch instead of reporting a number',
          keep is None and 'label mismatch' in st, st)

    # Labels indexed by `keep`: a dropped item must not cause a false alarm.
    keep, rows, st = join_on_question_id(zs, [10, 99, 11], [1, 0, 0],
                                         min_coverage=0.5)
    check('label cross-check aligns with the kept subset',
          keep is not None and list(keep) == [0, 2], st)

    keep, rows, st = join_on_question_id({**zs, 'question_ids': None}, [10], [1])
    check('join refuses on a pre-question_id zero-shot file',
          keep is None and 'predates' in st, st)

    keep, rows, st = join_on_question_id(zs, None, None)
    check('join refuses when results JSON has no test_question_ids',
          keep is None and 'test_question_ids' in st, st)


def test_pope_converter():
    from pope_to_vqa import convert, read_records
    recs = [{'question_id': 1, 'image': 'COCO_val2014_000000310196.jpg',
             'text': 'Is there a snowboard in the image?', 'label': 'yes'},
            {'question_id': 2, 'image': 'COCO_val2014_000000310196.jpg',
             'text': 'Is there a fork in the image?', 'label': 'no'},
            {'question_id': 3, 'image': 'x.jpg', 'text': '', 'label': 'yes'}]
    items, dropped = convert(recs, 'pope_random')
    check('pope converter keeps well-formed rows', len(items) == 2 and dropped == 1)
    check('pope converter emits the build_pairs schema',
          set(items[0]) >= {'question_id', 'image_id', 'question', 'answer'})
    check('pope image_id stays a filename string (find_image needs it literal)',
          items[0]['image_id'] == 'COCO_val2014_000000310196.jpg')

    pop, _ = convert(recs[:2], 'pope_popular')
    check('pope question_ids are namespaced per split',
          set(i['question_id'] for i in items).isdisjoint(
              i['question_id'] for i in pop))

    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / 'a.json'
        p.write_text('\n'.join(json.dumps(r) for r in recs[:2]))
        check('pope reader accepts JSON Lines', len(read_records(p)) == 2)
        p.write_text(json.dumps(recs[:2]))
        check('pope reader accepts a JSON list', len(read_records(p)) == 2)


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
    # principal angles: singular values of W^T V close to 1 <=> same subspace
    sv = np.linalg.svd(W.T @ Vt[:20].T, compute_uv=False)
    check('randomized PCA recovers subspace', sv.min() > 0.999, f'min sv={sv.min():.6f}')

    ptr, ntr, pte, nte = (rng.normal(size=(120, 300)).astype('f4') for _ in range(4))
    a, b, c, d = R.pca_reduce(ptr, ntr, pte, nte, 50)
    check('pca_reduce shapes', all(x.shape == (120, 50) for x in (a, b, c, d)))
    check('pca_reduce dtype float32', all(x.dtype == np.float32 for x in (a, b, c, d)))
    # shifting TEST must not change the TRAIN projection if PCA was fit on train only
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


def test_locate_positions():
    from extract import locate_positions
    # Qwen2 case: [... text, Yes, <|im_end|>, \n]
    # 151645 = <|im_end|> id in the Qwen2 tokenizer, 198 = '\n'; 9999 stands for 'Yes'
    ids_qwen = np.array([10, 20, 30, 9999, 151645, 198])
    m_qwen = locate_positions(ids_qwen, 151645)
    check('locate_positions Qwen: answer < final',
          m_qwen == {'answer': 3, 'eot': 4, 'final': 5} and m_qwen['answer'] < m_qwen['final'])

    # LLaVA case: [... text, Yes, </s>]
    ids_llava = np.array([10, 20, 30, 9999, 2])
    m_llava = locate_positions(ids_llava, 2)
    check('locate_positions LLaVA: eot == final & answer < final',
          m_llava == {'answer': 3, 'eot': 4, 'final': 4} and m_llava['answer'] < m_llava['final'])

    # Missing eot token fallback
    ids_missing = np.array([10, 20, 30, 9999])
    m_missing = locate_positions(ids_missing, 999)
    check('locate_positions missing eot fallback',
          m_missing == {'answer': 2, 'eot': 3, 'final': 3})


if __name__ == '__main__':
    for fn in (test_pair_reconstruction, test_alignment, test_auroc, test_splits,
               test_normalize, test_cluster_norm, test_kmeans, test_baselines,
               test_find_cache, test_score_report,
               test_pca_control, test_gaussian_control, test_diagnostics,
               test_locate_positions, test_derange, test_banner_distractor, test_prompts,
               test_zeroshot_join, test_pope_converter):
        print(f'\n-- {fn.__name__} --')
        fn()
    print('\n' + ('ALL PASS' if _ok else 'FAILURES PRESENT'))
    sys.exit(0 if _ok else 1)

