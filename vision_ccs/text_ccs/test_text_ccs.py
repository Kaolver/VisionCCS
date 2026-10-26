"""CPU unit tests: python text_ccs/test_text_ccs.py"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import burns_ccs as B  # noqa: E402
import prompts_text as PT  # noqa: E402
import prepare_data as PD  # noqa: E402

FAKE_ROWS = {
    'imdb': {'text': 'A fine film.<br />Loved it.', 'label': 1},
    'amazon_polarity': {'title': 'Broke', 'content': 'Stopped working.', 'label': 0},
    'ag_news': {'text': 'Stocks rose today.', 'label': 2},
    'dbpedia_14': {'title': 'Oak', 'content': 'The oak is a tree.', 'label': 10},
    'rte': {'sentence1': 'A man sleeps.', 'sentence2': 'A man rests.', 'label': 0},
    'qnli': {'question': 'Who won?', 'sentence': 'Bob won.', 'label': 1},
    'copa': {'premise': 'The man fell.', 'choice1': 'He slipped.', 'choice2': 'He sang.',
             'question': 'cause', 'label': 0},
    'boolq': {'question': 'is water wet', 'passage': 'Water is wet.', 'answer': True},
    'piqa': {'goal': 'Open a jar', 'sol1': 'Twist the lid.', 'sol2': 'Eat the lid.', 'label': 0},
}


class ToyTok:
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self):
        self.vocab = {}

    def _ids(self, s):
        import re
        return [self.vocab.setdefault(w, len(self.vocab) + 10) for w in re.findall(r'\S+|\n', s)]

    def __call__(self, s, add_special_tokens=True):
        class R:
            pass
        r = R()
        r.input_ids = ([2] if add_special_tokens else []) + self._ids(s)
        return r


class TestPrompts(unittest.TestCase):
    def test_every_template_renders_two_distinct_answers(self):
        rng = np.random.default_rng(0)
        for ds in PT.DATASETS:
            it = PD.to_items(ds, [FAKE_ROWS[ds]], rng)[0]
            self.assertGreaterEqual(PT.n_templates(ds), 5, ds)
            for t in range(PT.n_templates(ds)):
                q, (a0, a1) = PT.render(ds, t, it)
                self.assertTrue(q.strip(), (ds, t))
                self.assertNotEqual(a0, a1, (ds, t))
                self.assertNotIn('{', q, (ds, t))

    def test_label_semantics(self):
        rng = np.random.default_rng(0)
        it = lambda ds: PD.to_items(ds, [FAKE_ROWS[ds]], rng)[0]
        self.assertEqual(it('imdb')['label'], 1)
        self.assertNotIn('<br />', it('imdb')['text'])
        self.assertEqual(it('rte')['label'], 1)
        self.assertEqual(it('qnli')['label'], 0)
        self.assertEqual(it('boolq')['label'], 1)
        self.assertEqual(it('copa')['label'], 0)
        for ds in ('ag_news', 'dbpedia_14'):
            x = it(ds)
            true_name = x['label0'] if x['label'] == 0 else x['label1']
            self.assertEqual(true_name, x['true_topic'])
            self.assertNotEqual(x['label0'], x['label1'])

    def test_topic_answer_balance(self):
        rng = np.random.default_rng(0)
        rows = [FAKE_ROWS['ag_news']] * 2000
        labels = [x['label'] for x in PD.to_items('ag_news', rows, rng)]
        self.assertAlmostEqual(np.mean(labels), 0.5, delta=0.05)

    def test_join_raw(self):
        self.assertEqual(PT.join_raw('Q?', 'yes'), ('Q?\n', 'yes'))
        self.assertEqual(PT.join_raw('Q: ', 'yes'), ('Q: ', 'yes'))


class TestEncode(unittest.TestCase):
    def test_answer_span_and_suffix(self):
        sys.modules.setdefault('torch', __import__('torch'))
        import extract_text as E
        tok = ToyTok()
        ids, s, e = E.encode(tok, 'Is it good?', 'very good', 'raw', None, eos=True)
        self.assertEqual(ids[-1], tok.eos_token_id)
        self.assertEqual(e - s, 2)
        self.assertEqual(ids[0], 2)
        ids2, s2, e2 = E.encode(tok, 'Is it good?', 'no', 'chat', ('USER: ', [7, 8]), eos=True)
        self.assertEqual(ids2[-2:], [7, 8])
        self.assertEqual(e2 - s2, 1)
        self.assertEqual(ids2[e2 - 1], tok.vocab['no'])


def planted(n=400, d=64, seed=0, n_t=3, signal=3.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = np.repeat(np.arange(n_t), n // n_t + 1)[:n]
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    base = rng.normal(size=(n, d)) + 5 * t[:, None]
    s = np.where(y == 1, 1.0, -1.0)[:, None] * signal / 2
    h1 = base + s * u + 0.5 * rng.normal(size=(n, d))
    h0 = base - s * u + 0.5 * rng.normal(size=(n, d))
    return h0.astype(np.float32), h1.astype(np.float32), y, t


class TestCCS(unittest.TestCase):
    def test_normalize_paper(self):
        x = np.random.default_rng(0).normal(3, 2, size=(200, 50))
        z = B.normalize(x, 'paper')
        self.assertTrue(np.allclose(z.mean(0), 0, atol=1e-6))
        self.assertAlmostEqual(np.linalg.norm(z, axis=1).mean(), np.sqrt(50), places=4)

    def test_ccs_finds_planted_direction(self):
        h0, h1, y, t = planted()
        groups = np.arange(len(y))
        res, _ = B.evaluate_cell(h0, h1, y, groups, t, seed=0,
                                 cfg={'epochs': 300, 'ntries': 4, 'skip_lr': True})
        self.assertGreater(res['ccs']['acc'], 0.9)

    def test_ccs_chance_without_signal(self):
        h0, h1, y, t = planted(signal=0.0)
        res, _ = B.evaluate_cell(h0, h1, y, np.arange(len(y)), t, seed=0,
                                 cfg={'epochs': 200, 'ntries': 2, 'skip_lr': True})
        self.assertLess(res['ccs']['acc'], 0.65)

    def test_batched_tries_equal_single_runs(self):
        h0, h1, _, _ = planted(n=100, d=16)
        W, b, losses, _ = B.train_ccs(h0, h1, ntries=3, epochs=50, seed=5)
        import torch
        g = torch.Generator().manual_seed(5)
        bound = 1 / np.sqrt(16)
        W0 = (torch.rand(3, 16, generator=g) * 2 - 1) * bound
        b0 = (torch.rand(3, generator=g) * 2 - 1) * bound
        w = W0[1:2].clone().requires_grad_(); bb = b0[1:2].clone().requires_grad_()
        opt = torch.optim.AdamW([w, bb], lr=1e-2, weight_decay=0.01)
        X0, X1 = torch.tensor(h0), torch.tensor(h1)
        for _ in range(50):
            p0 = torch.sigmoid(X0 @ w.T + bb); p1 = torch.sigmoid(X1 @ w.T + bb)
            loss = (torch.min(p0, p1) ** 2).mean(0) + ((p0 - (1 - p1)) ** 2).mean(0)
            opt.zero_grad(); loss.sum().backward(); opt.step()
        self.assertTrue(np.allclose(w.detach().numpy()[0], W[1], atol=1e-5))
        self.assertAlmostEqual(float(loss), float(losses[1]), places=5)

    def test_logreg_matches_sklearn(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 20)); y = (X[:, 0] + 0.5 * rng.normal(size=300) > 0.4).astype(int)
        for C in (0.01, 1.0):
            sk = LogisticRegression(class_weight='balanced', C=C, max_iter=5000, tol=1e-10).fit(X, y)
            w, b = B.fit_logreg(X, y, C)
            self.assertTrue(np.allclose(w, sk.coef_[0], atol=2e-3), (C, w[:3], sk.coef_[0][:3]))
            self.assertAlmostEqual(b, sk.intercept_[0], delta=2e-3)

    def test_zero_shot_calibration_balances(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        lp0 = rng.normal(size=400); lp1 = lp0 + 2 + 0.5 * (y - 0.5)
        raw, cal = B.zero_shot(lp0, lp1, y, np.zeros(400, int))
        self.assertAlmostEqual(raw, np.mean(y == 1), places=6)
        self.assertGreater(cal, 0.6)

    def test_split_keeps_groups_together(self):
        g = np.repeat(np.arange(50), 4)
        tr, te = B.split_items(g, 0)
        self.assertFalse(set(g[tr]) & set(g[te]))
        self.assertEqual(len(set(g[tr])), 30)


class TestEndToEnd(unittest.TestCase):
    def test_evaluate_and_summarize_on_synthetic_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'c').mkdir(); (tmp / 'v').mkdir()
            n_items, n_t, L, d = 120, 3, 2, 32
            y = np.repeat(np.random.default_rng(0).integers(0, 2, n_items), n_t)
            h0, h1, _, _ = planted(n=n_items * n_t, d=d, n_t=n_t, seed=1)
            rng = np.random.default_rng(2)
            u = rng.normal(size=d); u /= np.linalg.norm(u)
            s = np.where(y == 1, 1.5, -1.5)[:, None]
            h1 = (h0 + 2 * s * u).astype(np.float32)
            stack = lambda h: np.stack([np.stack([h, h], 1)] * L, 1).astype(np.float16)
            items = np.repeat([f'x-{i}' for i in range(n_items)], n_t)
            tids = np.tile(np.arange(n_t), n_items)
            for ds in ('imdb', 'rte'):
                np.savez(tmp / 'c' / f'hs_toy_raw_{ds}.npz', h0=stack(h0), h1=stack(h1),
                         lp0=np.zeros(len(y), np.float32), lp1=(y - 0.5).astype(np.float32),
                         labels=y, item_ids=items, template_ids=tids,
                         layers=np.array([4, 8]), positions=np.array(['answer', 'final']))
            np.savez(tmp / 'v' / 'hs_toy_object_detection.npz', pos_hiddens=stack(h1),
                     neg_hiddens=stack(h0), labels=y, image_ids=np.repeat(np.arange(n_items), n_t),
                     template_ids=np.array(['plain'] * len(y)), layers=np.array([4, 8]),
                     positions=np.array(['answer', 'final']))
            py = sys.executable
            common = ['--model', 'toy', '--seeds', '0', '--epochs', '100', '--ntries', '2']
            subprocess.run([py, str(HERE / 'evaluate_text.py'), *common, '--cache-dir', str(tmp / 'c'),
                            '--datasets', 'imdb', 'rte', '--out', str(tmp / 'r' / 'toy_text-raw.json')],
                           check=True, capture_output=True)
            subprocess.run([py, str(HERE / 'evaluate_text.py'), *common, '--vision-cache-dir', str(tmp / 'v'),
                            '--categories', 'object_detection', '--skip-lr',
                            '--out', str(tmp / 'r' / 'toy_vision-vqa.json')],
                           check=True, capture_output=True)
            r = json.loads((tmp / 'r' / 'toy_text-raw.json').read_text())
            c = r['datasets']['imdb']['cells']['8|final']
            self.assertTrue(c['is_last_layer'])
            self.assertGreater(c['ccs'], 0.9)
            self.assertGreater(c['lr'], 0.9)
            self.assertAlmostEqual(r['datasets']['imdb']['zero_shot']['raw'], 1.0)
            v = json.loads((tmp / 'r' / 'toy_vision-vqa.json').read_text())
            self.assertGreater(v['datasets']['object_detection']['cells']['8|final']['ccs'], 0.9)
            p = subprocess.run([py, str(HERE / 'summarize.py'), '--results-dir', str(tmp / 'r'),
                                '--out-dir', str(tmp / 'rep')], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            md = (tmp / 'rep' / 'text_ccs_tables.md').read_text(encoding='utf-8')
            self.assertIn('Paper protocol', md)
            self.assertIn('Layer curves', md)


if __name__ == '__main__':
    unittest.main(verbosity=2)
