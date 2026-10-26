"""Download the text datasets and write label-balanced pools to text_ccs/data/.

Needs internet and pyarrow==17.0.0; run once on the login node from vision_ccs/.
Story Cloze is left out: it is only distributed through a request form.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from prompts_text import AG_NEWS_LABELS, DBPEDIA_LABELS, DATASETS

# train is pooled in where test/validation is too small (RTE, COPA)
SOURCES = {
    'imdb': ('stanfordnlp/imdb', ['plain_text/test-00000-of-00001.parquet']),
    'amazon_polarity': ('fancyzhx/amazon_polarity', ['amazon_polarity/test-00000-of-00001.parquet']),
    'ag_news': ('fancyzhx/ag_news', ['data/test-00000-of-00001.parquet']),
    'dbpedia_14': ('fancyzhx/dbpedia_14', ['dbpedia_14/test-00000-of-00001.parquet']),
    'rte': ('nyu-mll/glue', ['rte/train-00000-of-00001.parquet',
                             'rte/validation-00000-of-00001.parquet']),
    'qnli': ('nyu-mll/glue', ['qnli/validation-00000-of-00001.parquet']),
    'copa': ('aps/super_glue', ['copa/train-00000-of-00001.parquet',
                                'copa/validation-00000-of-00001.parquet']),
    'boolq': ('google/boolq', ['data/validation-00000-of-00001.parquet']),
    # ybisk/piqa is a loading script; this is a parquet mirror. Test is unlabeled.
    'piqa': ('baber/piqa', ['piqa_validation.parquet']),
}


def read_rows(repo, files):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rows = []
    for f in files:
        path = hf_hub_download(repo, f, repo_type='dataset')
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def to_items(name, rows, rng):
    out = []
    for i, r in enumerate(rows):
        if name == 'imdb':
            it = {'text': r['text'].replace('<br />', ' '), 'label': int(r['label'])}
        elif name == 'amazon_polarity':
            it = {'title': r['title'], 'content': r['content'], 'label': int(r['label'])}
        elif name in ('ag_news', 'dbpedia_14'):
            names = AG_NEWS_LABELS if name == 'ag_news' else DBPEDIA_LABELS
            true = int(r['label'])
            # Burns I.2.1: a random wrong topic, the pair in random order
            wrong = int(rng.choice([k for k in range(len(names)) if k != true]))
            coin = int(rng.integers(2))
            pair = (true, wrong) if coin == 0 else (wrong, true)
            it = {'label0': names[pair[0]], 'label1': names[pair[1]], 'label': coin,
                  'true_topic': names[true]}
            it['text' if name == 'ag_news' else 'content'] = r['text' if name == 'ag_news' else 'content']
        elif name == 'rte':
            # GLUE: 0 = entailment, so the true answer is "yes"
            it = {'premise': r['sentence1'], 'hypothesis': r['sentence2'],
                  'label': 1 if int(r['label']) == 0 else 0}
        elif name == 'qnli':
            it = {'question': r['question'], 'sentence': r['sentence'],
                  'label': 1 if int(r['label']) == 0 else 0}
        elif name == 'copa':
            it = {'premise': r['premise'], 'choice1': r['choice1'],
                  'choice2': r['choice2'], 'question': r['question'],
                  'label': int(r['label'])}
        elif name == 'boolq':
            it = {'question': r['question'], 'passage': r['passage'],
                  'label': int(bool(r['answer']))}
        elif name == 'piqa':
            it = {'goal': r['goal'], 'sol1': r['sol1'], 'sol2': r['sol2'],
                  'label': int(r['label'])}
        else:
            raise ValueError(name)
        if it['label'] not in (0, 1):
            continue
        it['id'] = f'{name}-{i}'
        out.append(it)
    return out


def balanced_pool(items, size, rng, balance=True):
    if not balance:
        return [items[i] for i in rng.permutation(len(items))[:size]]
    by = {0: [], 1: []}
    for it in items:
        by[it['label']].append(it)
    k = min(len(by[0]), len(by[1]), size // 2)
    pool = [by[c][i] for c in (0, 1) for i in rng.permutation(len(by[c]))[:k]]
    return [pool[i] for i in rng.permutation(len(pool))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    ap.add_argument('--pool', type=int, default=2000,
                    help='items kept per dataset, more than are used')
    ap.add_argument('--out-dir', default=str(Path(__file__).parent / 'data'))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        rng = np.random.default_rng([args.seed, DATASETS.index(name)])
        repo, files = SOURCES[name]
        rows = read_rows(repo, files)
        items = to_items(name, rows, rng)
        pool = balanced_pool(items, args.pool, rng, balance=(name != 'copa'))
        f = out / f'{name}.jsonl'
        with open(f, 'w', encoding='utf-8') as fh:
            for it in pool:
                fh.write(json.dumps(it, ensure_ascii=False) + '\n')
        print(f'{name:16s} {len(rows):7d} rows -> {len(pool):5d} items '
              f'(true answer = 1 for {np.mean([it["label"] for it in pool]):.1%})  {f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
