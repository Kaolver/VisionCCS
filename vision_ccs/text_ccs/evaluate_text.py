"""Score cached hidden states with burns_ccs.py.

Input is either text caches (extract_text.py) or vision caches (extract.py,
via --vision-cache-dir). Vision rows use 'Yes' as answer 1 and are split by image.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import burns_ccs as B  # noqa: E402
import prompts_text as PT  # noqa: E402

VISION_CATEGORIES = {
    'vqa': ['object_detection', 'attribute_recognition', 'spatial_recognition'],
    'vg': ['object', 'attribute', 'spatial'],
}


def load_text(cache_dir, model, fmt, ds):
    f = Path(cache_dir) / f'hs_{model}_{fmt}_{ds}.npz'
    if not f.exists():
        return None
    d = np.load(f)
    return {'h0': d['h0'], 'h1': d['h1'], 'labels': d['labels'].astype(int),
            'groups': d['item_ids'], 'templates': d['template_ids'].astype(int),
            'lp0': d['lp0'], 'lp1': d['lp1'],
            'layers': [int(x) for x in d['layers']],
            'positions': [str(x) for x in d['positions']], 'file': str(f)}


def load_vision(cache_dir, model, cat):
    f = Path(cache_dir) / f'hs_{model}_{cat}.npz'
    if not f.exists():
        return None
    d = np.load(f)
    tids = d['template_ids'] if 'template_ids' in d else np.zeros(len(d['labels']))
    return {'h0': d['neg_hiddens'], 'h1': d['pos_hiddens'],
            'labels': d['labels'].astype(int), 'groups': d['image_ids'],
            'templates': np.unique(tids, return_inverse=True)[1],
            'lp0': None, 'lp1': None,
            'layers': [int(x) for x in d['layers']],
            'positions': [str(x) for x in d['positions']], 'file': str(f)}


def run_dataset(c, cfg, seeds, layers, positions, device, last_layer_positions=()):
    out = {'file': c['file'], 'n_rows': int(len(c['labels'])),
           'n_items': int(len(np.unique(c['groups']))),
           'n_prompts': int(len(np.unique(c['templates']))),
           'true_is_1': float(np.mean(c['labels'])), 'cells': {}}
    if c['lp0'] is not None:
        zs = []
        for s in seeds:
            _, te = B.split_items(c['groups'], s, cfg['train_frac'])
            zs.append(B.zero_shot(c['lp0'][te], c['lp1'][te], c['labels'][te], c['templates'][te]))
        out['zero_shot'] = {'raw': float(np.mean([z[0] for z in zs])),
                            'calibrated': float(np.mean([z[1] for z in zs]))}
    for li, layer in enumerate(c['layers']):
        if layers and layer not in layers and not ('last' in layers and li == len(c['layers']) - 1):
            continue
        for pi, pos in enumerate(c['positions']):
            is_last = li == len(c['layers']) - 1
            if positions and pos not in positions and not (is_last and pos in last_layer_positions):
                continue
            h0 = c['h0'][:, li, pi, :].astype(np.float32)
            h1 = c['h1'][:, li, pi, :].astype(np.float32)
            runs = [B.evaluate_cell(h0, h1, c['labels'], c['groups'], c['templates'],
                                    s, cfg, device)[0] for s in seeds]
            cell = {'layer': layer, 'position': pos, 'is_last_layer': li == len(c['layers']) - 1}
            for k in ('ccs', 'lr', 'lr_tuned'):
                if k in runs[0]:
                    a = [r[k]['acc'] for r in runs]
                    cell[k] = float(np.mean(a))
                    cell[k + '_sd_seeds'] = float(np.std(a))
                    cell[k + '_sd_prompts'] = float(np.mean([r[k]['sd_prompts'] for r in runs]))
            cell['ccs_loss'] = float(np.mean([r['ccs']['loss'] for r in runs]))
            if 'lr_tuned' in runs[0]:
                cell['lr_tuned_C'] = [r['lr_tuned']['C'] for r in runs]
            out['cells'][f'{layer}|{pos}'] = cell
            print(f'    layer {layer:2d} {pos:6s}  CCS {cell["ccs"]:.3f}  '
                  + (f'LR {cell["lr"]:.3f}  LR-tuned {cell["lr_tuned"]:.3f}' if 'lr' in cell else ''),
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--cache-dir', default='./caches_text')
    ap.add_argument('--format', default='raw', choices=['raw', 'chat'])
    ap.add_argument('--datasets', nargs='+', default=PT.DATASETS)
    ap.add_argument('--vision-cache-dir', default=None,
                    help='score extract.py caches instead of text caches')
    ap.add_argument('--vision-dataset', default='vqa', choices=list(VISION_CATEGORIES))
    ap.add_argument('--categories', nargs='+', default=None)
    ap.add_argument('--layers', nargs='+', default=None,
                    help='layer numbers to score, or "last"; default every cached layer')
    ap.add_argument('--positions', nargs='+', default=None,
                    help='positions to score at every selected layer; default all cached')
    ap.add_argument('--last-layer-positions', nargs='+', default=[],
                    help='extra positions scored at the last layer only')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--train-frac', type=float, default=0.6)
    ap.add_argument('--norm', default='paper', choices=['paper', 'mean', 'var'])
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--ntries', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--skip-lr', action='store_true')
    ap.add_argument('--out', required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = {'train_frac': args.train_frac, 'norm': args.norm, 'epochs': args.epochs,
           'ntries': args.ntries, 'lr': args.lr, 'weight_decay': args.weight_decay,
           'skip_lr': args.skip_lr}
    layers = None
    if args.layers:
        layers = [l if l == 'last' else int(l) for l in args.layers]

    if args.vision_cache_dir:
        source = f'vision-{args.vision_dataset}'
        names = args.categories or VISION_CATEGORIES[args.vision_dataset]
        loader = lambda n: load_vision(args.vision_cache_dir, args.model, n)
    else:
        source = f'text-{args.format}'
        names = args.datasets
        loader = lambda n: load_text(args.cache_dir, args.model, args.format, n)

    out_f = Path(args.out)
    out_f.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out_f.read_text(encoding='utf-8')) if (args.resume and out_f.exists()) else {}
    res.update({'model': args.model, 'source': source, 'cfg': cfg, 'seeds': args.seeds})
    res.setdefault('datasets', {})
    print(f'device={device} model={args.model} source={source} cfg={cfg}')
    missing = []
    for n in names:
        if n in res['datasets']:
            print(f'{n}: already in {out_f}'); continue
        c = loader(n)
        if c is None:
            print(f'{n}: no cache, skipped'); missing.append(n); continue
        t0 = time.time()
        print(f'{n}: {c["file"]}  rows={len(c["labels"])}  layers={c["layers"]}')
        res['datasets'][n] = run_dataset(c, cfg, args.seeds, layers, args.positions, device,
                                         args.last_layer_positions)
        if 'zero_shot' in res['datasets'][n]:
            z = res['datasets'][n]['zero_shot']
            print(f'    zero-shot raw {z["raw"]:.3f}  calibrated {z["calibrated"]:.3f}')
        print(f'  {n} done in {time.time() - t0:.0f}s', flush=True)
        out_f.write_text(json.dumps(res, indent=1), encoding='utf-8')
    out_f.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'wrote {out_f}' + (f'  (no cache for {missing})' if missing else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
