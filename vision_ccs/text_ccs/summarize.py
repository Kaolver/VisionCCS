"""Writes <out-dir>/text_ccs_tables.md from the evaluate_text.py results."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Burns et al. 2022, Table 1. The mean excludes T0, as in the paper.
PAPER = {
    'GPT-J': {'0-shot': 53.2, 'cal 0-shot': 56.0, 'CCS': 61.7, 'LR': 78.1},
    'paper mean (5 models)': {'0-shot': 62.8, 'cal 0-shot': 67.2, 'CCS': 71.2, 'LR': 83.7},
}

NAMES = {'gptj': 'GPT-J-6B', 'llava': 'LLaVA-1.5-7B', 'qwen2': 'Qwen2-VL-7B',
         'qwen2_5': 'Qwen2.5-VL-7B', 'vicuna': 'Vicuna-7B-v1.5',
         'qwen2_llm': 'Qwen2-7B-Instruct', 'qwen2_5_llm': 'Qwen2.5-7B-Instruct'}
ORDER = ['gptj', 'llava', 'qwen2', 'qwen2_5', 'vicuna', 'qwen2_llm', 'qwen2_5_llm']
VLM_OF = {'vicuna': 'llava', 'qwen2_llm': 'qwen2', 'qwen2_5_llm': 'qwen2_5'}


def pct(x):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{100 * x:.1f}'


def load(results_dir):
    R = {}
    for f in sorted(Path(results_dir).glob('*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        if 'datasets' in d and d['datasets']:
            R[(d['model'], d['source'])] = d
    return R


def cell(ds, position, layer='last'):
    for c in ds['cells'].values():
        if c['position'] != position:
            continue
        if (layer == 'last' and c['is_last_layer']) or c['layer'] == layer:
            return c
    return None


def averaged(d, position, layer='last', datasets=None):
    acc = {'ccs': [], 'lr': [], 'lr_tuned': [], 'ccs_sd_prompts': [], 'zs': [], 'zs_cal': []}
    names = []
    for n, ds in d['datasets'].items():
        if datasets and n not in datasets:
            continue
        c = cell(ds, position, layer)
        if c is None:
            continue
        names.append(n)
        for k in ('ccs', 'lr', 'lr_tuned', 'ccs_sd_prompts'):
            if k in c:
                acc[k].append(c[k])
        if 'zero_shot' in ds:
            acc['zs'].append(ds['zero_shot']['raw'])
            acc['zs_cal'].append(ds['zero_shot']['calibrated'])
    out = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}
    out['datasets'] = names
    return out


def ordered(R, source):
    have = {m for m, s in R if s == source}
    return [m for m in ORDER if m in have] + sorted(have - set(ORDER))


def table1(R, source, position):
    models = ordered(R, source)
    if not models:
        return None, []
    avs = {m: averaged(R[(m, source)], position) for m in models}
    hdr = ['Method'] + [NAMES.get(m, m) for m in models] + list(PAPER)
    rows = [('0-shot', 'zs', '0-shot'), ('Calibrated 0-shot', 'zs_cal', 'cal 0-shot'),
            ('CCS', 'ccs', 'CCS'), ('LR (ceiling, C=1)', 'lr', 'LR'),
            ('LR (C tuned on train)', 'lr_tuned', None)]
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for label, k, pk in rows:
        vals = []
        for m in models:
            v = pct(avs[m][k])
            if k == 'ccs' and avs[m]['ccs_sd_prompts'] is not None:
                v += f' ({100 * avs[m]["ccs_sd_prompts"]:.1f})'
            vals.append(v)
        vals += [f'{PAPER[p][pk]:.1f}' if pk else '-' for p in PAPER]
        lines.append(f'| {label} | ' + ' | '.join(vals) + ' |')
    nds = {m: len(avs[m]['datasets']) for m in models}
    return '\n'.join(lines), [f'{NAMES.get(m, m)}: {n} datasets' for m, n in nds.items()]


def per_dataset(R, source, position, key='ccs'):
    models = ordered(R, source)
    dss = sorted({n for m in models for n in R[(m, source)]['datasets']})
    if not models:
        return None
    lines = ['| Model | ' + ' | '.join(dss) + ' | mean |', '|' + '---|' * (len(dss) + 2)]
    for m in models:
        vals = []
        for n in dss:
            ds = R[(m, source)]['datasets'].get(n)
            if key in ('zs_cal',):
                vals.append(ds['zero_shot']['calibrated'] if ds and 'zero_shot' in ds else None)
            else:
                c = cell(ds, position) if ds else None
                vals.append(c.get(key) if c else None)
        mean = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
        lines.append(f'| {NAMES.get(m, m)} | ' + ' | '.join(pct(v) for v in vals) + f' | {pct(mean)} |')
    return '\n'.join(lines)


def text_vs_vision(R):
    rows = []
    for m in ('llava', 'qwen2', 'qwen2_5'):
        r = [NAMES[m]]
        for src, pos in (('text-raw', 'final'), ('text-chat', 'final'),
                         ('vision-vqa', 'final'), ('vision-vg', 'final')):
            if (m, src) in R:
                a = averaged(R[(m, src)], pos)
                r += [pct(a['ccs']), pct(a['lr_tuned'])]
            else:
                r += ['-', '-']
        rows.append('| ' + ' | '.join(r) + ' |')
    hdr = ('| Model | text raw CCS | text raw LR | text chat CCS | text chat LR | '
           'VQAv2 CCS | VQAv2 LR | VG CCS | VG LR |')
    return '\n'.join([hdr, '|' + '---|' * 9] + rows)


def layer_curves(R, position='final'):
    out = []
    for (m, src), d in sorted(R.items(), key=lambda kv: (ORDER.index(kv[0][0]) if kv[0][0] in ORDER else 99, kv[0][1])):
        layers = sorted({c['layer'] for ds in d['datasets'].values()
                         for c in ds['cells'].values() if c['position'] == position})
        if len(layers) < 2:
            continue
        ccs, lr = [], []
        for l in layers:
            a = averaged(d, position, layer=l)
            ccs.append(pct(a['ccs'])); lr.append(pct(a['lr_tuned']))
        out.append(f'**{NAMES.get(m, m)}, {src}** (position {position})\n\n'
                   + '| layer | ' + ' | '.join(map(str, layers)) + ' |\n'
                   + '|' + '---|' * (len(layers) + 1) + '\n'
                   + '| CCS | ' + ' | '.join(ccs) + ' |\n'
                   + '| LR (tuned C) | ' + ' | '.join(lr) + ' |\n')
    return '\n'.join(out)


def verdict(R):
    lines = []
    g = R.get(('gptj', 'text-raw'))
    if g:
        a = averaged(g, 'final')
        gap = 100 * a['ccs'] - PAPER['GPT-J']['CCS']
        lines.append(f'- **Reproduction (GPT-J, raw, last layer):** CCS {pct(a["ccs"])} vs '
                     f'{PAPER["GPT-J"]["CCS"]} in the paper ({gap:+.1f}), LR {pct(a["lr"])} vs '
                     f'{PAPER["GPT-J"]["LR"]}, calibrated 0-shot {pct(a["zs_cal"])} vs '
                     f'{PAPER["GPT-J"]["cal 0-shot"]}. '
                     + ('Within 5 points of the paper.' if abs(gap) <= 5 else
                        'More than 5 points off the paper: check prompts and normalization first.'))
    else:
        lines.append('- **Reproduction:** no GPT-J result yet.')
    for m in ('llava', 'qwen2', 'qwen2_5'):
        t = R.get((m, 'text-raw'))
        if not t:
            continue
        a = averaged(t, 'final')
        s = (f'- **{NAMES[m]}, text only:** CCS {pct(a["ccs"])}, calibrated 0-shot '
             f'{pct(a["zs_cal"])}, LR {pct(a["lr"])} (tuned {pct(a["lr_tuned"])}). ')
        lrs = [x for x in (a['lr'], a['lr_tuned']) if x is not None]
        ok_order = a['zs_cal'] is not None and bool(lrs) and a['zs_cal'] <= a['ccs'] <= max(lrs)
        s += ('0-shot <= CCS <= LR holds. ' if ok_order else '0-shot <= CCS <= LR does not hold. ')
        base = next((k for k, v in VLM_OF.items() if v == m), None)
        if base and (base, 'text-raw') in R:
            b = averaged(R[(base, 'text-raw')], 'final')
            s += f'Its language model {NAMES[base]}: CCS {pct(b["ccs"])}. '
        for vs in ('vision-vqa', 'vision-vg'):
            if (m, vs) in R:
                v = averaged(R[(m, vs)], 'final')
                s += f'Same evaluator on {vs[7:].upper()} images: CCS {pct(v["ccs"])}, LR {pct(v["lr_tuned"])}. '
        lines.append(s)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', default='./text_ccs_results')
    ap.add_argument('--out-dir', default='./text_ccs_report')
    args = ap.parse_args()
    R = load(args.results_dir)
    if not R:
        print(f'no results in {args.results_dir}'); return 1
    md = ['# Original CCS on the vision-CCS models, text only\n',
          'Accuracy %, mean over datasets, prompts and seeds. CCS s.d. across prompts in '
          'parentheses. Paper columns: Burns et al. 2022, Table 1. Results: '
          +', '.join(f'{m}/{s}' for m, s in sorted(R)) + '.\n',
          '## Reading\n', verdict(R), '']
    for src, pos, title in (
            ('text-raw', 'final', 'Paper protocol: raw text, last layer, last token (EOS, as in the released code)'),
            ('text-raw', 'answer', 'Raw text, last layer, last answer token (Algorithm 1, no EOS)'),
            ('text-chat', 'final', 'Chat-template format (as the vision pipeline), last layer, last token')):
        t, notes = table1(R, src, pos)
        if t:
            md += [f'## {title}\n', t, '', 'Datasets per model: ' + '; '.join(notes) + '.', '']
    for src in ('text-raw', 'text-chat'):
        for key, label in (('ccs', 'CCS'), ('lr_tuned', 'LR (tuned C)'), ('zs_cal', 'Calibrated 0-shot')):
            t = per_dataset(R, src, 'final', key)
            if t:
                md += [f'## Per dataset: {label}, {src}, last layer\n', t, '']
    md += ['## Same evaluator, text vs images (last layer, last token)\n', text_vs_vision(R), '']
    md += ['## Layer curves (mean over datasets)\n', layer_curves(R)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'text_ccs_tables.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'wrote {out / "text_ccs_tables.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
