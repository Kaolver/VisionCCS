"""Text-only contrast-pair hidden states and zero-shot log-probs (Burns et al. 2022).

Formats:
  raw   question + "\n" + answer + EOS, as in ccs/utils.py
  chat  the same text as the user turn of the model's chat template

Positions: 'answer' is the last answer token, 'final' the last token.
Writes <out-dir>/hs_<model>_<format>_<dataset>.npz, one row per (item, template).
Run from vision_ccs/.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompts_text as PT  # noqa: E402

MODELS = {
    'llava': ('llava-hf/llava-1.5-7b-hf', 'vlm'),
    'qwen2': ('Qwen/Qwen2-VL-7B-Instruct', 'vlm'),
    'qwen2_5': ('Qwen/Qwen2.5-VL-7B-Instruct', 'vlm'),
    'vicuna': ('lmsys/vicuna-7b-v1.5', 'llm'),
    'qwen2_llm': ('Qwen/Qwen2-7B-Instruct', 'llm'),
    'qwen2_5_llm': ('Qwen/Qwen2.5-7B-Instruct', 'llm'),
    'gptj': ('EleutherAI/gpt-j-6B', 'llm'),
}
# Only .bin on main, which transformers 4.57 will not load under torch < 2.6.
# These are the hub's safetensors conversions of the same weights.
REVISIONS = {'gptj': 'refs/pr/40', 'vicuna': 'refs/pr/13'}
USER_PREFIX_MODELS = ('llava', 'vicuna')


def load(model_tag, device):
    import torch
    path, kind = MODELS[model_tag]
    if kind == 'vlm':
        sys.path.insert(0, str(HERE.parent))
        from extract import load_model
        model, proc = load_model(model_tag, device)
        return model, proc.tokenizer, proc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    rev = {'revision': REVISIONS[model_tag]} if model_tag in REVISIONS else {}
    # Vicuna ships only a SentencePiece model; load it slow, as extract.py does for LLaVA
    tok = AutoTokenizer.from_pretrained(path, use_fast=model_tag != 'vicuna', **rev)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, device_map='auto' if device == 'cuda' else None,
        use_safetensors=True, **rev)
    if device == 'cpu':
        model = model.to(device)
    model.eval()
    return model, tok, None


SENTINEL = '⁣CONTENT⁣'


def chat_wrap(model_tag, tok, proc):
    """(prefix string, suffix token ids) around the user turn."""
    if model_tag == 'gptj':
        raise ValueError('gptj has no chat format; use --format raw')
    if model_tag in USER_PREFIX_MODELS:
        # extract.py uses 'USER: <image>\n{text}</s>'
        return 'USER: ', [tok.eos_token_id]
    if proc is not None:
        msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': SENTINEL}]}]
        s = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    else:
        msgs = [{'role': 'user', 'content': SENTINEL}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    pre, post = s.split(SENTINEL)
    return pre, tok(post, add_special_tokens=False).input_ids


def encode(tok, question, answer, fmt, wrap, eos):
    """(ids, start, end) with [start, end) the answer tokens."""
    q, a = PT.join_raw(question, answer)
    pre_s = q if fmt == 'raw' else wrap[0] + q
    pre = tok(pre_s).input_ids
    full = tok(pre_s + a).input_ids
    s = 0
    while s < min(len(pre), len(full)) and pre[s] == full[s]:
        s += 1
    if s >= len(full):
        raise ValueError(f'answer {answer!r} produced no tokens')
    if fmt == 'raw':
        suffix = [tok.eos_token_id] if eos else []
    else:
        suffix = list(wrap[1])
    return full + suffix, s, len(full)


def build_rows(dataset, items, tok, fmt, wrap, eos, n, max_tokens, max_templates):
    T = PT.n_templates(dataset)
    if max_templates:
        T = min(T, max_templates)
    kept, rows, skipped = [], [], {'too_long': 0, 'same_answers': 0}
    per_label = {0: 0, 1: 0}
    cap = None if dataset == 'copa' else (n + 1) // 2
    for it in items:
        if len(kept) >= n:
            break
        if cap is not None and per_label[it['label']] >= cap:
            continue
        enc, ok = [], True
        for t in range(T):
            question, answers = PT.render(dataset, t, it)
            e0 = encode(tok, question, answers[0], fmt, wrap, eos)
            e1 = encode(tok, question, answers[1], fmt, wrap, eos)
            if max(len(e0[0]), len(e1[0])) > max_tokens:
                skipped['too_long'] += 1; ok = False; break
            if e0[0] == e1[0]:
                skipped['same_answers'] += 1; ok = False; break
            enc.append((t, e0, e1))
        if not ok:
            continue
        kept.append(it)
        per_label[it['label']] += 1
        for t, e0, e1 in enc:
            rows.append({'item': it['id'], 'label': it['label'], 't': t, 'e': (e0, e1)})
    return kept, rows, T, skipped


def forward_rows(model, tok, seqs, layers_req, positions, batch_tokens, device):
    import torch
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = np.argsort([-len(x[0]) for x in seqs])
    lm_head = model.get_output_embeddings()
    dev = next(model.parameters()).device
    H = None
    LP = np.zeros(len(seqs), dtype=np.float32)
    layers = None
    i = 0
    while i < len(order):
        L = len(seqs[order[i]][0])
        bs = max(1, batch_tokens // L)
        idx = order[i:i + bs]
        i += bs
        # right padding leaves the real positions unchanged under causal attention
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for b, k in enumerate(idx):
            x = seqs[k][0]
            ids[b, :len(x)] = torch.tensor(x)
            mask[b, :len(x)] = 1
        with torch.no_grad():
            kw = dict(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                      output_hidden_states=True, return_dict=True)
            try:
                out = model(**kw, logits_to_keep=1)
            except TypeError:
                out = model(**kw)
            hs = out.hidden_states
            if layers is None:
                layers = sorted(set(layers_req(len(hs))))
                d = hs[-1].shape[-1]
                H = np.zeros((len(seqs), len(layers), len(positions), d), dtype=np.float16)
            for b, k in enumerate(idx):
                x, s, e = seqs[k]
                pos = {'answer': e - 1, 'final': len(x) - 1}
                take = [pos[p] for p in positions]
                st = torch.stack([hs[l][b, take, :] for l in layers], 0).float().cpu().numpy()
                np.clip(st, -65504, 65504, out=st)
                H[k] = st.astype(np.float16)
                # hs[-1] is post-norm, so lm_head on it gives the model's logits
                logits = lm_head(hs[-1][b, s - 1:e - 1, :].to(lm_head.weight.device)).float()
                lp = torch.log_softmax(logits, -1)
                tgt = torch.tensor(x[s:e], device=lp.device)
                LP[k] = lp.gather(1, tgt[:, None]).mean().item()
        del out, hs
        if device == 'cuda':
            torch.cuda.empty_cache()
    return H, LP, layers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--format', default='raw', choices=['raw', 'chat'])
    ap.add_argument('--datasets', nargs='+', default=PT.DATASETS, choices=PT.DATASETS)
    ap.add_argument('--data-dir', default=str(HERE / 'data'))
    ap.add_argument('--out-dir', default='./caches_text')
    ap.add_argument('--n', type=int, default=1000, help='items per dataset')
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--max-templates', type=int, default=None)
    ap.add_argument('--layer-stride', type=int, default=4,
                    help='keep every Nth layer, plus the last')
    ap.add_argument('--positions', nargs='+', default=['answer', 'final'], choices=['answer', 'final'])
    ap.add_argument('--no-eos', action='store_true')
    ap.add_argument('--batch-tokens', type=int, default=8192, help='padded tokens per forward pass')
    ap.add_argument('--limit', type=int, default=None, help='items per dataset, for smoke tests')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [ds for ds in args.datasets
            if args.overwrite or not (out_dir / f'hs_{args.model}_{args.format}_{ds}.npz').exists()]
    if not todo:
        print('all datasets cached'); return 0
    print(f'device={device} model={args.model} ({MODELS[args.model][0]}) format={args.format}')
    model, tok, proc = load(args.model, device)
    wrap = chat_wrap(args.model, tok, proc) if args.format == 'chat' else None
    if wrap:
        print(f'chat wrap: prefix={wrap[0]!r} suffix={tok.convert_ids_to_tokens(wrap[1])}')

    stride = args.layer_stride

    def layers_req(n_total):
        # hidden_states[0] is the embedding output
        return list(range(stride, n_total, stride)) + [n_total - 1]

    n = args.limit or args.n
    manifest_f = out_dir / f'manifest_{args.model}_{args.format}.json'
    manifest = json.loads(manifest_f.read_text(encoding='utf-8')) if manifest_f.exists() else {}
    for ds in todo:
        t0 = time.time()
        f_in = Path(args.data_dir) / f'{ds}.jsonl'
        if not f_in.exists():
            print(f'{ds}: no {f_in}, run text_ccs/prepare_data.py first', file=sys.stderr)
            return 1
        items = [json.loads(l) for l in open(f_in, encoding='utf-8')]
        kept, rows, T, skipped = build_rows(ds, items, tok, args.format, wrap,
                                            not args.no_eos, n, args.max_tokens,
                                            args.max_templates)
        if not rows:
            print(f'{ds}: nothing fits'); continue
        seqs = [r['e'][0] for r in rows] + [r['e'][1] for r in rows]
        H, LP, layers = forward_rows(model, tok, seqs, layers_req, args.positions,
                                     args.batch_tokens, device)
        R = len(rows)
        f = out_dir / f'hs_{args.model}_{args.format}_{ds}.npz'
        np.savez(f, h0=H[:R], h1=H[R:], lp0=LP[:R], lp1=LP[R:],
                 labels=np.array([r['label'] for r in rows]),
                 item_ids=np.array([r['item'] for r in rows]),
                 template_ids=np.array([r['t'] for r in rows]),
                 layers=np.array(layers), positions=np.array(args.positions),
                 model=np.array(args.model), format=np.array(args.format))
        x, s, e = rows[0]['e'][1]
        ex = tok.decode(x[:s]) + '[[' + tok.decode(x[s:e]) + ']]' + tok.decode(x[e:])
        manifest[ds] = {'items': len(kept), 'rows': R, 'templates': T,
                        'skipped': skipped, 'layers': layers,
                        'true_is_1': float(np.mean([it['label'] for it in kept])),
                        'example_answer1': ex,
                        'answer_token': tok.convert_ids_to_tokens([x[e - 1]])[0],
                        'final_token': tok.convert_ids_to_tokens([x[-1]])[0]}
        manifest_f.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'{ds}: {len(kept)} items x {T} templates -> {f}  shape {H[:R].shape}  '
              f'skipped {skipped}  {time.time() - t0:.0f}s')
        print(f'  example: {ex[-300:]!r}')
        del H; gc.collect()
    return 0


if __name__ == '__main__':
    sys.exit(main())
