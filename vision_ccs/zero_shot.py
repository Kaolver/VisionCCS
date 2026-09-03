"""Compute zero-shot Yes/No baseline logits and accuracy."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from reanalysis import build_pairs, CATEGORIES

YES_FORMS = ['Yes', ' Yes', 'yes', ' yes', 'YES']
NO_FORMS = ['No', ' No', 'no', ' no', 'NO']


def _first_token_ids(tokenizer, forms):
    """Ids of the first token of each surface form, deduplicated."""
    ids = []
    for f in forms:
        enc = tokenizer.encode(f, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


def find_image(image_id, image_dirs):
    name = f'{image_id:012d}.jpg' if isinstance(image_id, int) else image_id
    for d in image_dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def load_model(model_tag, cfg_paths, device):
    import torch
    if model_tag == 'llava':
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        proc = LlavaProcessor.from_pretrained(cfg_paths['llava'], use_fast=False)
        model = LlavaForConditionalGeneration.from_pretrained(
            cfg_paths['llava'],
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None)
    elif model_tag in ('qwen2', 'qwen2_5'):
        from transformers import AutoProcessor
        if model_tag == 'qwen2':
            from transformers import Qwen2VLForConditionalGeneration as Cls
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        proc = AutoProcessor.from_pretrained(cfg_paths[model_tag], trust_remote_code=True)
        model = Cls.from_pretrained(
            cfg_paths[model_tag],
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None,
            trust_remote_code=True)
    else:
        raise ValueError(f'unknown model {model_tag!r}')
    if device == 'cpu':
        model = model.to(device)
    model.eval()
    return model, proc


def build_inputs(model_tag, proc, image, question):
    """Build input prompt stopping before the answer with generation prompt on."""
    q = question.rstrip('?') + '?'
    if model_tag == 'llava':
        prompt = f'USER: <image>\n{q} Answer yes or no.\nASSISTANT:'
        return proc(images=image, text=prompt, return_tensors='pt')

    from qwen_vl_utils import process_vision_info
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': f'{q} Answer yes or no.'}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    return proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors='pt')


def calibrate(margin):
    """Burns' calibrated zero-shot, rank-based: the top half of items by margin
    are predicted yes, so the prediction rate is exactly 50/50 regardless of the
    model's yes-bias.

    Rank-based rather than `margin > median` because these logits are bf16, so
    margins are quantised and ties are common. With enough mass sitting exactly
    at the median a `>` threshold collapses to predicting one class for
    everything. Ranking breaks ties deterministically and cannot degenerate.
    """
    margin = np.asarray(margin, dtype=float)
    n = len(margin)
    order = np.argsort(margin, kind='mergesort')
    preds = np.zeros(n, dtype=int)
    preds[order[n - n // 2:]] = 1          # top half -> yes
    return preds


def calibrated_accuracy(margin, labels):
    """Calibrated zero-shot accuracy. Also returns the median margin, which is
    reported only as a descriptive statistic."""
    preds = calibrate(margin)
    return float((preds == np.asarray(labels)).mean()), float(np.median(margin))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='qwen2', choices=['llava', 'qwen2', 'qwen2_5'])
    ap.add_argument('--categories', nargs='+', default=CATEGORIES)
    ap.add_argument('--vqa-json', default='./vqav2_mapped.json')
    ap.add_argument('--image-dirs', nargs='+', default=[
        '/scratch-nvme/ml-datasets/coco/train/data',
        '/scratch-nvme/ml-datasets/coco/val/data'])
    ap.add_argument('--out-dir', default='./zeroshot')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap items per category (for a quick smoke test)')
    args = ap.parse_args()

    import torch
    from PIL import Image

    paths = {'llava': 'llava-hf/llava-1.5-7b-hf',
             'qwen2': 'Qwen/Qwen2-VL-7B-Instruct',
             'qwen2_5': 'Qwen/Qwen2.5-VL-7B-Instruct'}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}  model={args.model}')

    model, proc = load_model(args.model, paths, device)
    tok = proc.tokenizer
    yes_ids = _first_token_ids(tok, YES_FORMS)
    no_ids = _first_token_ids(tok, NO_FORMS)
    print(f'yes token ids {yes_ids} -> {tok.convert_ids_to_tokens(yes_ids)}')
    print(f'no  token ids {no_ids} -> {tok.convert_ids_to_tokens(no_ids)}')
    if not yes_ids or not no_ids or set(yes_ids) & set(no_ids):
        print('!! degenerate answer-token ids; aborting'); return 1

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    summary = {}

    for category in args.categories:
        pairs = build_pairs(args.vqa_json, category, mode='ccs')
        if args.limit:
            pairs = pairs[:args.limit]
        yl, nl, labels, skipped = [], [], [], 0

        for i, p in enumerate(pairs):
            path = find_image(p['image_id'], args.image_dirs)
            if path is None:
                skipped += 1
                continue
            try:
                image = Image.open(path).convert('RGB')
                inputs = build_inputs(args.model, proc, image, p['question'])
                inputs = {k: (v.to(device) if hasattr(v, 'to') else v)
                          for k, v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1].float().cpu().numpy()
                yl.append(float(logits[yes_ids].max()))
                nl.append(float(logits[no_ids].max()))
                labels.append(p['label'])
            except Exception as e:
                print(f"  error on {p['image_id']}: {e}")
                skipped += 1
            if device == 'cuda' and i % 200 == 0:
                torch.cuda.empty_cache()
            if i % 500 == 0:
                print(f'  {category}: {i}/{len(pairs)}')

        yl = np.array(yl); nl = np.array(nl); labels = np.array(labels)
        gc.collect()
        if len(labels) == 0:
            print(f'{category}: nothing extracted'); continue

        margin = yl - nl
        raw = float(((margin > 0).astype(int) == labels).mean())
        cal, thr = calibrated_accuracy(margin, labels)
        yes_rate = float((margin > 0).mean())

        f = out_dir / f'zeroshot_{args.model}_{category}.npz'
        np.savez(f, yes_logit=yl, no_logit=nl, labels=labels)
        summary[category] = {'n': int(len(labels)), 'skipped': int(skipped),
                             'raw_acc': raw, 'calibrated_acc': cal,
                             'median_margin': thr, 'predicted_yes_rate': yes_rate,
                             'true_yes_rate': float(labels.mean()), 'file': str(f)}
        print(f'\n{category}: n={len(labels)} skipped={skipped}')
        print(f'  raw acc        {raw:.1%}   (predicts yes {yes_rate:.1%} of the time)')
        print(f'  calibrated acc {cal:.1%}   (median margin {thr:+.3f})\n')

    (out_dir / f'zeroshot_{args.model}_summary.json').write_text(json.dumps(summary, indent=2))
    print('\n' + '=' * 70)
    print(f"{'category':26s} {'n':>6s} {'raw':>8s} {'calibrated':>11s}")
    for c, s in summary.items():
        print(f"{c:26s} {s['n']:6d} {s['raw_acc']:8.1%} {s['calibrated_acc']:11.1%}")
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
