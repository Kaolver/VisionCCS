"""Phase 1 extraction: all layers, explicit token positions, optional image shuffle.

Replaces the three near-duplicate extractors in vision_ccs.py. Three things the
old path got wrong or never did:

1. LAYER. The old code took hidden_states[-1] -- the final layer, whose entire
   job is to produce logits for the next token. Zero-shot IS that layer through
   the LM head, so probing it and comparing against zero-shot largely compares
   the readout layer with itself (measured agreement: 90.9%). CCS's claim is
   about INTERNAL representations, which needs mid-layer probing. All layers are
   captured in the same forward pass, so the sweep is free.

2. TOKEN POSITION. The old code took [:, -1, :]. For Qwen the templated text
   ends "...Yes<|im_end|>\\n", so that pooled a NEWLINE, two tokens past the
   answer -- verified, and not what the code comments claimed. We now locate the
   end-of-turn token explicitly and store both the answer token and the trailing
   token, making position an ablation axis instead of an assumption.

3. IMAGE SHUFFLE (control). --shuffle-images permutes which image goes with
   which question, keeping questions and labels intact. VQAv2 has a well-known
   language prior ("Is this a hospital?" is guessable from text alone); if
   accuracy survives shuffling, the probe never needed the image.

Output: caches_v3/hs_{model}_{category}{_shuffled}.npz with
    pos_hiddens, neg_hiddens : (n, n_layers, n_positions, d) float16
    labels                   : (n,)
    layers, positions        : which layer indices / position names, in order
    image_ids, question_ids  : (n,) provenance, so rows align with zero_shot.py

Storage: qwen2 at --layer-stride 2 is ~2 GB per model per condition. Use
--layer-stride 4 if space is tight; the curve just gets coarser.

    python extract.py --model qwen2
    python extract.py --model qwen2 --shuffle-images
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from reanalysis import build_pairs, CATEGORIES

MODEL_PATHS = {
    'qwen2': 'Qwen/Qwen2-VL-7B-Instruct',
    'qwen2_5': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'llava': 'llava-hf/llava-1.5-7b-hf',
}


def load_model(model_tag, device):
    """bf16 for every model -- the old code mixed fp16 (llava) with bf16 (qwen2)
    and torch_dtype='auto' (qwen2_5), an avoidable confound in a cross-model
    comparison."""
    import torch
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    path = MODEL_PATHS[model_tag]

    if model_tag == 'llava':
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        proc = LlavaProcessor.from_pretrained(path, use_fast=False)
        model = LlavaForConditionalGeneration.from_pretrained(
            path, torch_dtype=dtype, device_map='auto' if device == 'cuda' else None)
    else:
        from transformers import AutoProcessor
        if model_tag == 'qwen2':
            from transformers import Qwen2VLForConditionalGeneration as Cls
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = Cls.from_pretrained(path, torch_dtype=dtype,
                                    device_map='auto' if device == 'cuda' else None,
                                    trust_remote_code=True)
    if device == 'cpu':
        model = model.to(device)
    model.eval()
    return model, proc


def end_of_turn_id(model_tag, tok):
    """Token that closes the user turn: <|im_end|> for Qwen, EOS for LLaVA."""
    if model_tag == 'llava':
        return tok.eos_token_id
    tid = tok.convert_tokens_to_ids('<|im_end|>')
    return tid if tid is not None and tid >= 0 else tok.eos_token_id


def build_inputs(model_tag, proc, image, text):
    """Statement + end-of-turn, matching the ata extraction path exactly
    (add_generation_prompt=False, EOS appended for llava)."""
    if model_tag == 'llava':
        eos = proc.tokenizer.eos_token or ''
        return proc(images=image, text=f'USER: <image>\n{text}{eos}',
                    return_tensors='pt'), None
    from qwen_vl_utils import process_vision_info
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image}, {'type': 'text', 'text': text}]}]
    prompt = proc.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=False)
    imgs, vids = process_vision_info(messages)
    return proc(text=[prompt], images=imgs, videos=vids, padding=True,
                return_tensors='pt'), prompt


def locate_positions(input_ids, eot_id):
    """Map position names to indices in the sequence.

    'answer' is the token immediately before the LAST end-of-turn token, i.e.
    the Yes/No the statement ends with. 'eot' is that end-of-turn token.
    'final' is the last token overall -- what the old code used, kept so the
    old behaviour stays reproducible.
    """
    ids = input_ids.tolist()
    last = len(ids) - 1
    eot = last
    for i in range(last, -1, -1):
        if ids[i] == eot_id:
            eot = i
            break
    return {'answer': max(eot - 1, 0), 'eot': eot, 'final': last}


def find_image(image_id, image_dirs):
    name = f'{image_id:012d}.jpg' if isinstance(image_id, int) else image_id
    for d in image_dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def extract_one(model, proc, model_tag, image, text, layers, pos_names, eot_id):
    """One forward pass -> (n_layers, n_positions, d) float16."""
    import torch
    inputs, _ = build_inputs(model_tag, proc, image, text)
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)

    idx = locate_positions(inputs['input_ids'][0], eot_id)
    take = [idx[p] for p in pos_names]
    # hidden_states is a tuple of (n_layers+1) tensors, each (1, seq, d)
    stack = torch.stack([out.hidden_states[l][0, take, :] for l in layers], dim=0)
    return stack.float().cpu().numpy().astype(np.float16), idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='qwen2', choices=list(MODEL_PATHS))
    ap.add_argument('--categories', nargs='+', default=CATEGORIES)
    ap.add_argument('--vqa-json', default='./vqav2_mapped.json')
    ap.add_argument('--image-dirs', nargs='+', default=[
        '/scratch-nvme/ml-datasets/coco/train/data',
        '/scratch-nvme/ml-datasets/coco/val/data'])
    ap.add_argument('--out-dir', default='./caches_v3')
    ap.add_argument('--layer-stride', type=int, default=2,
                    help='keep every Nth layer (1 = all). Controls cache size.')
    ap.add_argument('--positions', nargs='+', default=['answer', 'final'],
                    choices=['answer', 'eot', 'final'])
    ap.add_argument('--shuffle-images', action='store_true',
                    help='control: permute image<->question pairing within category')
    ap.add_argument('--shuffle-seed', type=int, default=1234)
    ap.add_argument('--limit', type=int, default=None, help='smoke test')
    args = ap.parse_args()

    import torch
    from PIL import Image

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}  model={args.model}  shuffle_images={args.shuffle_images}')
    model, proc = load_model(args.model, device)
    tok = proc.tokenizer
    eot_id = end_of_turn_id(args.model, tok)
    print(f'end-of-turn token id {eot_id} -> {tok.convert_ids_to_tokens([eot_id])}')

    n_layers_total = model.config.text_config.num_hidden_layers + 1 \
        if hasattr(model.config, 'text_config') else \
        model.config.num_hidden_layers + 1
    layers = list(range(0, n_layers_total, args.layer_stride))
    if layers[-1] != n_layers_total - 1:
        layers.append(n_layers_total - 1)     # always keep the final layer
    print(f'capturing {len(layers)} of {n_layers_total} layers: {layers}')
    print(f'positions: {args.positions}')

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    manifest = {}

    for category in args.categories:
        pairs = build_pairs(args.vqa_json, category, mode='ccs')
        if args.limit:
            pairs = pairs[:args.limit]

        if args.shuffle_images:
            rng = np.random.default_rng(args.shuffle_seed)
            imgs = [p['image_id'] for p in pairs]
            perm = rng.permutation(len(imgs))
            # derangement-ish: retry any fixed points once
            for i, j in enumerate(perm):
                if imgs[j] == imgs[i]:
                    perm[i] = perm[(i + 1) % len(perm)]
            pairs = [{**p, 'image_id': imgs[perm[i]]} for i, p in enumerate(pairs)]

        P, N, y, iid, qid, failures, pos_log = [], [], [], [], [], [], None
        for i, p in enumerate(pairs):
            path = find_image(p['image_id'], args.image_dirs)
            if path is None:
                failures.append((p['question_id'], 'missing_image'))
                continue
            try:
                image = Image.open(path).convert('RGB')
                q = p['question'].rstrip('?')
                ph, idx = extract_one(model, proc, args.model, image, f'{q}? Yes',
                                      layers, args.positions, eot_id)
                nh, _ = extract_one(model, proc, args.model, image, f'{q}? No',
                                    layers, args.positions, eot_id)
                if pos_log is None:
                    pos_log = idx
                    ids = proc(text=[f'{q}? Yes'], return_tensors='pt') \
                        if args.model == 'llava' else None
                    print(f'  first item position map: {idx}')
                P.append(ph); N.append(nh); y.append(p['label'])
                iid.append(p['image_id']); qid.append(p['question_id'])
            except Exception as e:
                # logged and counted, not silently swallowed as before
                failures.append((p['question_id'], f'{type(e).__name__}: {e}'))
            if device == 'cuda' and i % 100 == 0:
                torch.cuda.empty_cache()
            if i % 500 == 0:
                print(f'  {category}: {i}/{len(pairs)}  kept={len(P)}  failed={len(failures)}')

        if not P:
            print(f'{category}: nothing extracted'); continue
        gc.collect()

        suffix = '_shuffled' if args.shuffle_images else ''
        f = out_dir / f'hs_{args.model}_{category}{suffix}.npz'
        np.savez(f,
                 pos_hiddens=np.stack(P), neg_hiddens=np.stack(N),
                 labels=np.array(y), layers=np.array(layers),
                 positions=np.array(args.positions),
                 image_ids=np.array(iid), question_ids=np.array(qid))
        shape = np.stack(P).shape
        print(f'\n{category}: wrote {f}')
        print(f'  shape {shape} (n, layers, positions, d)  '
              f'{f.stat().st_size / 1e9:.2f} GB')
        print(f'  kept {len(P)}/{len(pairs)}   failed {len(failures)}')
        by_reason = {}
        for _, why in failures:
            by_reason[why.split(':')[0]] = by_reason.get(why.split(':')[0], 0) + 1
        if by_reason:
            print(f'  failure breakdown: {by_reason}')
        manifest[category] = {'file': str(f), 'shape': list(shape),
                              'kept': len(P), 'failed': len(failures),
                              'failure_reasons': by_reason,
                              'positions_example': pos_log,
                              'yes_frac': float(np.mean(y))}

    mf = out_dir / f'manifest_{args.model}{"_shuffled" if args.shuffle_images else ""}.json'
    mf.write_text(json.dumps({'model': args.model, 'layers': layers,
                              'positions': args.positions,
                              'shuffled': args.shuffle_images,
                              'categories': manifest}, indent=2))
    print(f'\nWrote {mf}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
