"""Multi-layer hidden state extraction with token position tracking."""

# ============================================================================
# NOTE (review): ROLE OF THIS FILE
# 'v3' extractor for the layer/position sweep. Unlike vision_ccs.extract_in_
# batches (last layer, last token only) it keeps EVERY --layer-stride-th layer
# plus the last, at up to three token positions, stored as float16 with
# image_ids/question_ids. Output hs_<model>_<cat>.npz has shape
# (n, layers, positions, d) and is read by layer_sweep.load_cache_v3, NOT by
# reanalysis.find_cache (different file name and layout).
# ============================================================================
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from reanalysis import build_pairs, CATEGORIES
import prompts as P

MODEL_PATHS = {
    'qwen2': 'Qwen/Qwen2-VL-7B-Instruct',
    'qwen2_5': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'llava': 'llava-hf/llava-1.5-7b-hf',
}


def load_model(model_tag, device):
    """Load processor and model in bfloat16."""
    import torch
    # bfloat16 halves memory on GPU; on CPU fall back to float32
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


# ============================================================================
# NOTE (review): the token that closes the USER turn: <|im_end|> for Qwen,
# </s> (EOS) for LLaVA. Used by locate_positions to find the answer token.
# ============================================================================
def end_of_turn_id(model_tag, tok):
    """Token closing the user turn: <|im_end|> for Qwen, EOS for LLaVA."""
    if model_tag == 'llava':
        return tok.eos_token_id
    # Qwen chat template closes each turn with <|im_end|>; if the tokenizer does
    # not know it (returns None or the unk id) fall back to the generic EOS
    tid = tok.convert_tokens_to_ids('<|im_end|>')
    return tid if tid is not None and tid >= 0 else tok.eos_token_id


# ============================================================================
# NOTE (review): prompt construction mirrors vision_ccs.py exactly: LLaVA gets
# 'USER: <image>\n{text}' + EOS (vision_ccs.py ~375-376); Qwen uses the chat
# template with add_generation_prompt=False (~429 / ~483).
# NOTE POTENTIAL MISMATCH: load_model() loads LLaVA in bfloat16 on CUDA, while
# vision_ccs.py loads LLaVA in float16 (line ~149). Hidden states can differ
# in the low bits between the two extractors.
# ============================================================================
def build_inputs(model_tag, proc, image, text):
    """Build inputs for a statement prompt."""
    if model_tag == 'llava':
        # LLaVA: plain string prompt, statement followed directly by EOS (no
        # 'ASSISTANT:' cue) so the last token sits right after 'Yes'/'No'
        eos = proc.tokenizer.eos_token or ''
        return proc(images=image, text=f'USER: <image>\n{text}{eos}',
                    return_tensors='pt'), None
    from qwen_vl_utils import process_vision_info
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image}, {'type': 'text', 'text': text}]}]
    # Qwen: chat template WITHOUT generation prompt -> text ends '...Yes<|im_end|>\n'
    prompt = proc.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=False)
    imgs, vids = process_vision_info(messages)
    return proc(text=[prompt], images=imgs, videos=vids, padding=True,
                return_tensors='pt'), prompt


# ============================================================================
# NOTE (review): 'final' = last token of the sequence = what vision_ccs.py
# pools ([-1]). 'eot' = last occurrence of the end-of-turn id. 'answer' = the
# token right before it (the 'Yes'/'No' token). For Qwen the template ends
# '<|im_end|>\n', so final != eot; for LLaVA eot == final. If eot is absent
# everything falls back to the last token.
# ============================================================================
def locate_positions(input_ids, eot_id):
    """Map position names to token indices in sequence."""
    # walk backwards from the end to find the LAST end-of-turn token; the answer
    # token is the one just before it. Positions are indices into the sequence.
    ids = input_ids.tolist()
    last = len(ids) - 1
    eot = last
    for i in range(last, -1, -1):
        if ids[i] == eot_id:
            eot = i
            break
    # 'final' is what vision_ccs.py pools (index -1). For Qwen final = the '\n'
    # after <|im_end|>; for LLaVA eot == final. max(.., 0) guards a 1-token input.
    return {'answer': max(eot - 1, 0), 'eot': eot, 'final': last}


def find_image(image_id, image_dirs):
    name = f'{image_id:012d}.jpg' if isinstance(image_id, int) else image_id
    for d in image_dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


DISTRACTOR_WORDS = ('TRUE', 'FALSE')


def add_banner(image, word, seed=0):
    """Overlay a high-contrast text banner.

    PIL's built-in bitmap font, upscaled, so no TrueType file is needed.
    """
    from PIL import Image, ImageDraw
    img = image.convert('RGB').copy()
    W, H = img.size
    scale = max(2, W // 120)

    tile = Image.new('RGB', (len(word) * 6 + 4, 11), (0, 0, 0))
    ImageDraw.Draw(tile).text((2, 1), word, fill=(255, 255, 255))
    tile = tile.resize((tile.width * scale, tile.height * scale), Image.NEAREST)
    if tile.width > W:
        tile = tile.resize((W, int(tile.height * W / tile.width)), Image.NEAREST)
    img.paste(tile, (max(0, (W - tile.width) // 2), max(0, H // 20)))
    return img


def _derange(items, seed):
    """A permutation with no fixed point in VALUE space, as far as possible.

    Returns perm with items[perm[i]] != items[i] wherever achievable. Grouping
    by value and rotating by the largest block size is fixed-point-free by
    construction; the swap pass repairs blocks that wrapped onto themselves. A
    value holding more than half the data cannot be fully deranged, and the
    caller reports those residual fixed points.
    """
    rng = np.random.default_rng(seed)
    n = len(items)
    if n < 2:
        return np.arange(n)

    order = rng.permutation(n)
    by_value = {}
    for i in order:
        by_value.setdefault(items[i], []).append(int(i))
    blocks = sorted(by_value.values(), key=len, reverse=True)
    flat = [i for b in blocks for i in b]
    shift = max(len(b) for b in blocks)
    perm = np.array([flat[(k + shift) % n] for k in range(n)], dtype=int)

    # Repair residual collisions by swapping with a position safe for both.
    for k in range(n):
        if items[perm[k]] != items[flat[k]]:
            continue
        for j in range(n):
            if (items[perm[j]] != items[flat[k]]
                    and items[perm[k]] != items[flat[j]]):
                perm[k], perm[j] = perm[j], perm[k]
                break

    # perm is indexed by position in `flat`; map back to original positions.
    out = np.empty(n, dtype=int)
    for k, idx in enumerate(flat):
        out[idx] = perm[k]
    return out



# ============================================================================
# NOTE (review): one forward pass, then hidden_states[l][0, positions, :] for
# every requested layer; returned as float16 to keep caches small.
# ============================================================================
def extract_one(model, proc, model_tag, image, text, layers, pos_names, eot_id):
    """Run one forward pass and extract hidden states across selected layers and positions."""
    import torch
    inputs, _ = build_inputs(model_tag, proc, image, text)
    # with device_map='auto' the model may be sharded; the first parameter's
    # device is where the inputs must go
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)

    # out.hidden_states is a tuple of (1, seq, d) tensors, one per layer, index 0
    # = embedding output. Gather the requested positions from each requested layer
    # and stack -> (n_layers, n_positions, d); float16 to keep the cache small.
    idx = locate_positions(inputs['input_ids'][0], eot_id)
    take = [idx[p] for p in pos_names]
    stack = torch.stack([out.hidden_states[l][0, take, :] for l in layers], dim=0)
    return stack.float().cpu().numpy().astype(np.float16), idx


# ============================================================================
# NOTE (review): --shuffle-images is a CONTROL that pairs each question with
# another image of the same category. The self-match fix (perm[i] =
# perm[i+1]) makes perm a non-permutation: that image is then used twice and
# one image not at all. Harmless for a control, but perm is no longer a
# bijection. --limit N is a smoke test; --layer-stride controls cache size.
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
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
    ap.add_argument('--templates', nargs='+', default=['plain'],
                    help=f'surface forms for the contrast pairs; "all" expands to '
                         f'{P.ALL_TEMPLATES}. See prompts.py.')
    ap.add_argument('--shuffle-images', action='store_true',
                    help='control: permute image<->question pairing within category')
    ap.add_argument('--shuffle-seed', type=int, default=1234)
    ap.add_argument('--distractor', default='none', choices=['none', 'banner'],
                    help='control (Farquhar et al. 2312.10029): stamp a TRUE/FALSE '
                         'banner assigned independently of the answer. CCS is then '
                         'scored against the banner too; above chance means it '
                         'found the prominent feature, not knowledge.')
    ap.add_argument('--distractor-seed', type=int, default=99)
    ap.add_argument('--limit', type=int, default=None, help='smoke test')
    args = ap.parse_args()

    import torch
    from PIL import Image

    templates = P.template_names(args.templates)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}  model={args.model}  shuffle_images={args.shuffle_images}')
    print(f'templates: {templates}')
    model, proc = load_model(args.model, device)
    tok = proc.tokenizer
    eot_id = end_of_turn_id(args.model, tok)
    print(f'end-of-turn token id {eot_id} -> {tok.convert_ids_to_tokens([eot_id])}')

    # +1 because hidden_states also contains the embedding layer at index 0;
    # VLMs keep the LLM config under text_config, plain LLMs at the top level
    n_layers_total = model.config.text_config.num_hidden_layers + 1 \
        if hasattr(model.config, 'text_config') else \
        model.config.num_hidden_layers + 1
    # keep every stride-th layer, and always the last one (what CCS normally uses)
    layers = list(range(0, n_layers_total, args.layer_stride))
    if layers[-1] != n_layers_total - 1:
        layers.append(n_layers_total - 1)
    print(f'capturing {len(layers)} of {n_layers_total} layers: {layers}')
    print(f'positions: {args.positions}')

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    manifest = {}

    for category in args.categories:
        pairs = build_pairs(args.vqa_json, category, mode='ccs')
        if args.limit:
            pairs = pairs[:args.limit]

        if args.shuffle_images:
            # control: give each question a DIFFERENT image of the same
            # category. perm[i] is the source item whose image question i
            # receives. _derange returns a TRUE permutation, so each image
            # is still used exactly as often as before.
            imgs = [p['image_id'] for p in pairs]
            perm = _derange(imgs, args.shuffle_seed)
            pairs = [{**p, 'image_id': imgs[perm[i]]} for i, p in enumerate(pairs)]
            n_fixed = sum(1 for i in range(len(imgs)) if imgs[perm[i]] == imgs[i])
            print(f'  shuffled-image control: {len(imgs)} questions repaired to a '
                  f'different image, {n_fixed} unavoidable fixed points '
                  f'({n_fixed / max(len(imgs), 1):.2%})')

        # One ROW per (item, template), not an extra array axis, so templates can
        # be added later without rewriting caches and subset at load time. qid
        # keeps an item's sibling rows joinable for the grouped split.
        Ps, Ns, y, iid, qid, tid, did = [], [], [], [], [], [], []
        failures, pos_log = [], {}

        # Fair coin per item, independent of the label: the banner carries zero
        # information about truth, so scoring above chance against it is a tell.
        drng = np.random.default_rng(args.distractor_seed)
        dlab_all = (drng.integers(0, 2, len(pairs)) if args.distractor != 'none'
                    else np.zeros(len(pairs), dtype=int))

        # Ps/Ns: per-row (layers, positions, d) arrays for the Yes / No
        # statement; y: label; iid/qid/tid: ids stored so later splits can
        # group by image or question and select templates.
        for i, p in enumerate(pairs):
            path = find_image(p['image_id'], args.image_dirs)
            if path is None:
                failures.append((p['question_id'], 'missing_image'))
                continue
            try:
                image = Image.open(path).convert('RGB')
                    # the two contrast statements: same image, same prompt
                    # shape as vision_ccs.py, rendered from prompts.py
                if args.distractor == 'banner':
                    image = add_banner(image, DISTRACTOR_WORDS[dlab_all[i]])
                for t in templates:
                    ph, idx = extract_one(
                        model, proc, args.model, image,
                        P.render(t, p['question'], True),
                        layers, args.positions, eot_id)
                    nh, _ = extract_one(
                        model, proc, args.model, image,
                        P.render(t, p['question'], False),
                        layers, args.positions, eot_id)
                    if t not in pos_log:
                        pos_log[t] = idx
                        print(f'  template {t!r} position map: {idx}   '
                              f'example: {P.render(t, p["question"], True)!r}')
                    Ps.append(ph); Ns.append(nh); y.append(p['label'])
                    iid.append(p['image_id']); qid.append(p['question_id'])
                    tid.append(t); did.append(int(dlab_all[i]))
            except Exception as e:
                failures.append((p['question_id'], f'{type(e).__name__}: {e}'))
            if device == 'cuda' and i % 100 == 0:
                torch.cuda.empty_cache()
            if i % 500 == 0:
                print(f'  {category}: {i}/{len(pairs)}  kept={len(Ps)}  failed={len(failures)}')

        if not Ps:
            print(f'{category}: nothing extracted'); continue
        gc.collect()

        suffix = '_shuffled' if args.shuffle_images else ''
        if args.distractor != 'none':
            suffix += f'_distract-{args.distractor}'
        if templates != ['plain']:
            suffix += '_t' + '-'.join(templates)
        f = out_dir / f'hs_{args.model}_{category}{suffix}.npz'
        # np.stack(Ps) -> (n rows, layers, positions, d). layers/positions arrays
        # are saved alongside so layer_sweep can map indices back to layer
        # numbers / position names.
        np.savez(f,
                 pos_hiddens=np.stack(Ps), neg_hiddens=np.stack(Ns),
                 labels=np.array(y), layers=np.array(layers),
                 positions=np.array(args.positions),
                 image_ids=np.array(iid), question_ids=np.array(qid),
                 template_ids=np.array(tid), templates=np.array(templates),
                 distractor_labels=np.array(did),
                 distractor=np.array(args.distractor))
        shape = np.stack(Ps).shape
        print(f'\n{category}: wrote {f}')
        print(f'  shape {shape} (n_rows, layers, positions, d)  '
              f'{f.stat().st_size / 1e9:.2f} GB')
        print(f'  kept {len(Ps)} rows from {len(set(qid))}/{len(pairs)} items '
              f'x {len(templates)} templates   failed {len(failures)}')
        by_reason = {}
        for _, why in failures:
            by_reason[why.split(':')[0]] = by_reason.get(why.split(':')[0], 0) + 1
        if by_reason:
            print(f'  failure breakdown: {by_reason}')
        manifest[category] = {'file': str(f), 'shape': list(shape),
                              'rows': len(Ps), 'items': len(set(qid)),
                              'templates': templates,
                              'failed': len(failures),
                              'failure_reasons': by_reason,
                              'positions_example': pos_log,
                              'yes_frac': float(np.mean(y))}

    tag = '_shuffled' if args.shuffle_images else ''
    if templates != ['plain']:
        tag += '_t' + '-'.join(templates)
    mf = out_dir / f'manifest_{args.model}{tag}.json'
    mf.write_text(json.dumps({'model': args.model, 'layers': layers,
                              'positions': args.positions,
                              'templates': templates,
                              'shuffled': args.shuffle_images,
                              'categories': manifest}, indent=2))
    print(f'\nWrote {mf}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
