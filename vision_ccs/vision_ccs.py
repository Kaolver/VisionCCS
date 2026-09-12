import torch
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from qwen_vl_utils import process_vision_info
import tempfile
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


dataset = json.load(open('vqav2_mapped.json', 'r'))
distribution = {category: len(items) for category, items in dataset.items()}

CONFIG = {
    'n_samples_object_detection': distribution.get('object_detection', 0),
    'n_samples_attribute_recognition': distribution.get('attribute_recognition', 0),
    'n_samples_spatial_recognition': distribution.get('spatial_recognition', 0),
    'batch_size': 40,
    
    # Cache control
    'use_cache': False,
    
    # Paths
    'vqa_json': './vqav2_mapped.json',
    'image_dirs': [
        '/scratch-nvme/ml-datasets/coco/train/data',
        '/scratch-nvme/ml-datasets/coco/val/data',
    ],
    'cache_dir': './hidden_states_cache_final',
    'categories': ['object_detection', 'attribute_recognition', 'spatial_recognition'],
    
    # Model
    'model_llava': 'llava-hf/llava-1.5-7b-hf',
    'model_qwen2': 'Qwen/Qwen2-VL-7B-Instruct',
    'model_qwen2_5': 'Qwen/Qwen2.5-VL-7B-Instruct',

    # Choose model: 'llava', 'qwen2', or 'qwen2_5'
    'chosen_model': 'qwen2',
    
    'hf_token': None,


    # ==========================================================================
    # CHANGED (CCS alignment): train/test split is now a random 60/40 split, as
    # in the CCS paper (Sec 3.1: "randomly split each dataset into an
    # unsupervised training set (60% of the data) and test set (40%)").
    # Previously this was 0.7 (a 70/30 split); the original notebook uses an
    # unshuffled 50/50 cut.
    # ==========================================================================
    'train_split': 0.6,


    'ccs_epochs': 1000,
    'ccs_ntries': 10,


    # ==========================================================================
    # CHANGED (CCS alignment): learning rate raised from 1e-3 (the original
    # notebook's default) to 0.01, which is what the paper (Sec 3.1) reports
    # for optimizing CCS with AdamW.
    # ==========================================================================
    'ccs_lr': 1e-2,


    'ccs_weight_decay': 0.01,


    # ==========================================================================
    # CHANGED (CCS alignment): new options.
    # - 'ccs_var_normalize': after mean-centering, also divide features by
    #   their std. Matches the paper (Sec 2.2: "In practice we also normalize
    #   the scale of the features") and the report's claim that activations
    #   are normalised "to ensure comparable scale". The original notebook
    #   exposes this as var_normalize (default False there).
    # - 'random_seed': seed for the new random subsampling and the random split.
    # - 'run_lr_sanity_check': run the supervised logistic-regression check
    #   from the original notebook before CCS ("if logistic regression
    #   accuracy is bad, there's no hope of CCS doing well"). Diagnostic only;
    #   does not affect CCS training or its reported accuracy.
    # ==========================================================================
    'ccs_var_normalize': True,
    'random_seed': 42,
    'run_lr_sanity_check': True,
}


def load_vqa_data(config, category):
    """Load data for a specific category from the categorized VQA JSON."""
    data_path = Path(config['vqa_json'])
    with open(data_path, 'r') as f:
        all_data = json.load(f)
        vqa_data = all_data[category]
    
    n_samples_key = f'n_samples_{category}'
    n_samples = config.get(n_samples_key, len(vqa_data))


    # ==========================================================================
    # CHANGED (CCS alignment): the paper (Sec 3.1) balances the labels 50/50 and
    # randomly subsamples before splitting (the original notebook also samples
    # examples at random). Previously a deterministic prefix of the file was
    # taken, with whatever yes/no class balance the dataset happened to contain
    # — on an imbalanced test set the accuracy-orientation flip max(acc, 1-acc)
    # makes the majority-class rate a trivial floor, inflating results.
    # Now: split items by answer, randomly subsample an equal number of "yes"
    # and "no" items (seeded for reproducibility), and shuffle them together.
    # ==========================================================================
    rng = np.random.default_rng(config['random_seed'])
    yes_items = [item for item in vqa_data if item['answer'] == 'yes']
    no_items = [item for item in vqa_data if item['answer'] != 'yes']
    n_per_class = min(len(yes_items), len(no_items), n_samples // 2)
    yes_sel = [yes_items[i] for i in rng.permutation(len(yes_items))[:n_per_class]]
    no_sel = [no_items[i] for i in rng.permutation(len(no_items))[:n_per_class]]
    samples = yes_sel + no_sel
    samples = [samples[i] for i in rng.permutation(len(samples))]



    # Create contrast pairs
    pairs = []
    for item in samples:
        q = item['question'].rstrip('?')
        img_id = item['image_id']
        if isinstance(img_id, int):
            img_id = f"{img_id:012d}.jpg"

        pairs.append({
            'image_id': img_id,
            'question': q,
            'pos_text': f"{q}? Yes",
            'neg_text': f"{q}? No",
            'label': 1 if item['answer'] == 'yes' else 0
        })
    
    return pairs


def find_image(image_id, image_dirs):
    """Search for image in the given config image directories."""
    for img_dir in image_dirs:
        image_path = Path(img_dir) / image_id
        if image_path.exists():
            return image_path
    return None


def extract_in_batches(pairs, config, category):
    """Extract hidden states from LLaVA in batches with memory management."""
    print(f"\n{'='*70}")
    print(f"EXTRACTING HIDDEN STATES: {category.upper()}")
    print(f"{'='*70}")
    
    cache_dir = Path(config['cache_dir'])
    cache_dir.mkdir(exist_ok=True)
    
    # Check cache
    n = len(pairs)
    model_tag = config['chosen_model']


    # ==========================================================================
    # CHANGED (CCS alignment): cache filename tag bumped ("_ccs_aligned").
    # Hidden states are now pooled at the end of the user turn instead of after
    # the generation prompt, so earlier caches are incompatible.
    #
    # This path stores the FINAL layer at ONE position; extract.py writes the
    # richer cache new experiments should use. Kept for reproducibility.
    # ==========================================================================
    cache_file = cache_dir / f"cache_{category}_{n}_{model_tag}_ccs_aligned.npz"
    
    if config['use_cache'] and cache_file.exists():
        print("✓ Found cached hidden states!")
        print(f"  Loading from: {cache_file}")
        data = np.load(cache_file)
        print(f"  Loaded: pos={data['pos_hiddens'].shape}, neg={data['neg_hiddens'].shape}")
        return data['pos_hiddens'], data['neg_hiddens'], data['labels']
    
    if not config['use_cache']:
        print("⚠ Cache disabled (use_cache=False). Extracting new...")
    else:
        print("⚠ No cache found. Starting extraction...")
    
    print(f"\nProcessing {len(pairs)} samples in batches of {config['batch_size']}")
    print(f"Searching in {len(config['image_dirs'])} image directories")

    # Load model according to chosen_model
    print(f"LOADING MODEL: {config['chosen_model']}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cpu":
        print("WARNING: Running on CPU!")

    chosen_model = config['chosen_model']

    if chosen_model == 'llava':
        processor = LlavaProcessor.from_pretrained(
            config['model_llava'],
            use_fast=False
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            config['model_llava'],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)
            
    elif chosen_model == 'qwen2':
        hf_token = config.get('hf_token') or os.environ.get('HF_TOKEN')
        model_path = config['model_qwen2']
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=hf_token,
        )
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            token=hf_token,
        )
        
        if device == "cpu":
            model = model.to(device)

    elif chosen_model == 'qwen2_5':
        hf_token = config.get('hf_token') or os.environ.get('HF_TOKEN')
        model_path = config['model_qwen2_5']
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            token=hf_token,
        )
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else None,
            token=hf_token,
        )
        
        if device == "cpu":
            model = model.to(device)
        
    else:
        raise ValueError(f"Unsupported chosen_model in CONFIG: {chosen_model}")

    model.eval()
    print("✓ Model loaded successfully\n")
    
    # Extract hidden states
    all_pos_hiddens = []
    all_neg_hiddens = []
    all_labels = []
    skipped = []
    
    for i in tqdm(range(0, len(pairs), config['batch_size']), desc="Batches"):
        batch_pairs = pairs[i:i + config['batch_size']]
        
        for pair in batch_pairs:
            image_id = pair['image_id']
            image_path = find_image(image_id, config['image_dirs'])
            
            if image_path is None:
                skipped.append(image_id)
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')

                if config['chosen_model'] == 'llava':
                    pos_h = extract_one_llava(
                        model, processor, image, pair['pos_text'], device
                    )
                    neg_h = extract_one_llava(
                        model, processor, image, pair['neg_text'], device
                    )
                
                elif config['chosen_model'] == 'qwen2':
                    pos_h = extract_one_qwen2(
                        model, processor, image, pair['pos_text'], device
                    )
                    neg_h = extract_one_qwen2(
                        model, processor, image, pair['neg_text'], device
                    )
                
                elif config['chosen_model'] == 'qwen2_5':
                    pos_h = extract_one_qwen2_5(
                        model, processor, image, pair['pos_text'], device
                    )
                    neg_h = extract_one_qwen2_5(
                        model, processor, image, pair['neg_text'], device
                    )
                
                all_pos_hiddens.append(pos_h)
                all_neg_hiddens.append(neg_h)
                all_labels.append(pair['label'])
                
            except Exception as e:
                print(f"\n✗ Error processing {image_id}: {e}")
                skipped.append(image_id)
                continue
        
        # Memory management
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # Unload model
    del model
    del processor
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n{'='*70}")
    print(f"✓ Successfully processed: {len(all_pos_hiddens)}/{len(pairs)}")
    print(f"✗ Skipped (missing/error): {len(skipped)}/{len(pairs)}")
    
    if skipped and len(skipped) <= 10:
        print(f"\nSkipped images: {', '.join(skipped[:10])}")
    elif skipped:
        print(f"\nFirst 10 skipped: {', '.join(skipped[:10])}...")
    
    # Convert to arrays
    pos_hiddens = np.array(all_pos_hiddens)
    neg_hiddens = np.array(all_neg_hiddens)
    labels = np.array(all_labels)
    
    print(f"\nExtracted shapes:")
    print(f"  Positive: {pos_hiddens.shape}")
    print(f"  Negative: {neg_hiddens.shape}")
    print(f"  Labels: {labels.shape}")
    
    # Save cache
    np.savez(cache_file, 
             pos_hiddens=pos_hiddens, 
             neg_hiddens=neg_hiddens, 
             labels=labels)
    print(f"\nCached to: {cache_file}")
    
    return pos_hiddens, neg_hiddens, labels


def extract_one_llava(model, processor, image, text, device):
    """Extract hidden state from LLaVA for a single image-text pair."""
    
    # ==========================================================================
    # CHANGED (CCS alignment): the original CCS appends tokenizer.eos_token
    # directly after the statement for decoder models and pools the hidden
    # state at that last token. Previously the prompt ended with
    # "\nASSISTANT:", so the pooled last token sat after generation-prompt
    # tokens rather than right after the "Yes"/"No" answer. The trailing
    # "ASSISTANT:" is removed and EOS is appended, so the last token now
    # directly follows the answer (+ EOS), as in the original.
    # ==========================================================================
    eos = processor.tokenizer.eos_token or ""
    prompt = f"USER: <image>\n{text}{eos}"



    # Process inputs
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        padding=True
    )

    # Move tensors to device if necessary
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    # Extract hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

        # Use LAST TOKEN hidden state
        hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0)
    
    return hidden.cpu().float().numpy()


def extract_one_qwen2(model, processor, image, text, device):
    """Extract hidden state from Qwen2-VL for a single image-text pair."""
    
    # Qwen2-VL uses chat message format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    
    # Apply chat template
    # ==========================================================================
    # CHANGED (CCS alignment): add_generation_prompt is now False (was True).
    # Original CCS pools the last-token hidden state right after the statement
    # (+ EOS for decoder models); with the generation prompt, the pooled last
    # token was the end of the assistant header instead of the answer.
    #
    # NOTE: Qwen's template appends a newline after <|im_end|>, so the pooled
    # last token is that newline -- two positions past the answer, not directly
    # after it. Left as-is because it produced the existing caches; the
    # corrected positions live in extract.py:locate_positions.
    # ==========================================================================
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )



    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Prepare inputs
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Move to device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # Extract hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Use LAST TOKEN hidden state
        hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0)
    
    return hidden.cpu().float().numpy()


def extract_one_qwen2_5(model, processor, image, text, device):
    """Extract hidden state from Qwen2.5-VL for a single image-text pair."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    
    # Prepare inputs using Qwen's format
    # ==========================================================================
    # CHANGED (CCS alignment): add_generation_prompt is now False (was True),
    # same rationale as for the other models — the pooled state is no longer
    # taken after an assistant generation prompt.
    #
    # NOTE: as in extract_one_qwen2, the pooled last token is the newline Qwen
    # appends after <|im_end|>, not the answer token.
    # ==========================================================================
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move to device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # Extract hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Use LAST TOKEN hidden state
    hidden_states = outputs.hidden_states[-1]
    hidden = hidden_states[0, -1].squeeze(0)
    
    return hidden.cpu().float().numpy()


def train_ccs_probe(pos_hiddens, neg_hiddens, labels, config):
    print(f"\n{'='*70}")
    print(f"TRAINING CCS PROBE")
    print(f"{'='*70}")

    # Convert to tensors
    pos_hiddens = torch.FloatTensor(pos_hiddens)
    neg_hiddens = torch.FloatTensor(neg_hiddens)
    
    n = len(labels)
    indices = np.arange(n)


    # ==========================================================================
    # CHANGED (CCS alignment): the split is now a plain random split with NO
    # stratification (stratify=labels removed). Stratifying used the ground-
    # truth labels to construct the split, injecting a supervised signal into
    # an otherwise unsupervised pipeline; the original CCS is label-free until
    # evaluation (the notebook uses an unshuffled 50/50 cut, the paper a random
    # 60/40 split — the 60/40 ratio is set via 'train_split' in CONFIG).
    # ==========================================================================
    train_idx, test_idx = train_test_split(
        indices,
        test_size=1 - config['train_split'],
        random_state=config['random_seed']
    )



    pos_train_raw = pos_hiddens[train_idx]
    neg_train_raw = neg_hiddens[train_idx]
    pos_test_raw = pos_hiddens[test_idx]
    neg_test_raw = neg_hiddens[test_idx]
    labels_test = labels[test_idx]

    # ==========================================================================
    # CHANGED (CCS alignment): normalization now optionally divides by the
    # per-class std after mean-centering ('ccs_var_normalize', default True),
    # exactly like CCS.normalize with var_normalize in the original notebook.
    # The paper (Sec 2.2) states feature scale is normalized in practice; the
    # report also claims activations are normalised "to ensure comparable
    # scale". Previously only mean-centering was implemented.
    # (unbiased=False matches numpy's std used by the original.)
    # ==========================================================================
    def normalize(x):
        x = x - x.mean(dim=0)
        if config['ccs_var_normalize']:
            # eps guards constant dimensions; without it a dead dim yields nan
            # and poisons every gradient rather than failing.
            x = x / (x.std(dim=0, unbiased=False) + 1e-8)
        return x

    pos_train = normalize(pos_train_raw)
    neg_train = normalize(neg_train_raw)
    pos_test = normalize(pos_test_raw)
    neg_test = normalize(neg_test_raw)


    # ==========================================================================
    # CHANGED (CCS alignment, minor): the probe is now trained on GPU when
    # available, like the original (CCS(..., device="cuda")); previously the
    # probe and hidden states always stayed on CPU (numerically equivalent,
    # only slower).
    # ==========================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_train = pos_train.to(device)
    neg_train = neg_train.to(device)
    pos_test = pos_test.to(device)
    neg_test = neg_test.to(device)



    n_train = len(train_idx)
    n_test = len(test_idx)
    n_train_pos = (labels[train_idx] == 1).sum()
    n_train_neg = (labels[train_idx] == 0).sum()
    n_test_pos = (labels_test == 1).sum()
    n_test_neg = (labels_test == 0).sum()
    
    print(f"\nDataset split:")
    print(f"  Train: {n_train} samples ({n_train_pos} pos, {n_train_neg} neg)")
    print(f"  Test:  {n_test} samples ({n_test_pos} pos, {n_test_neg} neg)")
    print(f"  Hidden dim: {pos_hiddens.shape[1]}")
    
    class CCSProbe(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Multiple random restarts (avoid local minima)
    best_loss = float('inf')
    best_probe = None
    
    print(f"\n{'='*70}")
    print(f"TRAINING WITH MULTIPLE RANDOM RESTARTS")
    print(f"{'='*70}")
    
    for trial in range(config['ccs_ntries']):
        # Initialize fresh probe for this trial
        probe = CCSProbe(pos_hiddens.shape[1]).to(device)

        # Add weight decay (L2 regularization)
        optimizer = optim.AdamW(
            probe.parameters(),
            lr=config['ccs_lr'],
            weight_decay=config['ccs_weight_decay']
        )


        # ======================================================================
        # CHANGED (CCS alignment, minor): the examples are now randomly permuted
        # at the start of every training run, mirroring the original train().
        # With full-batch training this does not change the gradients, but it
        # keeps the procedure identical to the original's default setting.
        # ======================================================================
        permutation = torch.randperm(len(pos_train))
        pos_train_run = pos_train[permutation]
        neg_train_run = neg_train[permutation]


        # Training loop for this trial (full batch, as in the original's
        # default batch_size=-1 setting)
        probe.train()
        for epoch in range(config['ccs_epochs']):
            # Forward pass
            p_pos = probe(pos_train_run)
            p_neg = probe(neg_train_run)

            # NOTE: Original uses mean(0)
            consistency_loss = ((p_pos - (1 - p_neg)) ** 2).mean()
            # Confidence: predictions should be confident (far from 0.5)
            confidence_loss = (torch.min(p_pos, p_neg) ** 2).mean()

            loss = consistency_loss + confidence_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # ======================================================================
        # CHANGED (CCS alignment, minor): the loss used to select the best
        # restart is now the last training-step loss, exactly as returned by
        # the original train() and compared in repeated_train(); previously the
        # loss was recomputed over the full train set after training.
        # ======================================================================
        final_loss = loss.detach().cpu().item()


        print(f"  Trial {trial+1:2d}/{config['ccs_ntries']}: Loss = {final_loss:.6f}")

        # Keep best probe based on lowest loss (unsupervised criterion)
        if final_loss < best_loss:
            best_loss = final_loss
            best_probe = copy.deepcopy(probe)

    print(f"\n{'='*70}")
    print(f"EVALUATION WITH BEST PROBE")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.6f}")

    # Evaluate with best probe
    best_probe.eval()
    with torch.no_grad():
        p_pos = best_probe(pos_test).squeeze()
        p_neg = best_probe(neg_test).squeeze()

        # Average positive and negative predictions
        probs = 0.5 * (p_pos + (1 - p_neg))
        preds = (probs > 0.5).cpu().numpy()
    
    # Calculate metrics
    raw_accuracy = (preds == labels_test).mean()
    
    # Handle label ambiguity - determine if we need to flip labels
    if raw_accuracy < 0.5:
        # Probe learned inverted labels
        accuracy = 1 - raw_accuracy
        # Flip predictions to get correct count
        preds_corrected = 1 - preds
        correct = (preds_corrected == labels_test).sum()
    else:
        # Probe learned correct labels
        accuracy = raw_accuracy
        preds_corrected = preds
        correct = (preds == labels_test).sum()
    
    total = len(labels_test)
    
    # Class-wise accuracy (use corrected predictions)
    pos_mask = labels_test == 1
    neg_mask = labels_test == 0
    
    if pos_mask.sum() > 0:
        pos_acc = (preds_corrected[pos_mask] == labels_test[pos_mask]).mean()
    else:
        pos_acc = 0.0
    
    if neg_mask.sum() > 0:
        neg_acc = (preds_corrected[neg_mask] == labels_test[neg_mask]).mean()
    else:
        neg_acc = 0.0
    
    print(f"\nTest Results:")
    print(f"  Overall Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Positive samples: {pos_acc:.1%} ({pos_mask.sum()} samples)")
    print(f"  Negative samples: {neg_acc:.1%} ({neg_mask.sum()} samples)")
    
    return accuracy, best_probe


# ==============================================================================
# CHANGED (CCS alignment): new function. The original notebook runs a supervised
# logistic-regression check on the same hidden states BEFORE trying CCS
# ("if logistic regression accuracy is bad, there's no hope of CCS doing well"),
# using the difference between negative and positive hidden states as features.
# Previously this baseline lived only in supervised_vision.py and was never run
# as part of this pipeline. It uses the same seeded split as train_ccs_probe,
# and is diagnostic only — it does not affect CCS training or its accuracy.
# ==============================================================================
def lr_sanity_check(pos_hiddens, neg_hiddens, labels, config):
    """Supervised logistic-regression sanity check from the original CCS notebook."""
    n = len(labels)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=1 - config['train_split'],
        random_state=config['random_seed']
    )

    # as in the original: "for simplicity we can just take the difference
    # between positive and negative hidden states"
    x = neg_hiddens - pos_hiddens
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # C swept on a held-out slice of TRAIN, never the test set. At d=3584 with a
    # few hundred rows the sklearn default C=1.0 badly overfits.
    rng = np.random.default_rng(config['random_seed'])
    perm = rng.permutation(len(x_train))
    cut = int(round(0.8 * len(x_train)))
    fit_i, val_i = perm[:cut], perm[cut:]

    best = (-1.0, 1.0)
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(class_weight="balanced", max_iter=1000, C=C)
        m.fit(x_train[fit_i], y_train[fit_i])
        s = m.score(x_train[val_i], y_train[val_i])
        if s > best[0]:
            best = (s, C)
    val_acc, C = best

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=C)
    lr.fit(x_train, y_train)
    acc = lr.score(x_test, y_test)
    n_iter = int(np.max(lr.n_iter_))
    print(f"\nLogistic regression sanity-check accuracy: {acc:.1%} "
          f"(C={C}, held-out train {val_acc:.1%}, converged={n_iter < 1000})")
    return acc


def main():
    """Main VisionCCS pipeline."""
    model_key = f"model_{CONFIG['chosen_model']}"
    chosen_model_name = CONFIG[model_key]
    print(f"Model: {chosen_model_name}")
    
    all_results = {}
    
    for category in CONFIG['categories']:
        print(f"\n{'#'*70}")
        print(f"# CATEGORY: {category.upper()}")
        print(f"{'#'*70}")
        
        # 1. Load VQA data
        pairs = load_vqa_data(CONFIG, category)
        
        # 2. Extract hidden states
        pos_h, neg_h, labels = extract_in_batches(pairs, CONFIG, category)
        
        # Skip if no samples extracted
        if len(pos_h) == 0:
            print(f"\n✗ No samples extracted for '{category}'. Skipping...")
            all_results[category] = 0.0
            continue
        
        # ======================================================================
        # CHANGED (CCS alignment): run the original notebook's supervised
        # logistic-regression sanity check before CCS (see lr_sanity_check).
        # ======================================================================
        if CONFIG['run_lr_sanity_check']:
            lr_sanity_check(pos_h, neg_h, labels, CONFIG)


        # 3. Train CCS probe
        acc, probe = train_ccs_probe(pos_h, neg_h, labels, CONFIG)
        
        all_results[category] = acc
        print(f"\n✓ COMPLETE: {category} → {acc:.1%}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"\nFinal Results:")
    for category, acc in all_results.items():
        print(f"  {category:25s}: {acc:5.1%}")
    
    avg_acc = np.mean(list(all_results.values()))
    print(f"\n  {'Average':25s}: {avg_acc:5.1%}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()