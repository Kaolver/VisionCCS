import torch
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from qwen_vl_utils import process_vision_info
import os
from PIL import Image
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression


dataset = json.load(open('vqav2_mapped.json', 'r'))
distribution = {category: len(items) for category, items in dataset.items()}

CONFIG = {
    'n_samples_object_detection': distribution.get('object_detection', 0),
    'n_samples_attribute_recognition': distribution.get('attribute_recognition', 0),
    'n_samples_spatial_recognition': distribution.get('spatial_recognition', 0),
    'batch_size': 40,
    
    # Cache control
    'use_cache': False,  # Set to False to force re-extraction of hidden states
    
    # Paths
    'vqa_json': './vqav2_mapped.json',
    'image_dirs': [
        '/scratch-nvme/ml-datasets/coco/train/data',
        '/scratch-nvme/ml-datasets/coco/val/data',
    ],
    'cache_dir': './hidden_states_cache',
    'categories': ['object_detection', 'attribute_recognition', 'spatial_recognition'],
    
    'model_llava': 'llava-hf/llava-1.5-7b-hf',
    'model_qwen2': 'Qwen/Qwen2-VL-7B-Instruct',
    'model_qwen2_5': 'Qwen/Qwen2.5-VL-7B-Instruct',
    
    # Choose model: 'llava', 'qwen2', or 'qwen2_5'
    'chosen_model': 'qwen2',
    
    'hf_token': None,
    
    # Supervised training
    'train_split': 0.5,  # 50/50 train/test split
    'ccs_lr': 1e-3,
    'ccs_weight_decay': 0.01,
}


def load_vqa_data(config, category):
    """Load data for a specific category from the categorized VQA JSON."""
    data_path = Path(config['vqa_json'])
    with open(data_path, 'r') as f:
        all_data = json.load(f)
        vqa_data = all_data[category]
    
    # Get category-specific sample count
    n_samples_key = f'n_samples_{category}'
    n_samples = config.get(n_samples_key, len(vqa_data))
    
    # Take first N samples for this category
    samples = vqa_data[:n_samples]
    
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
    print(f"EXTRACTING HIDDEN STATES: {category.upper()}")
    
    cache_dir = Path(config['cache_dir'])
    cache_dir.mkdir(exist_ok=True)
    
    n = len(pairs)
    model_tag = config['chosen_model']
    cache_file = cache_dir / f"cache_{category}_{n}_supervised_contrast_{model_tag}.npz"
    
    if config['use_cache'] and cache_file.exists():
        data = np.load(cache_file)
        print(f"  Loaded: pos={data['pos_hiddens'].shape}, neg={data['neg_hiddens'].shape}, labels={data['labels'].shape}")
        return data['pos_hiddens'], data['neg_hiddens'], data['labels']
    
    if not config['use_cache']:
        print("⚠ Cache disabled (use_cache=False). Extracting new...")
    else:
        print("⚠ No cache found. Starting extraction...")

    print(f"\nProcessing {len(pairs)} samples in batches of {config['batch_size']}")
    print(f"Searching in {len(config['image_dirs'])} image directories")
    
    print(f"LOADING MODEL: {config['chosen_model']}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cpu":
        print("WARNING: Running on CPU!")

    chosen_model = config['chosen_model']
    hf_token = config.get('hf_token') or os.environ.get('HF_TOKEN')

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

    # Extract hidden states (positive and negative separately)
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
                    # Extract positive (Yes)
                    pos_hidden = extract_one_llava(
                        model, processor, image, pair['pos_text'], device
                    )
                    # Extract negative (No)
                    neg_hidden = extract_one_llava(
                        model, processor, image, pair['neg_text'], device
                    )
                
                elif config['chosen_model'] in ['qwen2', 'qwen2_5']:
                    # Extract positive (Yes)
                    pos_hidden = extract_one_qwen2(
                        model, processor, image, pair['pos_text'], device
                    )
                    # Extract negative (No)
                    neg_hidden = extract_one_qwen2(
                        model, processor, image, pair['neg_text'], device
                    )

                all_pos_hiddens.append(pos_hidden)
                all_neg_hiddens.append(neg_hidden)
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
    
    print(f"Successfully processed: {len(all_pos_hiddens)}/{len(pairs)}")
    print(f"Skipped (missing/error): {len(skipped)}/{len(pairs)}")
    
    if skipped:
        print(f"\nThere are skipped images: {', '.join(skipped[:10])}")
    
    # Convert to arrays
    pos_hiddens = np.array(all_pos_hiddens)
    neg_hiddens = np.array(all_neg_hiddens)
    labels = np.array(all_labels)
    
    print(f"\nExtracted shapes:")
    print(f"  Positive hidden states: {pos_hiddens.shape}")
    print(f"  Negative hidden states: {neg_hiddens.shape}")
    print(f"  Labels: {labels.shape}")
    
    # Save cache
    np.savez(cache_file, pos_hiddens=pos_hiddens, neg_hiddens=neg_hiddens, labels=labels)
    print(f"\nCached to: {cache_file}")
    
    return pos_hiddens, neg_hiddens, labels


def extract_one_llava(model, processor, image, text, device):
    """Extract hidden state from LLaVA for a single image-text pair."""
    
    prompt = f"USER: <image>\n{text}\nASSISTANT:"
    
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
    """
    Extract hidden state from Qwen2-VL (or Qwen2.5-VL) for a single image-text pair.
    
    Qwen models use AutoProcessor and a chat message format.
    """
    
    # Qwen chat message format
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
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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


def train_supervised_classifier(neg_hs, pos_hs, y, config):
    """
    Train supervised logistic regression classifier, following CCS.ipynb.
    
    From CCS.ipynb:
    - Split data 50/50 (or custom train_split)
    - Create features: x = neg_hs - pos_hs (simple difference, no normalization)
    - Train LogisticRegression with class_weight="balanced"
    - Evaluate on test set
    """
    print(f"\n{'='*70}")
    print(f"SUPERVISED LOGISTIC REGRESSION")
    print(f"{'='*70}")
    
    # 50/50 train split (the data is already randomized)
    n = len(y)
    n_train = int(n * config['train_split'])
    
    neg_hs_train, neg_hs_test = neg_hs[:n_train], neg_hs[n_train:]
    pos_hs_train, pos_hs_test = pos_hs[:n_train], pos_hs[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # for simplicity we can just take the difference between positive and negative hidden states
    x_train = neg_hs_train - pos_hs_train
    x_test = neg_hs_test - pos_hs_test
    
    n_train_pos = (y_train == 1).sum()
    n_train_neg = (y_train == 0).sum()
    n_test_pos = (y_test == 1).sum()
    n_test_neg = (y_test == 0).sum()
    
    print(f"\nDataset split:")
    print(f"  Train: {len(y_train)} samples ({n_train_pos} pos, {n_train_neg} neg)")
    print(f"  Test:  {len(y_test)} samples ({n_test_pos} pos, {n_test_neg} neg)")
    print(f"  Hidden dim: {x_train.shape[1]}")
    
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)
    
    test_acc = lr.score(x_test, y_test)
    print("\nLogistic regression accuracy: {:.1%}".format(test_acc))
    
    return test_acc, lr


def train_supervised_linear_probe(pos_train, neg_train, pos_test, neg_test, 
                                   labels_train, labels_test, config):
    """Supervised Linear Probe - Same architecture as CCS, supervised loss."""
    print(f"\n{'='*70}")
    print(f"SUPERVISED LINEAR PROBE")
    print(f"{'='*70}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pos_train = torch.FloatTensor(pos_train).to(device)
    neg_train = torch.FloatTensor(neg_train).to(device)
    pos_test = torch.FloatTensor(pos_test).to(device)
    neg_test = torch.FloatTensor(neg_test).to(device)
    labels_train_tensor = torch.FloatTensor(labels_train).to(device)
    
    # Normalize (same as CCS)
    pos_mean = pos_train.mean(dim=0)
    neg_mean = neg_train.mean(dim=0)
    pos_train = pos_train - pos_mean
    neg_train = neg_train - neg_mean
    pos_test = pos_test - pos_mean
    neg_test = neg_test - neg_mean
    
    # Same architecture as CCS
    probe = nn.Sequential(nn.Linear(pos_train.shape[1], 1), nn.Sigmoid()).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=config['ccs_lr'], weight_decay=config['ccs_weight_decay'])
    criterion = nn.BCELoss()
    
    # Training
    probe.train()
    for epoch in range(100):
        p_pos = probe(pos_train).squeeze()
        p_neg = probe(neg_train).squeeze()
        avg_pred = 0.5 * (p_pos + (1 - p_neg))
        loss = criterion(avg_pred, labels_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation
    probe.eval()
    with torch.no_grad():
        p_pos_train = probe(pos_train).squeeze()
        p_neg_train = probe(neg_train).squeeze()
        avg_pred_train = 0.5 * (p_pos_train + (1 - p_neg_train))
        train_preds = (avg_pred_train > 0.5).cpu().numpy()
        train_acc = (train_preds == labels_train).mean()
        
        p_pos_test = probe(pos_test).squeeze()
        p_neg_test = probe(neg_test).squeeze()
        avg_pred_test = 0.5 * (p_pos_test + (1 - p_neg_test))
        test_preds = (avg_pred_test > 0.5).cpu().numpy()
        test_acc = (test_preds == labels_test).mean()
        
        pos_mask = labels_test == 1
        neg_mask = labels_test == 0
        pos_acc = (test_preds[pos_mask] == labels_test[pos_mask]).mean() if pos_mask.sum() > 0 else 0.0
        neg_acc = (test_preds[neg_mask] == labels_test[neg_mask]).mean() if neg_mask.sum() > 0 else 0.0
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc:.1%}")
    print(f"  Test Accuracy:  {test_acc:.1%}")
    print(f"  Positive samples: {pos_acc:.1%} ({pos_mask.sum()} samples)")
    print(f"  Negative samples: {neg_acc:.1%} ({neg_mask.sum()} samples)")
    
    return test_acc


def main():
    model_key = f"model_{CONFIG['chosen_model']}"
    chosen_model_name = CONFIG.get(model_key, CONFIG['chosen_model'])
    print(f"{chosen_model_name} + Contrast Pairs + Supervised Methods")
    
    all_results = {}
    
    for category in CONFIG['categories']:
        print(f"\n{'#'*70}")
        print(f"# CATEGORY: {category.upper()}")
        print(f"{'#'*70}")
        
        # 1. Load VQA data
        pairs = load_vqa_data(CONFIG, category)
        
        # 2. Extract hidden states (both positive and negative)
        pos_h, neg_h, labels = extract_in_batches(pairs, CONFIG, category)
        
        # Skip if no samples extracted
        if len(pos_h) == 0:
            print(f"\n✗ No samples extracted for '{category}'. Skipping...")
            all_results[category] = {'logistic': 0.0, 'linear_probe': 0.0}
            continue
        
        # 3. Train supervised classifiers
        # 3a. Logistic Regression (sklearn baseline)
        logreg_acc, lr = train_supervised_classifier(neg_h, pos_h, labels, CONFIG)
        
        # 3b. Supervised Linear Probe (neural baseline)
        n = len(labels)
        n_train = int(n * CONFIG['train_split'])
        pos_train, pos_test = pos_h[:n_train], pos_h[n_train:]
        neg_train, neg_test = neg_h[:n_train], neg_h[n_train:]
        labels_train, labels_test = labels[:n_train], labels[n_train:]
        
        linear_probe_acc = train_supervised_linear_probe(
            pos_train, neg_train, pos_test, neg_test,
            labels_train, labels_test, CONFIG
        )
        
        all_results[category] = {
            'logistic': logreg_acc,
            'linear_probe': linear_probe_acc
        }
        
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"  Logistic Regression:     {logreg_acc:.1%}")
        print(f"  Supervised Linear Probe: {linear_probe_acc:.1%}")
        print(f"\n✓ COMPLETE: {category}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Category':<30} {'LogReg':<12} {'Linear Probe':<12}")
    print(f"{'-'*70}")
    
    logreg_accs = []
    linear_probe_accs = []
    
    for category, results in all_results.items():
        logreg_acc = results['logistic']
        linear_acc = results['linear_probe']
        
        logreg_accs.append(logreg_acc)
        linear_probe_accs.append(linear_acc)
        
        print(f"  {category:<28} {logreg_acc:>10.1%}  {linear_acc:>10.1%}")
    
    avg_logreg = np.mean(logreg_accs)
    avg_linear = np.mean(linear_probe_accs)
    
    print(f"{'-'*70}")
    print(f"  {'Average':<28} {avg_logreg:>10.1%}  {avg_linear:>10.1%}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()