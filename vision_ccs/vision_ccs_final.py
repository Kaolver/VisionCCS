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
    
    'train_split': 0.7,
    'ccs_epochs': 1000,
    'ccs_ntries': 10,
    'ccs_lr': 1e-3,
    'ccs_weight_decay': 0.01,
}


def load_vqa_data(config, category):
    """Load data for a specific category from the categorized VQA JSON."""
    data_path = Path(config['vqa_json'])
    with open(data_path, 'r') as f:
        all_data = json.load(f)
        vqa_data = all_data[category]
    
    n_samples_key = f'n_samples_{category}'
    n_samples = config.get(n_samples_key, len(vqa_data))
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
    print(f"\n{'='*70}")
    print(f"EXTRACTING HIDDEN STATES: {category.upper()}")
    print(f"{'='*70}")
    
    cache_dir = Path(config['cache_dir'])
    cache_dir.mkdir(exist_ok=True)
    
    # Check cache
    n = len(pairs)
    model_tag = config['chosen_model']
    cache_file = cache_dir / f"cache_{category}_{n}_{model_tag}.npz"
    
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
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    
    # Stratified train/test split to maintain class balance    
    n = len(labels)
    indices = np.arange(n)
    
    # Stratified split maintains class proportions
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=1 - config['train_split'],
        stratify=labels,
        random_state=42
    )
    
    pos_train_raw = pos_hiddens[train_idx]
    neg_train_raw = neg_hiddens[train_idx]
    pos_test_raw = pos_hiddens[test_idx]
    neg_test_raw = neg_hiddens[test_idx]
    labels_test = labels[test_idx]

    pos_mean_train = pos_train_raw.mean(dim=0)
    neg_mean_train = neg_train_raw.mean(dim=0)
    pos_train = pos_train_raw - pos_mean_train
    neg_train = neg_train_raw - neg_mean_train
    
    pos_mean_test = pos_test_raw.mean(dim=0)
    neg_mean_test = neg_test_raw.mean(dim=0)
    pos_test = pos_test_raw - pos_mean_test
    neg_test = neg_test_raw - neg_mean_test
    
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
        probe = CCSProbe(pos_hiddens.shape[1])
        
        # Add weight decay (L2 regularization)
        optimizer = optim.AdamW(
            probe.parameters(), 
            lr=config['ccs_lr'],
            weight_decay=config['ccs_weight_decay']
        )
        
        # Training loop for this trial
        probe.train()
        for epoch in range(config['ccs_epochs']):
            # Forward pass
            p_pos = probe(pos_train)
            p_neg = probe(neg_train)
            
            consistency_loss = ((p_pos - (1 - p_neg)) ** 2).mean()
            # Confidence: predictions should be confident (far from 0.5)
            confidence_loss = (torch.min(p_pos, p_neg) ** 2).mean()
            
            loss = consistency_loss + confidence_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate final loss for this trial
        probe.eval()
        with torch.no_grad():
            p_pos_final = probe(pos_train)
            p_neg_final = probe(neg_train)
            final_consistency = ((p_pos_final - (1 - p_neg_final)) ** 2).mean()
            final_confidence = (torch.min(p_pos_final, p_neg_final) ** 2).mean()
            final_loss = final_consistency + final_confidence
        
        print(f"  Trial {trial+1:2d}/{config['ccs_ntries']}: Loss = {final_loss.item():.6f}")
        
        # Keep best probe based on lowest loss (unsupervised criterion)
        if final_loss < best_loss:
            best_loss = final_loss
            best_probe = copy.deepcopy(probe)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION WITH BEST PROBE")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss.item():.6f}")
    
    # Evaluate with best probe
    best_probe.eval()
    with torch.no_grad():
        p_pos = best_probe(pos_test).squeeze()
        p_neg = best_probe(neg_test).squeeze()
        
        # Average positive and negative predictions
        probs = 0.5 * (p_pos + (1 - p_neg))
        preds = (probs > 0.5).numpy()
    
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