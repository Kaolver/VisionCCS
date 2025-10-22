import torch
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)
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
    'use_cache': False,  # Set to False to force re-extraction of hidden states
    
    # Paths
    'vqa_json': './vqav2_mapped.json',
    'image_dirs': [
        '/scratch-nvme/ml-datasets/coco/train/data',
        '/scratch-nvme/ml-datasets/coco/val/data',  # Fallback for validation images
    ],
    'cache_dir': './hidden_states_cache',
    'categories': ['object_detection', 'attribute_recognition', 'spatial_recognition'],
    
    # Model
    'model_llava': 'llava-hf/llava-1.5-7b-hf',
    'model_qwen': 'Qwen/Qwen-VL-Chat',
    'model_qwen2': 'Qwen/Qwen2-VL-7B-Instruct',

    # Choose model: 'llava' or 'qwen'
    'chosen_model': 'qwen2',
    # Optional: Hugging Face token for private/gated repos. If None, will use HF cache/login.
    'hf_token': None,
    
    'train_split': 0.7,  # 70% train, 30% test
    'ccs_epochs': 1000,  # Original uses 1000 epochs
    'ccs_ntries': 10,  # Multiple random restarts to avoid local minima
    'ccs_lr': 1e-3,
    'ccs_weight_decay': 0.01,  # L2 regularization
    'probe_hidden_dim': 256,  # Hidden layer size for probe
}


def load_vqa_data(config, category):
    """Load data for a specific category from the categorized VQA JSON."""
    print(f"LOADING DATA for category: '{category}'")
    
    data_path = Path(config['vqa_json'])
    with open(data_path, 'r') as f:
        all_data = json.load(f)
        vqa_data = all_data[category]
    
    # Get category-specific sample count
    n_samples_key = f'n_samples_{category}'
    n_samples = config.get(n_samples_key, len(vqa_data))
    
    # Take first N samples for this category
    samples = vqa_data[:n_samples]
    print(f"Using {len(samples)} samples from '{category}'")
    
    # Create contrast pairs
    pairs = []
    for item in samples:
        q = item['question'].rstrip('?')
        img_id = item['image_id']
        if isinstance(img_id, int):
            img_id = f"{img_id:012d}.jpg"  # Convert int to zero-padded filename
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
            use_fast=False  # Use slow tokenizer to avoid version conflicts
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            config['model_llava'],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)
    else: # Qwen
        # Qwen-VL uses custom processing classes, not standard AutoProcessor
        # Load tokenizer and model separately with trust_remote_code
        hf_token = config.get('hf_token') or os.environ.get('HF_TOKEN')
        
        tokenizer = AutoTokenizer.from_pretrained(
            config['model_qwen'],
            trust_remote_code=True,
            use_auth_token=hf_token,
        )
        
        # Load Qwen-VL model without device_map to avoid 'meta' device issues
        # The model will auto-convert to bf16 on CUDA
        model = AutoModelForCausalLM.from_pretrained(
            config['model_qwen'],
            trust_remote_code=True,
            use_auth_token=hf_token,
        ).eval()
        
        # Manually move to device after loading
        if device == "cuda":
            model = model.cuda()
        else:
            model = model.to(device)
        
        # For Qwen-VL, we use the tokenizer directly (no separate processor)
        processor = tokenizer

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
                    # Extract positive (Yes)
                    pos_h = extract_one_llava(
                        model, processor, image, pair['pos_text'], device
                    )
                    # Extract negative (No)
                    neg_h = extract_one_llava(
                        model, processor, image, pair['neg_text'], device
                    )
                else: # 'qwen'
                    # Extract positive (Yes)
                    pos_h = extract_one_qwen(
                        model, processor, image, pair['pos_text'], device
                    )
                    # Extract negative (No)
                    neg_h = extract_one_qwen(
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
        # outputs.hidden_states[-1] has shape [batch_size, seq_len, hidden_dim]
        # Take the last token position: [:, -1, :]
        hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0)
    
    return hidden.cpu().float().numpy()


def extract_one_qwen(model, tokenizer, image, text, device):
    """Extract hidden state from Qwen-VL for a single image-text pair.
    
    Qwen-VL uses a special format where images are embedded via tokenizer.from_list_format.
    """
    
    # Save image temporarily (Qwen-VL expects image path or PIL)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        image.save(tmp.name)
        image_path = tmp.name
    
    try:
        # Qwen-VL query format: list of dicts with 'image' and 'text' keys
        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': text},
        ])
        
        # Tokenize the query
        inputs = tokenizer(query, return_tensors='pt')
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Use LAST TOKEN hidden state (aligns with CCS paper)
        hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0)
        
        return hidden.cpu().float().numpy()
    
    finally:
        # Clean up temp file
        if os.path.exists(image_path):
            os.unlink(image_path)


def train_ccs_probe(pos_hiddens, neg_hiddens, labels, config):
    """Train CCS probe with consistency loss following original methodology."""
    print(f"\n{'='*70}")
    print(f"TRAINING CCS PROBE")
    print(f"{'='*70}")

    # Convert to tensors
    pos_hiddens = torch.FloatTensor(pos_hiddens)
    neg_hiddens = torch.FloatTensor(neg_hiddens)
    
    # CRITICAL FIX #1: Normalize independently (remove cluster bias)
    # This ensures the probe learns truth, not just "Yes" vs "No" tokens
    pos_hiddens = pos_hiddens - pos_hiddens.mean(dim=0)
    neg_hiddens = neg_hiddens - neg_hiddens.mean(dim=0)
    
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
    
    pos_train = pos_hiddens[train_idx]
    neg_train = neg_hiddens[train_idx]
    pos_test = pos_hiddens[test_idx]
    neg_test = neg_hiddens[test_idx]
    labels_test = labels[test_idx]
    
    n_train = len(train_idx)
    n_test = len(test_idx)
    n_train_pos = (labels[train_idx] == 1).sum()
    n_train_neg = (labels[train_idx] == 0).sum()
    n_test_pos = (labels_test == 1).sum()
    n_test_neg = (labels_test == 0).sum()
    
    print(f"\nDataset split (Stratified):")
    print(f"  Train: {n_train} samples ({n_train_pos} pos, {n_train_neg} neg)")
    print(f"  Test:  {n_test} samples ({n_test_pos} pos, {n_test_neg} neg)")
    print(f"  Hidden dim: {pos_hiddens.shape[1]}")
    
    # Define probe network
    class CCSProbe(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    print(f"\nProbe architecture:")
    print(f"  Input: {pos_hiddens.shape[1]}")
    print(f"  Hidden: {config['probe_hidden_dim']} → {config['probe_hidden_dim']//2}")
    print(f"  Output: 1 (probability)")
    print(f"\nTraining config:")
    print(f"  Epochs per trial: {config['ccs_epochs']}")
    print(f"  Number of trials: {config['ccs_ntries']}")
    print(f"  Learning rate: {config['ccs_lr']}")
    print(f"  Weight decay: {config['ccs_weight_decay']}")
    
    # CRITICAL FIX #3: Multiple random restarts (avoid local minima)
    best_loss = float('inf')
    best_probe = None
    
    print(f"\n{'='*70}")
    print(f"TRAINING WITH MULTIPLE RANDOM RESTARTS")
    print(f"{'='*70}")
    
    for trial in range(config['ccs_ntries']):
        # Initialize fresh probe for this trial
        probe = CCSProbe(pos_hiddens.shape[1], config['probe_hidden_dim'])
        
        # CRITICAL FIX #4: Add weight decay (L2 regularization)
        optimizer = optim.Adam(
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
            
            # CCS loss: consistency + confidence
            # Consistency: p(pos) should equal 1 - p(neg)
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
            print(f"    ✓ New best probe found!")
    
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
    
    # CRITICAL FIX #2: Handle label ambiguity
    # CCS doesn't know if it learned "truth=1" or "truth=0"
    # Determine if we need to flip labels
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
    chosen_model = CONFIG[f"model_{CONFIG['chosen_model']}"]
    print(f"\nConfiguration:")
    print(f"  Model: {chosen_model}")
    print(f"  Samples per category:")
    print(f"    - object_detection: {CONFIG['n_samples_object_detection']}")
    print(f"    - attribute_recognition: {CONFIG['n_samples_attribute_recognition']}")
    print(f"    - spatial_recognition: {CONFIG['n_samples_spatial_recognition']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Cache enabled: {CONFIG['use_cache']}")
    print(f"  Categories: {', '.join(CONFIG['categories'])}")
    print(f"\nCCS Training:")
    print(f"  Epochs per trial: {CONFIG['ccs_epochs']}")
    print(f"  Random restarts: {CONFIG['ccs_ntries']}")
    print(f"  Learning rate: {CONFIG['ccs_lr']}")
    print(f"  Weight decay: {CONFIG['ccs_weight_decay']}")
    
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
