import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc


CONFIG = {
    'n_samples': 1000,  # Samples per category
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
    'model_name': 'llava-hf/llava-1.5-7b-hf',  # LLaVA 1.5 with Vicuna-7B
    
    # CCS training (following original paper methodology)
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
    
    # Take first N samples for this category
    samples = vqa_data[:config['n_samples']]
    print(f"Using {len(samples)} samples from '{category}'")
    
    # Create contrast pairs
    pairs = []
    for item in samples:
        q = item['question'].rstrip('?')
        # Ensure image_id is a string (handle both int and str formats)
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
    cache_file = cache_dir / f"cache_{category}_{n}_llava.npz"
    
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
    
    # Load LLaVA model
    print(f"\n{'='*70}")
    print(f"LOADING LLAVA MODEL: {config['model_name']}")
    print(f"{'='*70}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cpu":
        print("⚠ WARNING: Running on CPU will be VERY slow!")
    
    # Load with legacy tokenizer to avoid enum ModelWrapper error
    processor = LlavaProcessor.from_pretrained(
        config['model_name'],
        use_fast=False  # Use slow tokenizer to avoid version conflicts
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
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
                
                # Extract positive (Yes)
                pos_h = extract_one_llava(
                    model, processor, image, pair['pos_text'], device
                )
                
                # Extract negative (No)
                neg_h = extract_one_llava(
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
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
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
    print(f"\n💾 Cached to: {cache_file}")
    
    return pos_hiddens, neg_hiddens, labels


def extract_one_llava(model, processor, image, text, device):
    """Extract hidden state from LLaVA for a single image-text pair."""
    
    # Simple prompt format for LLaVA 1.5
    prompt = f"USER: <image>\n{text}\nASSISTANT:"
    
    # Process inputs
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Extract hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Use last hidden state from language model, averaged across tokens
        # outputs.hidden_states[-1] has shape [batch_size, seq_len, hidden_dim]
        hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
    
    return hidden.cpu().float().numpy()


def train_ccs_probe(pos_hiddens, neg_hiddens, labels, config):
    """Train CCS probe with consistency loss following original methodology."""
    print(f"\n{'='*70}")
    print(f"TRAINING CCS PROBE")
    print(f"{'='*70}")
    
    import torch.nn as nn
    import torch.optim as optim
    import copy
    
    # Convert to tensors
    pos_hiddens = torch.FloatTensor(pos_hiddens)
    neg_hiddens = torch.FloatTensor(neg_hiddens)
    
    # CRITICAL FIX #1: Normalize independently (remove cluster bias)
    # This ensures the probe learns truth, not just "Yes" vs "No" tokens
    pos_hiddens = pos_hiddens - pos_hiddens.mean(dim=0)
    neg_hiddens = neg_hiddens - neg_hiddens.mean(dim=0)
    
    # Train/test split
    n = len(labels)
    n_train = int(config['train_split'] * n)
    
    pos_train = pos_hiddens[:n_train]
    neg_train = neg_hiddens[:n_train]
    pos_test = pos_hiddens[n_train:]
    neg_test = neg_hiddens[n_train:]
    labels_test = labels[n_train:]
    
    print(f"\nDataset split:")
    print(f"  Train: {n_train} samples")
    print(f"  Test:  {n - n_train} samples")
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
    accuracy = (preds == labels_test).mean()
    
    # CRITICAL FIX #2: Handle label ambiguity
    # CCS doesn't know if it learned "truth=1" or "truth=0"
    # Take max to handle both cases
    accuracy = max(accuracy, 1 - accuracy)
    
    correct = (preds == labels_test).sum()
    total = len(labels_test)
    
    # Class-wise accuracy
    pos_mask = labels_test == 1
    neg_mask = labels_test == 0
    
    if pos_mask.sum() > 0:
        pos_acc = (preds[pos_mask] == labels_test[pos_mask]).mean()
        pos_acc = max(pos_acc, 1 - pos_acc)  # Also correct for ambiguity
    else:
        pos_acc = 0.0
    
    if neg_mask.sum() > 0:
        neg_acc = (preds[neg_mask] == labels_test[neg_mask]).mean()
        neg_acc = max(neg_acc, 1 - neg_acc)  # Also correct for ambiguity
    else:
        neg_acc = 0.0
    
    print(f"\nTest Results:")
    print(f"  Overall Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Positive samples: {pos_acc:.1%} ({pos_mask.sum()} samples)")
    print(f"  Negative samples: {neg_acc:.1%} ({neg_mask.sum()} samples)")
    
    return accuracy, best_probe


def main():
    """Main VisionCCS pipeline."""
    print("\n" + "="*70)
    print(" " * 15 + "VISIONCCS PIPELINE")
    print(" " * 10 + "LLaVA + CCS (Original Methodology)")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Model: {CONFIG['model_name']}")
    print(f"  Samples per category: {CONFIG['n_samples']}")
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
    print(f"ALL EXPERIMENTS FINISHED")
    print(f"{'='*70}")
    print(f"\nFinal Results:")
    for category, acc in all_results.items():
        print(f"  {category:25s}: {acc:5.1%}")
    
    avg_acc = np.mean(list(all_results.values()))
    print(f"\n  {'Average':25s}: {avg_acc:5.1%}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
