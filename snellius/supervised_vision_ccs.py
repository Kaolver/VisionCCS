import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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
    
    # Model
    'model_name': 'llava-hf/llava-1.5-7b-hf',
    
    # Supervised training
    'train_split': 0.7,  # 70% train, 30% test
    'logistic_C': 1.0,  # Inverse regularization strength (higher = less regularization)
    'logistic_max_iter': 1000,  # Maximum iterations for solver
    'logistic_solver': 'lbfgs',  # Solver: 'lbfgs', 'liblinear', 'saga'
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
    
    # Create training samples (we only need one representation per question)
    pairs = []
    for item in samples:
        q = item['question'].rstrip('?')
        img_id = item['image_id']
        if isinstance(img_id, int):
            img_id = f"{img_id:012d}.jpg"
        pairs.append({
            'image_id': img_id,
            'question': q,
            'text': q,  # Just the question, no "Yes/No" appended
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
    cache_file = cache_dir / f"cache_{category}_{n}_supervised_llava.npz"
    
    if config['use_cache'] and cache_file.exists():
        print("✓ Found cached hidden states!")
        print(f"  Loading from: {cache_file}")
        data = np.load(cache_file)
        print(f"  Loaded: hiddens={data['hiddens'].shape}, labels={data['labels'].shape}")
        return data['hiddens'], data['labels']
    
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
    all_hiddens = []
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
                
                # Extract hidden state
                hidden = extract_one_llava(
                    model, processor, image, pair['text'], device
                )
                
                all_hiddens.append(hidden)
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
    print(f"✓ Successfully processed: {len(all_hiddens)}/{len(pairs)}")
    print(f"✗ Skipped (missing/error): {len(skipped)}/{len(pairs)}")
    
    if skipped and len(skipped) <= 10:
        print(f"\nSkipped images: {', '.join(skipped[:10])}")
    elif skipped:
        print(f"\nFirst 10 skipped: {', '.join(skipped[:10])}...")
    
    # Convert to arrays
    hiddens = np.array(all_hiddens)
    labels = np.array(all_labels)
    
    print(f"\nExtracted shapes:")
    print(f"  Hidden states: {hiddens.shape}")
    print(f"  Labels: {labels.shape}")
    
    # Save cache
    np.savez(cache_file, hiddens=hiddens, labels=labels)
    print(f"\nCached to: {cache_file}")
    
    return hiddens, labels


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


def train_supervised_classifier(hiddens, labels, config):
    """Train supervised logistic regression classifier following CCS methodology."""
    print(f"\n{'='*70}")
    print(f"TRAINING SUPERVISED LOGISTIC REGRESSION")
    print(f"{'='*70}")

    # Convert to numpy arrays
    if isinstance(hiddens, torch.Tensor):
        hiddens = hiddens.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # Center the features to zero mean
    # Normalize hidden states
    hiddens_mean = hiddens.mean(axis=0)
    hiddens_normalized = hiddens - hiddens_mean
    
    # Stratified train/test split to maintain class balance
    train_hiddens, test_hiddens, train_labels, test_labels = train_test_split(
        hiddens_normalized,
        labels,
        test_size=1 - config['train_split'],
        stratify=labels,
        random_state=42
    )
    
    n_train = len(train_labels)
    n_test = len(test_labels)
    n_train_pos = (train_labels == 1).sum()
    n_train_neg = (train_labels == 0).sum()
    n_test_pos = (test_labels == 1).sum()
    n_test_neg = (test_labels == 0).sum()
    
    print(f"\nDataset split (Stratified):")
    print(f"  Train: {n_train} samples ({n_train_pos} pos, {n_train_neg} neg)")
    print(f"  Test:  {n_test} samples ({n_test_pos} pos, {n_test_neg} neg)")
    print(f"  Hidden dim: {hiddens.shape[1]}")
    
    print(f"\nLogistic Regression config:")
    print(f"  C (inverse regularization): {config['logistic_C']}")
    print(f"  Solver: {config['logistic_solver']}")
    print(f"  Max iterations: {config['logistic_max_iter']}")
    
    # Train logistic regression
    print(f"\nTraining...")
    clf = LogisticRegression(
        C=config['logistic_C'],
        max_iter=config['logistic_max_iter'],
        solver=config['logistic_solver'],
        random_state=42,
        verbose=1
    )
    
    clf.fit(train_hiddens, train_labels)
    print(f"✓ Training complete!")
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print(f"EVALUATION")
    print(f"{'='*70}")
    
    train_preds = clf.predict(train_hiddens)
    test_preds = clf.predict(test_hiddens)
    
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"\nTraining Accuracy: {train_acc:.1%}")
    print(f"Test Accuracy: {test_acc:.1%}")
    
    # Class-wise accuracy
    pos_mask = test_labels == 1
    neg_mask = test_labels == 0
    
    if pos_mask.sum() > 0:
        pos_acc = (test_preds[pos_mask] == test_labels[pos_mask]).mean()
    else:
        pos_acc = 0.0
    
    if neg_mask.sum() > 0:
        neg_acc = (test_preds[neg_mask] == test_labels[neg_mask]).mean()
    else:
        neg_acc = 0.0
    
    print(f"\nTest Results (Class-wise):")
    print(f"  Positive samples: {pos_acc:.1%} ({pos_mask.sum()} samples)")
    print(f"  Negative samples: {neg_acc:.1%} ({neg_mask.sum()} samples)")
    
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['No', 'Yes']))
    
    return test_acc, clf


def main():
    """Main supervised learning pipeline."""
    print("\n" + "="*70)
    print(" " * 15 + "SUPERVISED VISION CLASSIFICATION")
    print(" " * 10 + "LLaVA + Logistic Regression")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Model: {CONFIG['model_name']}")
    print(f"  Samples per category:")
    print(f"    - object_detection: {CONFIG['n_samples_object_detection']}")
    print(f"    - attribute_recognition: {CONFIG['n_samples_attribute_recognition']}")
    print(f"    - spatial_recognition: {CONFIG['n_samples_spatial_recognition']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Cache enabled: {CONFIG['use_cache']}")
    print(f"  Categories: {', '.join(CONFIG['categories'])}")
    print(f"\nLogistic Regression:")
    print(f"  C (inverse regularization): {CONFIG['logistic_C']}")
    print(f"  Solver: {CONFIG['logistic_solver']}")
    print(f"  Max iterations: {CONFIG['logistic_max_iter']}")
    
    all_results = {}
    
    for category in CONFIG['categories']:
        print(f"\n{'#'*70}")
        print(f"# CATEGORY: {category.upper()}")
        print(f"{'#'*70}")
        
        # 1. Load VQA data
        pairs = load_vqa_data(CONFIG, category)
        
        # 2. Extract hidden states
        hiddens, labels = extract_in_batches(pairs, CONFIG, category)
        
        # Skip if no samples extracted
        if len(hiddens) == 0:
            print(f"\n✗ No samples extracted for '{category}'. Skipping...")
            all_results[category] = 0.0
            continue
        
        # 3. Train supervised classifier
        acc, clf = train_supervised_classifier(hiddens, labels, CONFIG)
        
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
