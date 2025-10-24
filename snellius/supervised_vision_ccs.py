import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
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
    
    # Model
    'model_name': 'llava-hf/llava-1.5-7b-hf',
    
    # Supervised training (following CCS.ipynb)
    'train_split': 0.5,  # 50/50 train/test split (like CCS.ipynb)
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
    
    # Check cache
    n = len(pairs)
    cache_file = cache_dir / f"cache_{category}_{n}_supervised_contrast_llava.npz"
    
    if config['use_cache'] and cache_file.exists():
        data = np.load(cache_file)
        print(f"  Loaded: pos={data['pos_hiddens'].shape}, neg={data['neg_hiddens'].shape}, labels={data['labels'].shape}")
        return data['pos_hiddens'], data['neg_hiddens'], data['labels']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
                
                # Extract positive hidden state (Yes)
                pos_hidden = extract_one_llava(
                    model, processor, image, pair['pos_text'], device
                )
                
                # Extract negative hidden state (No)
                neg_hidden = extract_one_llava(
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


def train_supervised_classifier(neg_hs, pos_hs, y, config):
    """
    Train supervised logistic regression classifier, exactly following CCS.ipynb.
    
    From CCS.ipynb:
    - Split data 50/50 (or custom train_split)
    - Create features: x = neg_hs - pos_hs (simple difference, no normalization)
    - Train LogisticRegression with class_weight="balanced"
    - Evaluate on test set
    """
    print(f"TRAINING SUPERVISED LOGISTIC REGRESSION")
    
    # let's create a simple 50/50 train split (the data is already randomized)
    n = len(y)
    n_train = int(n * config['train_split'])
    
    neg_hs_train, neg_hs_test = neg_hs[:n_train], neg_hs[n_train:]
    pos_hs_train, pos_hs_test = pos_hs[:n_train], pos_hs[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # for simplicity we can just take the difference between positive and negative hidden states
    # (concatenating also works fine)
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
    
    print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))
    
    return lr.score(x_test, y_test), lr


def main():
    print("LLaVA + Contrast Pairs + Logistic Regression")
    
    all_results = {}
    
    for category in CONFIG['categories']:
        print(f"\n{'#'*70}")
        print(f"# CATEGORY: {category.upper()}")
        
        # 1. Load VQA data
        pairs = load_vqa_data(CONFIG, category)
        
        # 2. Extract hidden states (both positive and negative)
        pos_h, neg_h, labels = extract_in_batches(pairs, CONFIG, category)
        
        # Skip if no samples extracted
        if len(pos_h) == 0:
            print(f"\n✗ No samples extracted for '{category}'. Skipping...")
            all_results[category] = 0.0
            continue
        
        # 3. Train supervised classifier (following CCS.ipynb exactly)
        acc, lr = train_supervised_classifier(neg_h, pos_h, labels, CONFIG)
        
        all_results[category] = acc
        print(f"\n✓ COMPLETE: {category} → {acc:.1%}")
    
    # Final summary
    print(f"\nFinal Results:")
    for category, acc in all_results.items():
        print(f"  {category:25s}: {acc:5.1%}")
    
    avg_acc = np.mean(list(all_results.values()))
    print(f"\n  {'Average':25s}: {avg_acc:5.1%}")


if __name__ == "__main__":
    main()