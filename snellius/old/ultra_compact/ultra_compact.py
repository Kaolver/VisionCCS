import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

# ============================================================================
# ULTRA LIGHTWEIGHT CONFIG otherwise local memory overload
# ============================================================================

CONFIG = {
    'n_samples': 200, 
    'batch_size': 10,
    
    # Cache control
    'use_cache': True,  # Set to False to force re-extraction of hidden states
    
    # Paths
    'vqa_json': './vqav2_filtered_categorized_checkpoint_7200_fixed.json',
    'image_dir': '/scratch-nvme/ml-datasets/coco/train/data',
    'cache_dir': './hidden_states_cache',
    'categories': ['object_detection', 'attribute_recognition', 'spatial_recognition'],
    
    # Model
    'model_name': 'nlpconnect/vit-gpt2-image-captioning',
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
        pairs.append({
            'image_id': item['image_id'],  # Use the image_id field from fixed JSON
            'pos_text': f"{q}? Yes",
            'neg_text': f"{q}? No",
            'label': 1 if item['answer'] == 'yes' else 0
        })
    
    return pairs


def extract_in_batches(pairs, config, category):
    """Extract hidden states in small batches to avoid OOM"""
    print(f"\nEXTRACTING HIDDEN STATES for '{category}' (BATCH MODE)")
    
    cache_dir = Path(config['cache_dir'])
    cache_dir.mkdir(exist_ok=True)
    
    # Check if already cached (category-specific cache)
    n = len(pairs)
    cache_file = cache_dir / f"cache_{category}_{n}_vit.npz"
    
    if config['use_cache'] and cache_file.exists():
        print("\nFound cached hidden states!")
        print("Loading from cache...")
        data = np.load(cache_file)
        return data['pos_hiddens'], data['neg_hiddens'], data['labels']
    
    if not config['use_cache']:
        print("\nCache disabled (use_cache=False). Extracting fresh hidden states...")
    else:
        print("\nNo cache found. Extracting...")
    
    print(f"Processing {len(pairs)} samples in batches of {config['batch_size']}")
    
    # Load lightweight model
    print("\nLoading lightweight model (ViT-GPT2, ~500MB)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = VisionEncoderDecoderModel.from_pretrained(config['model_name'])
    feature_extractor = ViTImageProcessor.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    model = model.to(device)
    model.eval()
    
    print("--- Model loaded ---\n")
    
    # Extract in batches
    all_pos_hiddens = []
    all_neg_hiddens = []
    all_labels = []
    
    image_dir = Path(config['image_dir'])
    
    for i in tqdm(range(0, len(pairs), config['batch_size']), desc="Batches"):
        batch_pairs = pairs[i:i + config['batch_size']]
        
        for pair in batch_pairs:
            image_id = pair['image_id']  # Already includes extension like "000000525380.jpg"
            image_path = image_dir / image_id
            
            if not image_path.exists():
                print(f"Skipping {image_id}")
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                
                # Extract positive
                pos_h = extract_one(model, feature_extractor, tokenizer, image, pair['pos_text'], device)
                
                # Extract negative
                neg_h = extract_one(model, feature_extractor, tokenizer, image, pair['neg_text'], device)
                
                all_pos_hiddens.append(pos_h)
                all_neg_hiddens.append(neg_h)
                all_labels.append(pair['label'])
                
            except Exception as e:
                print(f"Error {image_id}: {e}")
                continue
        
        # Clear cache after each batch
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # Unload model to free memory
    del model
    del feature_extractor
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    print("\nModel unloaded to free memory")
    
    # Convert to arrays
    pos_hiddens = np.array(all_pos_hiddens)
    neg_hiddens = np.array(all_neg_hiddens)
    labels = np.array(all_labels)
    
    print(f"\nExtracted: {pos_hiddens.shape}")
    
    # Cache for future runs
    np.savez(cache_file, pos_hiddens=pos_hiddens, neg_hiddens=neg_hiddens, labels=labels)
    print(f"💾 Cached to: {cache_file}")
    
    return pos_hiddens, neg_hiddens, labels


def extract_one(model, feature_extractor, tokenizer, image, text, device):
    """Extract single hidden state from ViT-GPT2 model"""
    # Process image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Tokenize text
    text_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        # Get encoder outputs (vision)
        encoder_outputs = model.encoder(pixel_values, output_hidden_states=True)
        
        # Get decoder outputs (text, conditioned on image)
        decoder_outputs = model.decoder(
            input_ids=text_inputs['input_ids'],
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            output_hidden_states=True
        )
        
        # Use last decoder hidden state, averaged across tokens
        hidden = decoder_outputs.hidden_states[-1].mean(dim=1).squeeze(0)
    
    return hidden.cpu().numpy()


def train_simple_ccs(pos_hiddens, neg_hiddens, labels):
    """Train CCS probe (simplified)"""
    print("TRAINING CCS PROBE")
    
    import torch.nn as nn
    import torch.optim as optim
    
    # Normalize
    pos_hiddens = torch.FloatTensor(pos_hiddens)
    neg_hiddens = torch.FloatTensor(neg_hiddens)
    
    pos_hiddens = pos_hiddens - pos_hiddens.mean(dim=0)
    neg_hiddens = neg_hiddens - neg_hiddens.mean(dim=0)
    
    # Simple 70/30 split
    n = len(labels)
    n_train = int(0.7 * n)
    
    pos_train = pos_hiddens[:n_train]
    neg_train = neg_hiddens[:n_train]
    pos_test = pos_hiddens[n_train:]
    neg_test = neg_hiddens[n_train:]
    labels_test = labels[n_train:]
    
    print(f"Train: {n_train}, Test: {n - n_train}")
    
    # Simple probe
    class Probe(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 100),
                nn.ReLU(),
                nn.Linear(100, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)
    
    probe = Probe(pos_hiddens.shape[1])
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    
    print("\nTraining...")
    for epoch in range(100):
        p_pos = probe(pos_train)
        p_neg = probe(neg_train)
        
        loss = ((p_pos - (1 - p_neg))**2).mean() + (torch.min(p_pos, p_neg)**2).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.6f}")
    
    # Test
    probe.eval()
    with torch.no_grad():
        p_pos = probe(pos_test).squeeze()
        p_neg = probe(neg_test).squeeze()
        probs = 0.5 * (p_pos + (1 - p_neg))
        preds = (probs > 0.5).numpy()
    
    acc = (preds == labels_test).mean()
    
    print(f"\nTest Accuracy: {acc:.1%} ({(preds == labels_test).sum()}/{len(labels_test)})")
    
    return acc


def main():
    """Main loop - iterates through each category"""
    all_results = {}
    
    for category in CONFIG['categories']:
        print(f"\n{'='*80}\nStarting pipeline for category: {category.upper()}\n{'='*80}")
        
        # 1. Load data for the specific category
        pairs = load_vqa_data(CONFIG, category)
        
        # 2. Extract hidden states (or load from category-specific cache)
        pos_h, neg_h, labels = extract_in_batches(pairs, CONFIG, category)
        
        # 3. Train and evaluate the CCS probe
        acc = train_simple_ccs(pos_h, neg_h, labels)
        
        all_results[category] = acc
        print(f"\nCOMPLETE for '{category}'! Final Accuracy: {acc:.1%}")
    
    print(f"\n{'='*80}\nALL EXPERIMENTS FINISHED\n{'='*80}")
    print("Final accuracies per category:")
    for category, acc in all_results.items():
        print(f"- {category}: {acc:.1%}")


if __name__ == "__main__":
    main()
