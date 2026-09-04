"""
Unsupervised CCS with a NON-LINEAR (MLP) probe.

Linear(d, 100) -> ReLU -> Linear(100, 1) -> Sigmoid ~
instead of a single linear layer + sigmoid. Everything else (contrast pairs,
hidden-state extraction, normalization, losses, random restarts, evaluation)
is identical to the linear CCS pipeline.

Data loading and hidden-state extraction are reused from vision_ccs.py. 
The training function is re-defined here only
because vision_ccs.train_ccs_probe has its linear probe class baked in; apart
from the probe architecture it follows the same procedure.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Reuse the shared (revised, CCS-aligned) pipeline pieces
from vision_ccs import (
    CONFIG as VISION_CCS_CONFIG,
    load_vqa_data,
    extract_in_batches,
    lr_sanity_check,
)


CONFIG = {
    **VISION_CCS_CONFIG,

    # 50/50 split, matching the original CCS notebook (see the note in
    # revised_supervised_vision.py: vision_ccs.py itself currently uses the
    # paper's 60/40 — set its 'train_split' to 0.5 to make all runs identical).
    'train_split': 0.5,

    # Hidden layer width of the MLP probe - 100 as in the original
    # notebook's MLPProbe.
    'mlp_hidden_size': 100,
}


class MLPProbe(nn.Module):
    """Non-linear probe from the original CCS notebook (class MLPProbe)."""

    def __init__(self, d, hidden_size=100):
        super().__init__()
        self.linear1 = nn.Linear(d, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)


def train_ccs_probe_nonlinear(pos_hiddens, neg_hiddens, labels, config):
    """
    Train the CCS probe with the non-linear MLPProbe.

    Identical to vision_ccs.train_ccs_probe (same split, normalization,
    losses, restarts, and evaluation) except for the probe architecture.
    """
    print(f"\n{'='*70}")
    print(f"TRAINING CCS PROBE (NON-LINEAR / MLP)")
    print(f"{'='*70}")

    # Convert to tensors
    pos_hiddens = torch.FloatTensor(pos_hiddens)
    neg_hiddens = torch.FloatTensor(neg_hiddens)

    n = len(labels)
    indices = np.arange(n)

    # Label-free seeded random split (same as vision_ccs.train_ccs_probe)
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

    # Normalization: per-set mean-centering, optional variance normalization
    # (same as vision_ccs.train_ccs_probe / CCS.normalize in the original)
    def normalize(x):
        x = x - x.mean(dim=0)
        if config['ccs_var_normalize']:
            x = x / x.std(dim=0, unbiased=False)
        return x

    pos_train = normalize(pos_train_raw)
    neg_train = normalize(neg_train_raw)
    pos_test = normalize(pos_test_raw)
    neg_test = normalize(neg_test_raw)

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
    print(f"  MLP hidden size: {config['mlp_hidden_size']}")

    # Multiple random restarts (avoid local minima)
    best_loss = float('inf')
    best_probe = None

    print(f"\n{'='*70}")
    print(f"TRAINING WITH MULTIPLE RANDOM RESTARTS")
    print(f"{'='*70}")

    for trial in range(config['ccs_ntries']):
        # ======================================================================
        # The only difference from vision_ccs.train_ccs_probe: the probe is the
        # original notebook's MLPProbe instead of Linear + Sigmoid.
        # ======================================================================
        probe = MLPProbe(pos_hiddens.shape[1], config['mlp_hidden_size']).to(device)

        optimizer = optim.AdamW(
            probe.parameters(),
            lr=config['ccs_lr'],
            weight_decay=config['ccs_weight_decay']
        )

        # Random permutation each run (mirrors the original train())
        permutation = torch.randperm(len(pos_train))
        pos_train_run = pos_train[permutation]
        neg_train_run = neg_train[permutation]

        # Training loop (full batch)
        probe.train()
        for epoch in range(config['ccs_epochs']):
            p_pos = probe(pos_train_run)
            p_neg = probe(neg_train_run)

            consistency_loss = ((p_pos - (1 - p_neg)) ** 2).mean()
            confidence_loss = (torch.min(p_pos, p_neg) ** 2).mean()

            loss = consistency_loss + confidence_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Last training-step loss selects the best restart (as in the original)
        final_loss = loss.detach().cpu().item()

        print(f"  Trial {trial+1:2d}/{config['ccs_ntries']}: Loss = {final_loss:.6f}")

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

        probs = 0.5 * (p_pos + (1 - p_neg))
        preds = (probs > 0.5).cpu().numpy()

    raw_accuracy = (preds == labels_test).mean()

    # Orientation flip (max(acc, 1-acc)), as in the original
    if raw_accuracy < 0.5:
        accuracy = 1 - raw_accuracy
        preds_corrected = 1 - preds
        correct = (preds_corrected == labels_test).sum()
    else:
        accuracy = raw_accuracy
        preds_corrected = preds
        correct = (preds == labels_test).sum()

    total = len(labels_test)

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
    """Non-linear (MLP) Vision-CCS pipeline."""
    model_key = f"model_{CONFIG['chosen_model']}"
    chosen_model_name = CONFIG[model_key]
    print(f"Model: {chosen_model_name} — CCS with NON-LINEAR (MLP) probe")

    all_results = {}

    for category in CONFIG['categories']:
        print(f"\n{'#'*70}")
        print(f"# CATEGORY: {category.upper()}")
        print(f"{'#'*70}")

        # 1. Load VQA data (shared, balanced + shuffled loader)
        pairs = load_vqa_data(CONFIG, category)

        # 2. Extract hidden states (shared, aligned extraction; with
        #    use_cache=True the cache is shared with the other pipelines)
        pos_h, neg_h, labels = extract_in_batches(pairs, CONFIG, category)

        if len(pos_h) == 0:
            print(f"\n✗ No samples extracted for '{category}'. Skipping...")
            all_results[category] = 0.0
            continue

        # Supervised logistic-regression sanity check (as in the original
        # notebook, diagnostic only)
        if CONFIG['run_lr_sanity_check']:
            lr_sanity_check(pos_h, neg_h, labels, CONFIG)

        # 3. Train the non-linear CCS probe
        acc, probe = train_ccs_probe_nonlinear(pos_h, neg_h, labels, CONFIG)

        all_results[category] = acc
        print(f"\n✓ COMPLETE: {category} → {acc:.1%}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"\nFinal Results (CCS, non-linear MLP probe):")
    for category, acc in all_results.items():
        print(f"  {category:25s}: {acc:5.1%}")

    avg_acc = np.mean(list(all_results.values()))
    print(f"\n  {'Average':25s}: {avg_acc:5.1%}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
