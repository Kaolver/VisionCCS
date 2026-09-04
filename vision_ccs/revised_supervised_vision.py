"""
REVISED version of supervised_vision.py — aligned with the original CCS project
(ccs/CCS.ipynb + the CCS paper) and with the revised vision_ccs.py.

The original supervised_vision.py is left untouched (on request); every
deviation from it is marked with a clearly visible comment block starting with
"CHANGED". The guiding principle: a supervised baseline is only a valid
ceiling/comparison for CCS if it runs on EXACTLY the same data and the same
hidden states — so everything data-related is imported from vision_ccs.py
instead of being duplicated here.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ==============================================================================
# CHANGED (vs supervised_vision.py): all data loading and hidden-state
# extraction is reused from the revised vision_ccs.py instead of keeping a
# duplicated (and now outdated) copy here. This guarantees the supervised
# baselines see EXACTLY the same balanced, shuffled data and the same aligned
# hidden states (pooled right after the answer + EOS/end-of-turn) as the CCS
# probe. Previously supervised_vision.py still extracted at the generation
# prompt ("\nASSISTANT:" / add_generation_prompt=True), took a deterministic
# unbalanced prefix of the dataset, and wrote to its own separate cache
# directory — so the baseline was computed on different representations than
# the CCS probe it was compared against.
# (Also unified: the old file reused its Qwen2 extraction function for
# Qwen2.5; vision_ccs.py has a dedicated, equivalent function per model.)
# ==============================================================================
from vision_ccs import (
    CONFIG as VISION_CCS_CONFIG,
    load_vqa_data,
    extract_in_batches,
)


CONFIG = {
    **VISION_CCS_CONFIG,


    # ==========================================================================
    # CHANGED (vs the revised vision_ccs.py; same value as supervised_vision.py):
    # the split stays 50/50, matching the original CCS notebook ("let's create
    # a simple 50/50 train split"). NOTE: vision_ccs.py currently uses the
    # paper's 60/40 split ('train_split': 0.6); to make the CCS run use the
    # notebook's 50/50 split too (and thereby identical train/test halves to
    # this file), set 'train_split': 0.5 there — a one-value change, not made
    # here on request.
    # ==========================================================================
    'train_split': 0.5,
}


# ==============================================================================
# CHANGED (vs supervised_vision.py): shared seeded random split. The original
# file cut the data sequentially ([:n_train] / [n_train:]), copying the
# notebook's 50/50 cut — but the notebook could only do that because its data
# was "already randomized", while this project's loader used to return a
# deterministic file prefix, so the two halves could differ systematically.
# The revised loader shuffles (seeded), and this split is the same seeded
# train_test_split used in vision_ccs.train_ccs_probe — with an equal
# 'train_split' both pipelines train and evaluate on identical example sets.
# ==============================================================================
def split_indices(n_examples, config):
    """Seeded random train/test split shared with the CCS pipeline."""
    indices = np.arange(n_examples)
    return train_test_split(
        indices,
        test_size=1 - config['train_split'],
        random_state=config['random_seed']
    )


def train_supervised_classifier(neg_hs, pos_hs, y, config):
    """
    Train supervised logistic regression classifier, following CCS.ipynb.

    From CCS.ipynb:
    - Create features: x = neg_hs - pos_hs (simple difference, no normalization)
    - Train LogisticRegression with class_weight="balanced"
    - Evaluate on test set
    """
    print(f"\n{'='*70}")
    print(f"SUPERVISED LOGISTIC REGRESSION")
    print(f"{'='*70}")


    # ==========================================================================
    # CHANGED (vs supervised_vision.py): seeded random split via split_indices
    # instead of the sequential [:n_train] / [n_train:] cut (see the comment on
    # split_indices for why).
    # ==========================================================================
    train_idx, test_idx = split_indices(len(y), config)


    neg_hs_train, neg_hs_test = neg_hs[train_idx], neg_hs[test_idx]
    pos_hs_train, pos_hs_test = pos_hs[train_idx], pos_hs[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # for simplicity we can just take the difference between positive and
    # negative hidden states, exactly as in the original notebook
    # (unchanged; note the paper's LR instead uses the pair of normalized
    # representations as covariates — the notebook's difference is kept here
    # because the project follows the notebook implementation)
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

    # unchanged: exactly the notebook's classifier
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)

    test_acc = lr.score(x_test, y_test)
    print("\nLogistic regression accuracy: {:.1%}".format(test_acc))

    return test_acc, lr


def make_linear_probe(input_dim):
    """Linear probe — same architecture as the (linear) CCS probe."""
    return nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())


# ==============================================================================
# CHANGED (vs supervised_vision.py): several fixes to the supervised probe.
# - Training budget now matches the CCS probe ('ccs_epochs' = 1000 epochs at
#   'ccs_lr' = 0.01 from the shared config) instead of a hardcoded 100 epochs
#   at lr 1e-3 — the old probe was heavily undertrained relative to CCS,
#   making the supervised-vs-unsupervised comparison unfair.
# - Normalization now mirrors the (revised) CCS probe exactly: each set is
#   normalized with its OWN statistics (the old code normalized the test set
#   with train-set means, unlike CCS.get_acc in the original notebook), and
#   the optional variance normalization ('ccs_var_normalize') is applied the
#   same way.
# - The split now happens inside this function using the shared seeded
#   split_indices (the old code cut sequentially in main()).
# - The probe architecture is injectable via probe_factory (defaults to the
#   linear probe), so the non-linear supervised probe can reuse this exact
#   training procedure.
# ==============================================================================
def train_supervised_probe(pos_hiddens, neg_hiddens, labels, config,
                           probe_factory=make_linear_probe,
                           title="SUPERVISED LINEAR PROBE"):
    """Supervised probe on contrast pairs — CCS-style inputs, supervised BCE loss."""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pos_hiddens = torch.FloatTensor(pos_hiddens)
    neg_hiddens = torch.FloatTensor(neg_hiddens)

    train_idx, test_idx = split_indices(len(labels), config)

    pos_train_raw = pos_hiddens[train_idx]
    neg_train_raw = neg_hiddens[train_idx]
    pos_test_raw = pos_hiddens[test_idx]
    neg_test_raw = neg_hiddens[test_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]

    # Normalize exactly like the (revised) CCS probe: per-set statistics,
    # optional variance normalization
    def normalize(x):
        x = x - x.mean(dim=0)
        if config['ccs_var_normalize']:
            x = x / x.std(dim=0, unbiased=False)
        return x

    pos_train = normalize(pos_train_raw).to(device)
    neg_train = normalize(neg_train_raw).to(device)
    pos_test = normalize(pos_test_raw).to(device)
    neg_test = normalize(neg_test_raw).to(device)
    labels_train_tensor = torch.FloatTensor(labels_train).to(device)

    n_train_pos = (labels_train == 1).sum()
    n_train_neg = (labels_train == 0).sum()
    n_test_pos = (labels_test == 1).sum()
    n_test_neg = (labels_test == 0).sum()

    print(f"\nDataset split:")
    print(f"  Train: {len(labels_train)} samples ({n_train_pos} pos, {n_train_neg} neg)")
    print(f"  Test:  {len(labels_test)} samples ({n_test_pos} pos, {n_test_neg} neg)")
    print(f"  Hidden dim: {pos_train.shape[1]}")

    probe = probe_factory(pos_train.shape[1]).to(device)
    optimizer = optim.AdamW(
        probe.parameters(),
        lr=config['ccs_lr'],
        weight_decay=config['ccs_weight_decay']
    )
    criterion = nn.BCELoss()

    # Training (full batch, same budget as the CCS probe)
    probe.train()
    for epoch in range(config['ccs_epochs']):
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

    return test_acc, probe


def main():
    model_key = f"model_{CONFIG['chosen_model']}"
    chosen_model_name = CONFIG.get(model_key, CONFIG['chosen_model'])
    print(f"{chosen_model_name} + Contrast Pairs + Supervised Methods (revised)")

    all_results = {}

    for category in CONFIG['categories']:
        print(f"\n{'#'*70}")
        print(f"# CATEGORY: {category.upper()}")
        print(f"{'#'*70}")

        # 1. Load VQA data (shared, balanced + shuffled loader from vision_ccs)
        pairs = load_vqa_data(CONFIG, category)

        # 2. Extract hidden states (shared, aligned extraction from vision_ccs;
        #    with use_cache=True this also shares the CCS run's cache files)
        pos_h, neg_h, labels = extract_in_batches(pairs, CONFIG, category)

        # Skip if no samples extracted
        if len(pos_h) == 0:
            print(f"\n✗ No samples extracted for '{category}'. Skipping...")
            all_results[category] = {'logistic': 0.0, 'linear_probe': 0.0}
            continue

        # 3. Train supervised classifiers
        # 3a. Logistic Regression (sklearn baseline, as in the original notebook)
        logreg_acc, lr = train_supervised_classifier(neg_h, pos_h, labels, CONFIG)

        # 3b. Supervised Linear Probe (neural baseline)
        linear_probe_acc, probe = train_supervised_probe(pos_h, neg_h, labels, CONFIG)

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
