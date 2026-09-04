"""
Supervised NON-LINEAR (MLP) probe baseline.

the same MLP probe
architecture, but trained with a supervised BCE loss on the
ground-truth labels instead of the unsupervised CCS loss, like the
supervised linear probe in revised_supervised_vision.py, whose training
procedure is reused here unchanged (only the probe_factory differs).

Data loading and hidden-state extraction are reused from vision_ccs.py, hence
this baseline runs on the same data and hidden states.
"""

import numpy as np

# Reuse the shared (revised, CCS-aligned) pipeline pieces
from vision_ccs import (
    load_vqa_data,
    extract_in_batches,
)

# Reuse the revised supervised training procedure (seeded shared split,
# CCS-style normalization, same training budget as the CCS probe) and its
# config (which already sets the 50/50 split)
from revised_supervised_vision import (
    CONFIG as SUPERVISED_CONFIG,
    train_supervised_probe,
)

# Reuse the non-linear probe class of the unsupervised pipeline
from unsupervised_nonlinear import MLPProbe


CONFIG = {
    **SUPERVISED_CONFIG,

    # Hidden layer width of the MLP probe: 100 as in the original
    # notebook's MLPProbe (same value as in unsupervised_nonlinear.py).
    'mlp_hidden_size': 100,
}


def make_mlp_probe(input_dim):
    """Probe factory for train_supervised_probe: the notebook's MLPProbe."""
    return MLPProbe(input_dim, CONFIG['mlp_hidden_size'])


def main():
    model_key = f"model_{CONFIG['chosen_model']}"
    chosen_model_name = CONFIG.get(model_key, CONFIG['chosen_model'])
    print(f"{chosen_model_name} + Contrast Pairs + Supervised NON-LINEAR (MLP) Probe")

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
            print(f"\n!No samples extracted for '{category}'. Skipping...")
            all_results[category] = 0.0
            continue

        # 3. Train the supervised non-linear probe — same procedure as the
        #    supervised linear probe, only the architecture differs
        acc, probe = train_supervised_probe(
            pos_h, neg_h, labels, CONFIG,
            probe_factory=make_mlp_probe,
            title="SUPERVISED NON-LINEAR (MLP) PROBE"
        )

        all_results[category] = acc
        print(f"\n✓ COMPLETE: {category} → {acc:.1%}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"\nFinal Results (supervised, non-linear MLP probe):")
    for category, acc in all_results.items():
        print(f"  {category:25s}: {acc:5.1%}")

    avg_acc = np.mean(list(all_results.values()))
    print(f"\n  {'Average':25s}: {avg_acc:5.1%}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
