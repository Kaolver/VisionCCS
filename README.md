# Vision CCS

Unsupervised truth detection in vision-language models using Contrast Consistent Search (CCS).

## Note

Changes to this README and project organization were made after the project deadline for ease of use in terms of reproducibility. **No changes have been made to the actual code, implementation, or methodology.** Only organizational improvements and documentation were added.

If ease of use is part of the evaluation rubric, please refer to the project state at the deadline on the **martin branch**(which was later merged to the main for visibility) via the commit history. The core implementation and results were completed on that branch by the deadline.

## Setup

1. Navigate to the project directory:
```bash
cd /path/to/vision_ccs
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

**Before running, modify the hardcoded paths in the shell scripts to match your setup.**

From the project root directory, run either:

**For supervised approach:**
```bash
sbatch run_supervised_vision.sh
```

**For unsupervised CCS approach:**
```bash
sbatch run_vision_ccs.sh
```

## Configuration

Edit `CONFIG` in the respective script:

- `n_samples_*`: Number of samples per category (auto-loaded from dataset)
- `batch_size`: Processing batch size (default: 40)
- `use_cache`: Whether to use cached hidden states
- `vqa_json`: Path to VQA dataset
- `image_dirs`: List of COCO image directories
- `cache_dir`: Directory for cached hidden states
- `categories`: Question categories to evaluate
- `chosen_model`: Model selection ('llava', 'qwen2', or 'qwen2_5')
- `train_split`: Train/test split ratio
- `ccs_epochs`: Training epochs (unsupervised only)
- `ccs_ntries`: Random restarts (unsupervised only)
- `ccs_lr`: Learning rate
- `ccs_weight_decay`: Weight decay for regularization

## Output

- Hidden states cached in respective cache directories
- Accuracy results printed per category and averaged
