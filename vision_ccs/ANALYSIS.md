# VisionCCS analysis pipeline

What each script is for, which claims it can and cannot support, and the order
to run things in. The original v1 pipeline (`vision_ccs.py`,
`supervised_vision.py`) is unchanged in behaviour and still reproduces the
report; everything else here exists to test whether the report's conclusion
survives contact with the right baselines.

## The claim under test

The report concludes that CCS beats a supervised probe and that LVLM internal
representations are therefore "already discriminative". Two things make that
conclusion unsafe:

1. **Burns treats logistic regression on the internal states as the CEILING** --
   "approximately the best any method can do on just the model's internal
   states". An unsupervised method scoring above it is a signal that the
   baseline is broken, not a finding. The original baseline was
   `LogisticRegression(class_weight="balanced")` at sklearn defaults (`C=1.0`,
   `max_iter=100`) on d≈3584 features with a few hundred rows.
2. **The comparison CCS is actually about is against calibrated zero-shot** --
   whether the model's internals know more than its outputs say. That was never
   computed. Burns reports CCS 71.2% vs calibrated zero-shot 67.2%.

So the target result is `zero-shot ≤ CCS ≤ supervised`, and the interesting
question is the size of the first gap.

## Known limitations that no script here removes

- **Single prompt template by default.** Burns runs 8-13 templates per dataset,
  which is load-bearing: with one surface form there is no variation for truth
  to be the consistent feature *across*. `extract.py --templates all` fixes this
  but requires re-extraction. See `prompts.py`.
- **The final layer is the readout layer.** `hidden_states[-1]` is post-final-norm,
  i.e. the vector `lm_head` reads, and the zero-shot decision is a linear
  functional of it. Probing there and comparing to zero-shot is close to
  circular. `extract.py` + `layer_sweep.py` move off it.
- **Hallucination is motivated but not measured.** VQAv2 accuracy is not caption
  hallucination. `pope_to_vqa.py` swaps in POPE, which is the standard
  object-hallucination benchmark and has the same yes/no shape.
- **The image is never shown to matter.** `extract.py --shuffle-images` is the
  control that decides whether this is vision or a VQAv2 language prior.
- **Identifiability is not testable by accuracy.** Farquhar et al. prove
  arbitrary prominent features satisfy CCS's consistency structure, so a high
  score is compatible with the probe having found something other than truth.
  `extract.py --distractor banner` stamps a TRUE/FALSE banner on the image,
  assigned by a coin flip **independent of the answer**, and `layer_sweep.py`
  then scores the same CCS predictions against the banner as well as against
  truth. The banner column must sit at chance; above chance means the probe
  found the most prominent feature instead.

## Scripts

| script | what it does | needs GPU |
|---|---|---|
| `prompts.py` | contrast-pair templates; zero-shot prefixes are derived from the same strings so prompt parity is mechanical | - |
| `extract.py` | all-layer, named-token-position extraction; `--templates`, `--shuffle-images` control | yes |
| `zero_shot.py` | calibrated Yes/No baseline; `--template`, `--no-instruction` | yes |
| `pope_to_vqa.py` | converts POPE annotations into the `vqav2_mapped.json` schema | - |
| `reanalysis.py` | protocol-matched comparison of CCS / baselines / supervised on one cache | no (CPU fine) |
| `layer_sweep.py` | the same analysis at every (layer, position) | no |
| `transfer.py` | fit on one category, evaluate on another | no |
| `select_criteria.py` | compares label-free restart-selection rules | no |
| `compare_zeroshot.py` | item-level CCS vs zero-shot cross-tab + McNemar | no |
| `test_reanalysis.py` | unit tests; run before trusting any of the above | no |

## Protocol rules this codebase enforces

**Joins are by id, never by position.** `zero_shot.py` and `extract.py` skip
items independently, so `results[i]` and `zeroshot[i]` are not the same item.
`compare_zeroshot.py` joins on `question_id`, cross-checks the labels, reports
coverage, and refuses below `--min-coverage`.

**Splits are grouped.** 32-64% of items share a COCO image with another item, so
an ungrouped split leaks. Multi-template caches additionally hold paraphrase
siblings, so `layer_sweep.py` *requires* `--group-by image|question` for them
rather than silently allowing the leak.

**Selection is label-free and inductive.** Restart selection and layer selection
both use consistency on a held-out slice of train (`val_consistency_err`), never
the test set. `consistency_err` (measured on test inputs) is still label-free but
transductive and is offered only as an explicit opt-in.

**The full curve is the result.** Never quote the best layer. `--pick-layer`
names one honestly and prints what that choice cost against the oracle.

**Orientation is applied before any comparison.** CCS polarity is arbitrary (the
loss is invariant under `p -> 1-p` on both branches), so predictions are oriented
with the same flip `score_report` applies before being compared to zero-shot.

## Order to run

```bash
python test_reanalysis.py                 # always first, no GPU needed
sbatch run_smoke_test.sh                  # exercises every path at --limit 5

# Block A -- runs on the EXISTING caches, no re-extraction.
sbatch run_baselines.sh                   # CRC-TPC vs CCS + normalization sweep
sbatch run_transfer.sh                    # cross-category transfer
sbatch run_phase2_ablations.sh            # unit-norm x var-normalize grid

# Block B -- re-extraction, the expensive half.
sbatch run_zeroshot.sh                    # zero-shot; needed before run_analysis.sh
sbatch run_analysis.sh                    # headline + selection + zero-shot cross-tab
sbatch run_extract_v3.sh                  # all layers + shuffle + banner controls
TEMPLATES=all sbatch run_extract_v3.sh    #   ... and the multi-template caches
sbatch run_layer_sweep.sh                 # layer/position curves for all variants
```

`run_zeroshot.sh` is in block B because the existing `zeroshot/*.npz` predate
`question_id` logging: `compare_zeroshot.py` refuses to join them, by design.
Until it is re-run, the previously reported 90.9% CCS/zero-shot agreement cannot
be verified.

## How to read the results

- `CRC-TPC ≈ CCS`: the contrast pairs carry the signal, not the consistency
  loss. In text, PCA and LDA on contrast-pair differences reach 97% and 98% of
  CCS. This is the single most informative row in the output.
- `random_dir` above chance: something is wrong with the split or the
  normalization, not a discovery.
- Transfer off-diagonal near chance: the probe learned the category, not a truth
  direction.
- Shuffled-image curve tracking the real curve: the probe never needed the
  image.
- `corr(log loss, accuracy)` positive or near zero: the CCS loss is not a usable
  selection criterion, which is the scale-degeneracy result. `--weight-norm unit`
  should turn it strongly negative without raising accuracy.

### A caution about `--norm cluster`

Cluster-norm removes *between-cluster* variation. That is the point when the
clusters track a distracting feature, but if `k` is set so that the clusters end
up tracking the truth signal itself, it removes the signal. On synthetic data
with a planted direction and no distractor, `--norm cluster --cluster-k 8` drops
CCS from 88.9% to 59.7% — the method working exactly as specified, on data that
had nothing for it to strip. So it is swept as an ablation rather than made the
default, and a drop under cluster-norm is not on its own evidence that the
original signal was a distractor. The informative comparison is whether the
*ordering* of CCS against CRC-TPC survives the change.

## References

- Burns et al., *Discovering Latent Knowledge in Language Models Without
  Supervision*, arXiv:2212.03827 — the method, and the LR-as-ceiling framing.
- Farquhar et al., *Challenges with unsupervised LLM knowledge discovery*,
  arXiv:2312.10029 — arbitrary prominent features satisfy CCS's consistency
  structure.
- Fry et al., *Comparing Optimization Targets for Contrast-Consistent Search*,
  arXiv:2311.00488 — the loss is scale-degenerate.
- Burger et al., *Cluster-norm for Unsupervised Probing of Knowledge*,
  arXiv:2407.18712 — implemented as `--norm cluster`.
- Emmons, *Contrast Pairs Drive the Empirical Performance of CCS* — PCA/LDA on
  contrast-pair differences reach 97%/98% of CCS.
- Li et al., *Evaluating Object Hallucination in Large Vision-Language Models*
  (POPE) — the benchmark `pope_to_vqa.py` targets.
