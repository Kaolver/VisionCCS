# Snellius runbook — VisionCCS

Replace `USER` with your SURF username.
LOCAL = `/home/martin/Documents/Snellius/VisionCCS/VisionCCS/vision_ccs`

COCO is NOT uploaded. Both CONFIGs read it from the shared cluster path
`/scratch-nvme/ml-datasets/coco/{train,val}/data`. Your home dir starting
empty is correct.

## Datasets

Two question sets, both over COCO images already on the cluster:

| Key    | Files                          | Categories (one probe each)          |
|--------|--------------------------------|--------------------------------------|
| `vqa2` | `vqav2_mapped.json`            | object_detection, attribute_recognition, spatial_recognition |
| `pope` | `pope/coco_pope_{random,popular,adversarial}.json` | random, popular, adversarial (3000 q each, 1500 yes / 1500 no) |

`vqa2` is the default. Switch per job with an environment variable — no file
edits, no re-upload:

    VISIONCCS_DATASET=pope sbatch run_linear_ccs.sh
    VISIONCCS_DATASET=pope sbatch --dependency=afterok:<id> run_linear_supervised.sh

The value is printed at the top of every `.out` (`dataset : pope`). The two
datasets keep separate hidden-state caches (`cache_pope_*` vs `cache_*`), so
stage 1 must be run once per dataset; stages 2-4 then reuse the matching
cache. On int3, `export VISIONCCS_DATASET=pope` before `python linear_ccs.py`.

## Stages

All four share one CONFIG and one cache (`./hidden_states_cache_final`).
`use_cache` is now `True`, so stage 1 extracts once and stages 2-4 reuse it.
Run stage 1 first and let it finish.

| # | Script | Entrypoint |
|---|--------|-----------|
| 1 | `run_linear_ccs.sh`           | `linear_ccs.py`           |
| 2 | `run_linear_supervised.sh`    | `linear_supervised.py`    |
| 3 | `run_nonlinear_ccs.sh`        | `nonlinear_ccs.py`        |
| 4 | `run_nonlinear_supervised.sh` | `nonlinear_supervised.py` |

## 1. Check the home dir is clean

    ssh USER@snellius.surf.nl
    ls -la ~
    rm -rf ~/VisionCCS          # only if an old copy exists
    ls /scratch-nvme/ml-datasets/coco/train/data | head -3

## 2. Upload (from your laptop)

    LOCAL=/home/martin/Documents/Snellius/VisionCCS/VisionCCS/vision_ccs
    ssh USER@snellius.surf.nl 'mkdir -p ~/VisionCCS/vision_ccs'
    rsync -avz \
      "$LOCAL"/linear_ccs.py "$LOCAL"/linear_supervised.py \
      "$LOCAL"/nonlinear_ccs.py "$LOCAL"/nonlinear_supervised.py \
      "$LOCAL"/vqav2_mapped.json "$LOCAL"/pope "$LOCAL"/requirements.txt \
      "$LOCAL"/_common.sh "$LOCAL"/run_*.sh \
      USER@snellius.surf.nl:~/VisionCCS/vision_ccs/

That is the complete runtime set: the four entrypoints plus the shared
`_common.sh` and the dataset JSONs (`vqav2_mapped.json` and the `pope/` dir).

## 3a. Submit to the gpu_mig reservation

    ssh USER@snellius.surf.nl
    cd ~/VisionCCS/vision_ccs
    chmod +x run_*.sh

    sbatch run_linear_ccs.sh                       # note the job id
    sbatch --dependency=afterok:<id1> run_linear_supervised.sh
    sbatch --dependency=afterok:<id1> run_nonlinear_ccs.sh
    sbatch --dependency=afterok:<id1> run_nonlinear_supervised.sh

Stages 2-4 all depend on stage 1 (for the cache and the venv) but not on each
other. Chain them one-by-one instead if you want to read each result first.

Do NOT submit stage 1 and another stage at the same time: both would build the
same `./venv`, and two concurrent pip installs into one venv corrupt it. The
`venv/.complete` marker makes stages 2-4 skip the build once it exists.

## 3b. Alternative: the free int3 node

int3 is interactive and free (1/7 A100), reachable only from the CLI. There is
no SLURM there — run the entrypoint directly, inside tmux so a dropped
connection does not kill the run:

    ssh USER@snellius.surf.nl
    ssh int3
    cd ~/VisionCCS/vision_ccs
    tmux new -s ccs
    source _common.sh          # same modules + venv as the batch jobs
    python linear_ccs.py 2>&1 | tee linear-ccs_int3.out

Detach with `ctrl-b d`, reattach with `tmux attach -t ccs`. Sourcing
`_common.sh` works outside SLURM too: `$SLURM_SUBMIT_DIR` is unset, so it falls
back to `$HOME/VisionCCS/vision_ccs`.

int3 has no 3 h wall-clock limit, so nothing stops a runaway job — use
`--time`-capped batch jobs if you want the guarantee.

## 4. Watch

    squeue -u $USER
    tail -f linear-ccs_<jobid>.out
    scancel <jobid>

## 5. Fetch results

    rsync -avz USER@snellius.surf.nl:'~/VisionCCS/vision_ccs/*.out' "$LOCAL"/results/

## Credit safety

Every script carries `--time=03:00:00`; SLURM hard-kills at 3 h. Billing is on
elapsed time, so a 40-minute run bills 40 minutes. Override per submit with
`sbatch --time=01:00:00 run_linear_ccs.sh`.

A MIG slice is ~10GB of GPU memory, far less than a full A100's 40/80GB. If a
7B model still hits CUDA OOM at `batch_size: 8`, lower it further in the
`linear_ccs.py` CONFIG. Raising `--mem` will not help: that is host RAM.
