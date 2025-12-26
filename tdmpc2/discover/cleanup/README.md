# `discover/cleanup`: W&B storage cleanup utilities

This folder contains targeted utilities for cleaning up W&B storage usage.

## Model registry cleanup (`cleanup-models`)

Training logs checkpoints to W&B as artifacts of type `model`.

### What the cleanup keeps

- **Latest checkpoint step per expert**: For each expert key (derived from artifact collection name),
  keep only the **max-step** checkpoint collection and delete older-step collections.
- **Only first+last versions per kept step**: For the kept collection, keep only `v0` and the
  latest version `vN` (plus any protected aliases), and delete intermediate versions.

### Why `--artifact-name` exists (avoid “stuck” scans)

Project-wide scans can be slow because W&B requires iterating many artifact collections.

- Use `--artifact-name <name>` to fetch a specific artifact base name directly (fast-path).
- Use `--name-regex <regex>` to filter by artifact base name while scanning (slower but broad).

### Examples

### What to run now (recommended)

Dry-run first (recommended). This will keep only the latest checkpoint step per expert, and for that
kept step it will keep only the **first + last** versions (`v0` and `vN`):

```bash
module load miniconda/24.11_environmentally && conda activate newt
cd tdmpc2

python -m discover --wandb-project wm-planning/mmbench cleanup-models --name-regex 'expert-'
```

Apply deletions (set a safety cap high enough for your project):

```bash
python -m discover --wandb-project wm-planning/mmbench cleanup-models \
  --name-regex 'expert-' \
  --apply --max-delete 200000
```

If project-wide scans feel slow, you can target specific artifact base names without scanning:

```bash
python -m discover --wandb-project wm-planning/mmbench cleanup-models \
  --artifact-name 'walker-walk-incline-expert-walker-walk-incline-1-2_500_000'
```

### More examples

Dry-run one artifact base name:

```bash
python -m discover --wandb-project wm-planning/mmbench cleanup-models \
  --artifact-name 'walker-walk-incline-expert-walker-walk-incline-1-2_500_000'
```

Apply deletions (set a safety cap high enough):

```bash
python -m discover --wandb-project wm-planning/mmbench cleanup-models \
  --name-regex 'expert-' \
  --apply --max-delete 200000
```


