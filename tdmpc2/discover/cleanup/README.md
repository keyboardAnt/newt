# `discover/cleanup`: W&B storage cleanup utilities

This folder contains targeted utilities for cleaning up W&B storage usage.

## Model registry cleanup (`cleanup-models`)

Training logs checkpoints to W&B as artifacts of type `model`.

### What the cleanup keeps

- **Latest checkpoint step per expert**: For each expert key (derived from artifact collection name),
  keep only the **max-step** checkpoint collection and delete older-step collections.
- **Only first+last versions per kept step**: For the kept collection, keep only `v0` and the
  latest version `vN` (plus any protected aliases), and delete intermediate versions.

### Why `--collection` exists (avoid “stuck” scans)

Project-wide scans can be slow because W&B requires iterating many artifact collections.

- Use `--collection <name>` to fetch a specific collection directly (fast-path).
- Use `--name-regex <regex>` to filter by collection name while scanning (slower but broad).

### Examples

Dry-run one collection:

```bash
python -m discover --wandb-project wm-planning/mmbench cleanup-models \
  --collection 'walker-walk-incline-expert-walker-walk-incline-1-2_500_000'
```

Apply deletions (set a safety cap high enough):

```bash
python -m discover --wandb-project wm-planning/mmbench cleanup-models \
  --name-regex 'expert-' \
  --apply --max-delete 200000
```


