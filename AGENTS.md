# Repository Instructions

## Scope
This repository is a legacy ML project for forest-cover semantic segmentation. Keep it reproducible, honest about requirements, and safe to run locally.

## Commands
- `PYTHONPYCACHEPREFIX=/private/tmp/deeplearning-pycache python3 -m compileall train.py src tests`
- `python3 -m unittest discover -s tests`
- `python train.py --data-dir <dataset> --dry-run`
- `python train.py --data-dir <dataset> --epochs 1 --models unet`
- `python train.py --download-data --dry-run` only when Kaggle credentials/network are intentionally available

## Conventions
- Do not commit datasets, checkpoints, Kaggle credentials, virtualenvs, generated caches, or experiment logs.
- Keep CLI defaults suitable for local experimentation; document GPU-heavy runs clearly.
- Keep the CLI local-first: do not make network downloads implicit.
- Prefer deterministic helpers that can be tested without downloading datasets or importing the full deep-learning stack.
- Keep `train.py --dry-run` lightweight: it may validate local data and splits, but it must not build models, create dataloaders, initialize Torch runtime state, create checkpoint directories, or start training.
- Validate user-provided model names, split ratios, dataset paths, and image/mask reads before training.
- Keep tests focused on pure helpers unless provider datasets and GPU resources are explicitly available.
