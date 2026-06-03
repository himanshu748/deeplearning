# Repository Instructions

## Scope
This repository is a legacy ML project for forest-cover semantic segmentation. Keep it reproducible, honest about requirements, and safe to run locally.

## Commands
- `python3 -m compileall train.py src tests`
- `python3 -m unittest discover -s tests`
- `python train.py --data-dir <dataset> --epochs 1 --models unet`

## Conventions
- Do not commit datasets, checkpoints, Kaggle credentials, virtualenvs, generated caches, or experiment logs.
- Keep CLI defaults suitable for local experimentation; document GPU-heavy runs clearly.
- Prefer deterministic helpers that can be tested without downloading datasets or importing the full deep-learning stack.
- Validate user-provided model names, split ratios, dataset paths, and image/mask reads before training.
