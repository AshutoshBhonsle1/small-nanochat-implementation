#!/bin/bash
set -e

# Activate the virtual environment
source .venv/bin/activate

# Set the base directory for nanochat (dataset, tokenizer, checkpoints, logs all live under task2/)
export NANOCHAT_BASE_DIR="./task2"

# Plot the bpb curve from the training log.
python task2/plot_bpb.py

