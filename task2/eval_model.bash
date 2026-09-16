#!/bin/bash
set -e

# Activate the virtual environment
source .venv/bin/activate

# Set the base directory for nanochat (dataset, tokenizer, checkpoints, logs all live under task2/)
export NANOCHAT_BASE_DIR="./task2"

# Due to a nanochat bug, it can't correctly detect Flash Attention 3 support on RTX 5050 GPUs
# (https://github.com/Dao-AILab/flash-attention/issues/1987),
# so we force the SDPA attention instead, via the NANOCHAT_ATTN_IMPL env var added in
# nanochat/flash_attention.py.
export NANOCHAT_ATTN_IMPL="sdpa"

# Plot the bpb curve from the training log.
python task2/plot_bpb.py

python -m scripts.base_eval --eval sample --model-tag d4 --prompt "In the beginning" --prompt "The chemical formula of water is" --prompt "In the year 2050" --prompt "The racecar sped" --prompt "The sandwich is"

