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

# Reuse the 32k-vocabulary tokenizer trained in Task 1 instead of retraining one.
# base_train.py's get_tokenizer()/get_token_bytes() always look for "<NANOCHAT_BASE_DIR>/tokenizer",
# so the tokenizer is copied into place under that exact name. Remove any previous copy first:
# `cp -r src dst` copies src *into* dst (nesting) instead of overwriting it if dst already exists
# as a directory, which would otherwise pile up stale nested copies on repeated runs.
if [ ! -d "task1/tokenizer_32k" ]; then
    echo "task1/tokenizer_32k not found. Run task1/script.bash first to train it." >&2
    exit 1
fi
mkdir -p task2
rm -rf task2/tokenizer
cp -r task1/tokenizer_32k task2/tokenizer

# Download ClimbMix shards. base_train.py prints the exact required token count (from
# --target-param-data-ratio=12, its default) at startup; increase -s and rerun if 500MB
# turns out not to be enough for this depth's token horizon.
python -m nanochat.dataset -s 500

# Pretrain the depth-4 base model.
# --window-pattern=L: SDPA (forced above) has no sliding-window support, so nanochat's own
# base_train.py warns at startup and recommends --window-pattern=L (full-context attention on
# every layer) instead of its default alternating pattern when Flash Attention 3 isn't in use.
python -m scripts.base_train \
    --depth=4 \
    --eval-every=100 \
    --window-pattern=L
