#!/bin/bash
set -e

# Activate the virtual environment
source .venv/bin/activate

# Set the base directory for nanochat (dataset, tokenizer, checkpoints, logs all live under task2/)
export NANOCHAT_BASE_DIR="./task2"

# Reuse the 32k-vocabulary tokenizer trained in Task 1 instead of retraining one.
# base_train.py's get_tokenizer()/get_token_bytes() always look for "<NANOCHAT_BASE_DIR>/tokenizer",
# so the tokenizer is copied into place under that exact name.
if [ ! -d "task1/tokenizer_32k" ]; then
    echo "task1/tokenizer_32k not found. Run task1/script.bash first to train it." >&2
    exit 1
fi
mkdir -p task2
cp -r task1/tokenizer_32k task2/tokenizer

# Download ClimbMix shards. Depth 2's compute-optimal token horizon (target-param-data-ratio=12,
# the base_train.py default) is small for such a tiny model, so 500MB is generous headroom;
# base_train.py prints the exact token requirement at startup if this ever needs to be increased.
python -m nanochat.dataset -s 500

# Pretrain the depth-2 base model.
# --window-pattern=L and --core-metric-every=-1 because this GPU (RTX 5050) has no Flash Attention 3
# support (not Hopper): SDPA's fallback has no sliding-window support, and CORE accuracy isn't a
# Task 2 deliverable, so skip downloading/running it to save time.
python -m scripts.base_train \
    --depth=2 \
    --window-pattern=L \
    --core-metric-every=-1 \
    --eval-every=100 \
    --save-every=500

# Final evaluation: bpb on both splits plus qualitative (unconditioned) completions from the
# finished checkpoint, saved to a log file for pulling numbers/quotes into the report.
mkdir -p task2/logs
python -m scripts.base_eval --eval bpb,sample --model-tag d2 | tee task2/logs/eval_d2.txt
