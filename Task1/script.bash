#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Set the base directory for nanochat (Optional, directory is set by default to the home directory, but this is useful if you want control of which directory the dataset is downloaded to)
export NANOCHAT_BASE_DIR="./Task1"

# Download the dataset (each shard is ~90Mb, so 6 shards is apprximately 500Mb)
python -m nanochat.dataset -n 6