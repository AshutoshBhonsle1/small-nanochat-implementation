#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Set the base directory for nanochat (Optional, directory is set by default to the home directory, but this is useful if you want control of which directory the dataset is downloaded to)
export NANOCHAT_BASE_DIR="./task1"

# Download the dataset using custom argument for size in MB.
python -m nanochat.dataset -s 500

# Train the 32k vocabulary size tokenizer and save it
python -m scripts.tok_train --vocab-size 32768 
mv task1/tokenizer task1/tokenizer_32k

# Train the 8k vocabulary size tokenizer and save it
python -m scripts.tok_train --vocab-size 8192 
mv task1/tokenizer task1/tokenizer_8k