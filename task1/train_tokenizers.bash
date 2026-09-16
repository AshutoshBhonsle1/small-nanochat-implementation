#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Set the base directory for nanochat (Optional, directory is set by default to the home directory, but this is useful if you want control of which directory the dataset is downloaded to)
export NANOCHAT_BASE_DIR="./task1"

# Download the dataset using custom argument for size in MB.
python -m nanochat.dataset -s 500

# Train the 32k vocabulary size tokenizer and save it.
# Remove any previous copy first: `mv src dst` moves src *into* dst (nesting) instead of
# overwriting it if dst already exists as a directory, which would otherwise pile up stale
# nested copies on repeated runs.
python -m scripts.tok_train --vocab-size 32768
rm -rf task1/tokenizer_32k
mv task1/tokenizer task1/tokenizer_32k

# Train the 8k vocabulary size tokenizer and save it
python -m scripts.tok_train --vocab-size 8192
rm -rf task1/tokenizer_8k
mv task1/tokenizer task1/tokenizer_8k

# Compare the two tokenizers using the evaluation script
python -m scripts.tok_eval --text "A sandwich is a dish typically consisting variously of meat, cheese, sauces, and vegetables used as a filling between slices of bread, or placed atop a slice of bread (known as an open sandwich); or, more generally, any dish in which bread serves as a container or wrapper for another food type. The sandwich began as a portable, convenient food in the Western world, and over time it has become prevalent worldwide."

# Compare the tokenizers on latin script (Is a direct translation of the above text into Latin)
python -m scripts.tok_eval --text "Paniculus fartus est cibus qui plerumque constat ex carne, caseo, condimentis, et holeribus inter segmenta panis interpositis, vel super unum segmentum impositis (quod genus "apertum" vocatur); vel, latiore sensu, est quilibet cibus in quo panis pro vase vel involucro alterius generis cibi fungitur. Hic cibus, initio in orbe Occidentali ut res facilis ad portandum et commoda ortus, tempore procedente ubique terrarum innotuit."

# Compare the tokenizers on numerical data (pi to 100 decimal places)
python -m scripts.tok_eval --text "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Compare the tokenizers on source code (Python code for a simple function)
python -m scripts.tok_eval --text 'if args.text is not None:
    for tokenizer_name, tokenizer in tokenizers.items():
        encoded = tokenizer.encode(args.text)
        pieces = [tokenizer.id_to_token(token_id) for token_id in encoded]
        print(f"\n{tokenizer_name} tokenizer split ({len(pieces)} tokens):")
        print("|".join(pieces))'