# Agentic LLM Assignment 1

This repository contains the code for Assignment 1 of the Agentic LLM course.

The project is intended to run on Linux. Windows users should use WSL (Windows Subsystem for Linux) and perform **every step inside WSL**, including cloning the repository, installing dependencies, creating the virtual environment, and running the scripts.

## Requirements

- Linux, or Windows with WSL2
- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Internet access for installing dependencies and downloading the dataset
- A CUDA-capable GPU for the GPU setup, or a CPU for the CPU setup

## Windows users: set up WSL

Open PowerShell as Administrator and install WSL2 with Ubuntu:

```powershell
wsl --install -d Ubuntu
```

Restart your computer if Windows asks you to. Then open **Ubuntu** from the Start menu and create a Linux username and password.

If WSL is already installed, start it with `wsl` from PowerShell, then continue with the Linux instructions below inside the WSL terminal.

## Installation

### 1. Clone the repository

Run this command in Linux or WSL:

```bash
git clone https://git.liacs.nl/s5107911/agenticllm-assignment1-group3.git
cd agenticllm-assignment1-group3
```

### 2. Install `uv`

If `uv` is not installed yet, install it inside Linux/WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal, or load the shell configuration suggested by the installer. Verify the installation with `uv --version`.

### 3. Create the environment and install dependencies

Choose **one** option from the repository root:

For a CUDA-capable NVIDIA GPU:

```bash
uv sync --extra gpu
```

For CPU-only execution:

```bash
uv sync --extra cpu
```

These commands create `.venv` and install the project dependencies. Do not create the environment from PowerShell; it must be created inside Linux/WSL.

### 4. Activate the environment

```bash
source .venv/bin/activate
```

The environment must be active whenever you run the assignment code.

## Download the ClimbMix shards

From the repository root, with the virtual environment activated, run:

```bash
bash Task1/script.bash
```

The script downloads 500 MegaBytes of ClimbMix shards (approximately 5.4, rounded to 6) into `Task1/base_data_climbmix`.

This is done via a new argument added to the original dataset.py for passing the desired dataset size.

The code diff is as follows:

```diff
--- a/nanochat/dataset.py
+++ b/nanochat/dataset.py
@@ -136,14 +136,26 @@
 if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
     parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
     parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
+    parser.add_argument("-s", "--size-dataset", type=int, default=0, help="Size of the dataset in MB that you want to download.")
     args = parser.parse_args()
 
     # Prepare the output directory
     os.makedirs(DATA_DIR, exist_ok=True)
 
     # The way this works is that the user specifies the number of train shards to download via the -n flag.
     # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
 
-    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
+    if args.size_dataset == 0:
+        num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
+    else:
+        import math
+        #get the size of a single shard in mb
+        url_of_single_shard = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_00000.parquet"
+        response = requests.head(url_of_single_shard, allow_redirects=True)
+        size_shard_bytes = int(response.headers.get('content-length', 0))
+        size_shard_mb = size_shard_bytes / 1000000
+        #calculate the number of required shards
+        num_train_shards = math.ceil(args.size_dataset/size_shard_mb)
 
     ids_to_download = list(range(num_train_shards))
     ids_to_download.append(MAX_SHARD) # always download the validation shard
```

## Train the tokenizers

Tokenizer training is included in [`Task1/script.bash`](Task1/script.bash). Running the script trains both required tokenizers with nanochat's Rust BPE tokenizer trainer:

- a 32k-vocabulary tokenizer, saved to `Task1/tokenizer_32k`
- an 8k-vocabulary tokenizer, saved to `Task1/tokenizer_8k`

The script sets `NANOCHAT_BASE_DIR="./Task1"`, so the tokenizers are saved alongside the downloaded dataset.

The resulting files are:

```text
Task1/
├── tokenizer_32k/
│   ├── tokenizer.pkl
│   └── token_bytes.pt
└── tokenizer_8k/
    ├── tokenizer.pkl
    └── token_bytes.pt
```

## Quick start

For a fresh Linux or WSL installation, the complete sequence is:

```bash
git clone https://git.liacs.nl/s5107911/agenticllm-assignment1-group3.git
cd agenticllm-assignment1-group3
uv sync --extra gpu  # use --extra cpu for CPU-only execution
source .venv/bin/activate
bash Task1/script.bash
```

Run all commands from the WSL/Linux terminal. If you chose the CPU setup, replace `gpu` with `cpu` in the `uv sync` command.

## Repository layout

```text
Task1/
|-- base_data_climbmix/  # downloaded ClimbMix shards
`-- script.bash          # dataset download script
nanochat/                # project source code
scripts/                 # executable Python scripts
tests/                   # test suite
pyproject.toml           # project and dependency configuration
uv.lock                 # locked dependency versions
```

## Troubleshooting

- **`uv: command not found`**: restart the Linux/WSL terminal after installing `uv`, or follow the PATH instructions printed by the installer.
- **`git: command not found`**: install Git inside Linux/WSL with `sudo apt update && sudo apt install git`.
- **GPU setup problems in WSL**: ensure WSL2 and the Windows NVIDIA driver are installed and that `nvidia-smi` works inside WSL. Otherwise use the CPU setup.
