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
bash task1/script.bash
```

The script downloads 500 MegaBytes of ClimbMix shards (approximately 5.4, rounded to 6) into `task1/base_data_climbmix`.

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

Tokenizer training is included in [`task1/script.bash`](task1/script.bash). Running the script trains both required tokenizers with nanochat's Rust BPE tokenizer trainer:

- a 32k-vocabulary tokenizer, saved to `task1/tokenizer_32k`
- an 8k-vocabulary tokenizer, saved to `task1/tokenizer_8k`

The script sets `NANOCHAT_BASE_DIR="./task1"`, so the tokenizers are saved alongside the downloaded dataset.

The resulting files are:

```text
task1/
├── tokenizer_32k/
│   ├── tokenizer.pkl
│   └── token_bytes.pt
└── tokenizer_8k/
    ├── tokenizer.pkl
    └── token_bytes.pt
```

## Compare tokenizers

[`scripts/tok_eval.py`](scripts/tok_eval.py) measures each tokenizer's compression ratio (bytes of raw text per token) and resulting sequence length (number of tokens) across a handful of text samples (news, Korean, code, math/LaTeX, science) plus the actual ClimbMix train/val shards.

Originally the script compared nanochat's default tokenizer (`get_tokenizer()`, loaded from `<NANOCHAT_BASE_DIR>/tokenizer`) against the pretrained GPT-2 and GPT-4 (`cl100k_base`) tokenizers. Since this assignment only produces the 32k and 8k tokenizers above, and neither is ever saved to the plain `tokenizer/` directory, the script was changed to load `tokenizer_32k` and `tokenizer_8k` directly and compare them against each other instead of against GPT-2/GPT-4:

```diff
-from nanochat.tokenizer import get_tokenizer, RustBPETokenizer
+import os
+
+from nanochat.tokenizer import RustBPETokenizer
 from nanochat.dataset import parquets_iter_batched
+from nanochat.common import get_base_dir

@@
-# Try out current default compared to GPT-2 and GPT-4 tokenizers
+# Compare our 32k and 8k vocabulary tokenizers against each other
+base_dir = get_base_dir()
+tokenizer_dirs = {
+    "32k": os.path.join(base_dir, "tokenizer_32k"),
+    "8k": os.path.join(base_dir, "tokenizer_8k"),
+}
+
 tokenizer_results = {}
 vocab_sizes = {}

-for tokenizer_name in ["gpt2", "gpt4", "ours"]:
+for tokenizer_name, tokenizer_dir in tokenizer_dirs.items():

-    if tokenizer_name == "gpt2":
-        tokenizer = RustBPETokenizer.from_pretrained("gpt2") # gpt-2 base model tokenizer
-    elif tokenizer_name == "gpt4":
-        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base") # gpt-4 base model tokenizer
-    else:
-        tokenizer = get_tokenizer()
+    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

     vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
     tokenizer_results[tokenizer_name] = {}

@@
 # Print vocab sizes
 print(f"\nVocab sizes:")
-print(f"GPT-2: {vocab_sizes['gpt2']}")
-print(f"GPT-4: {vocab_sizes['gpt4']}")
-print(f"Ours: {vocab_sizes['ours']}")
+print(f"32k tokenizer: {vocab_sizes['32k']}")
+print(f"8k tokenizer: {vocab_sizes['8k']}")

-def print_comparison(baseline_name, baseline_results, ours_results, all_text):
-    """Print comparison table between baseline tokenizer and ours."""
-    print(f"\nComparison with {baseline_name}:")
+def print_comparison(baseline_name, baseline_results, other_name, other_results, all_text):
+    """Print comparison table between two tokenizers."""
+    print(f"\nComparison: {baseline_name} vs {other_name}:")
     print("=" * 95)
-    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
+    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {other_name:<15} {'Relative':<12} {'Better':<10}")
     print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
     print("-" * 95)

     for name, text in all_text:
         baseline_data = baseline_results[name]
-        ours_data = ours_results[name]
+        other_data = other_results[name]

-        # Calculate relative difference (positive means ours is better, negative means worse)
-        # Using tokens: fewer tokens is better, so we calculate (baseline_tokens - ours_tokens) / baseline_tokens
-        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
+        # Calculate relative difference (positive means other is better, negative means worse)
+        # Using tokens: fewer tokens is better, so we calculate (baseline_tokens - other_tokens) / baseline_tokens
+        relative_diff = ((baseline_data['tokens'] - other_data['tokens']) / baseline_data['tokens']) * 100

         # Determine which has better compression (higher ratio = better)
-        if baseline_data['ratio'] > ours_data['ratio']:
+        if baseline_data['ratio'] > other_data['ratio']:
             baseline_color, ours_color = GREEN, RED
             better = baseline_name
             diff_color = RED
-        elif ours_data['ratio'] > baseline_data['ratio']:
+        elif other_data['ratio'] > baseline_data['ratio']:
             baseline_color, ours_color = RED, GREEN
-            better = "Ours"
+            better = other_name
             diff_color = GREEN
         else:
             baseline_color, ours_color = "", ""
             better = "Tie"
             diff_color = ""

@@
         print(f"{name:<10} {baseline_data['bytes']:<8} "
               f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
               f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
-              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
-              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
+              f"{ours_color}{other_data['tokens']:<7}{RESET} "
+              f"{ours_color}{other_data['ratio']:<7.2f}{RESET} "
               f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
               f"{better:<10}")

-# Print comparisons
-print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
-print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)
+# Print comparison
+print_comparison("32k", tokenizer_results['32k'], "8k", tokenizer_results['8k'], all_text)
```

Run it from the repository root, with the virtual environment activated and `NANOCHAT_BASE_DIR` set as in [`task1/script.bash`](task1/script.bash):

```bash
export NANOCHAT_BASE_DIR="./task1"
python -m scripts.tok_eval
```

For each text sample, the printed table reports the byte count, each tokenizer's token count (sequence length) and ratio (bytes per token), and the relative difference in sequence length between the two. As expected, the 32k tokenizer compresses text into fewer tokens than the 8k tokenizer across every sample, at the cost of a larger vocabulary.

To evaluate the tokenizers on your own text instead of the built-in samples, a `--text` argument was added. When it is passed, the built-in samples (and the ClimbMix train/val shard loading, which is otherwise done unconditionally) are skipped, and the comparison runs on just the given text:

```diff
+parser = argparse.ArgumentParser(description="Evaluate tokenizer compression ratio and sequence length")
+parser.add_argument("--text", type=str, default=None, help="Custom text to evaluate instead of the built-in samples")
+args = parser.parse_args()
+
 # Random text I got from a random website this morning
 news_text = r"""
@@
-# The tokenizer was trained on data from earlier shards, so it has seen this data
-train_docs = next(parquets_iter_batched(split="train"))
-train_text = "\n".join(train_docs)
-val_docs = next(parquets_iter_batched(split="val"))
-val_text = "\n".join(val_docs)
-
-all_text = [
-    ("news", news_text),
-    ("korean", korean_text),
-    ("code", code_text),
-    ("math", math_text),
-    ("science", science_text),
-    ("climbmix-train", train_text),
-]
-if val_text:
-    all_text.append(("climbmix-val", val_text))
+if args.text is not None:
+    all_text = [("custom", args.text)]
+else:
+    # The tokenizer was trained on data from earlier shards, so it has seen this data
+    train_docs = next(parquets_iter_batched(split="train"))
+    train_text = "\n".join(train_docs)
+    val_docs = next(parquets_iter_batched(split="val"))
+    val_text = "\n".join(val_docs)
+
+    all_text = [
+        ("news", news_text),
+        ("korean", korean_text),
+        ("code", code_text),
+        ("math", math_text),
+        ("science", science_text),
+        ("climbmix-train", train_text),
+    ]
+    if val_text:
+        all_text.append(("climbmix-val", val_text))
```

Usage:

```bash
python -m scripts.tok_eval --text "your custom sentence here"
```

To make the compression tangible rather than just a ratio number, when `--text` is passed the script also prints the actual token split for each tokenizer, using `id_to_token` to decode each token id back to its text piece:

```diff
 # Print comparison
 print_comparison("32k", tokenizer_results['32k'], "8k", tokenizer_results['8k'], all_text)
+
+# When evaluating custom text, show the actual token split for each tokenizer
+if args.text is not None:
+    for tokenizer_name, tokenizer in tokenizers.items():
+        encoded = tokenizer.encode(args.text)
+        pieces = [tokenizer.id_to_token(token_id) for token_id in encoded]
+        print(f"\n{tokenizer_name} tokenizer split ({len(pieces)} tokens):")
+        print("|".join(pieces))
```

For example, `python -m scripts.tok_eval --text "The quick brown fox jumps over the lazy dog, tokenization is fascinating."` prints:

```text
32k tokenizer split (15 tokens):
The| quick| brown| fox| jumps| over| the| lazy| dog|,| token|ization| is| fascinating|.

8k tokenizer split (20 tokens):
The| quick| brown| f|ox| j|umps| over| the| l|az|y| dog|,| to|ken|ization| is| fascinating|.
```

The smaller 8k vocabulary has to break common words like "fox", "jumps", and "lazy" into multiple subword pieces, while the 32k vocabulary has room to keep them as single tokens — visibly showing where the extra compression comes from.

## Task 2: Pre-Training

Task 2 pre-trains a `--depth 4` nanochat base model on ClimbMix under `task2/` (mirroring the `task2/script.bash` layout used for Task 1), logging bits-per-byte (bpb) — not raw loss — for both the training and validation sets so the deliverable curves are apples-to-apples across runs. Training happens on a separate machine from where this repo is edited, so all logging is written to a local CSV file instead of relying on a Weights & Biases account or internet access.

### Running Task 2

From the repository root, with the virtual environment activated, and after `task1/script.bash` has already been run at least once (Task 2 reuses the 32k-vocabulary tokenizer it trains):

```bash
bash task2/script.bash
```

[`task2/script.bash`](task2/script.bash):

- Sets `NANOCHAT_BASE_DIR="./task2"`, so the dataset, tokenizer, checkpoints, and logs all live under `task2/`, separate from Task 1's `task1/`.
- Copies `task1/tokenizer_32k` to `task2/tokenizer`. `base_train.py` always loads the tokenizer from `<NANOCHAT_BASE_DIR>/tokenizer` (via `nanochat.tokenizer.get_tokenizer()`/`get_token_bytes()`), and Task 1 never saves a tokenizer under that plain name (only `tokenizer_32k`/`tokenizer_8k`), so a copy is needed. The 32k tokenizer is used (over the 8k one) because it's the more standard choice for actually training a model, rather than a comparison point.
- Downloads 500MB of ClimbMix via `python -m nanochat.dataset -s 500` (the same `-s` flag added for Task 1). `base_train.py` prints the exact required token count (from `--target-param-data-ratio=12`, its default) at startup; increase `-s` and rerun if 500MB turns out not to be enough for depth 4's token horizon.
- Sets `NANOCHAT_ATTN_IMPL="sdpa"` to force PyTorch's SDPA attention fallback instead of nanochat's auto-detected Flash Attention 3. On the RTX 5050 Laptop GPU (Blackwell/SM 120) this was run on, nanochat's `has_kernel()` availability check finds a matching community FA3 kernel *package*, but that package has no compiled binary for SM 120, which crashes at runtime instead of falling back gracefully — see Change 3 below.
- Runs `scripts.base_train --depth=4` with `--core-metric-every=-1` (nanochat's default is every 2000 steps): CORE accuracy isn't a Task 2 deliverable, so it's skipped to avoid downloading a separate eval bundle and to save time. `--eval-every=50` logs validation bpb every 50 steps. `--save-every` and `--window-pattern` are both left at nanochat's own defaults: a checkpoint is only saved at the very end of training either way (`--save-every=-1` is already the default), and the forced SDPA fallback (below) still computes sliding-window attention correctly via an explicit mask even under the default `SSSL` window pattern, just somewhat less efficiently than a specialized windowed kernel — a real but minor cost for a model this tiny.
- Finishes with `scripts.base_eval --eval bpb,sample --model-tag d4`, piped to `task2/logs/eval_d4.txt`, for a clean final bpb readout on both splits plus qualitative (unconditioned, no-system-prompt) sample completions — take 5 of the printed unconditioned samples for the report's qualitative-inspection deliverable.

### Why a training-set bpb curve needs a code change

Out of the box, nanochat only computes bpb for the **validation** set (`nanochat.loss_eval.evaluate_bpb`, called periodically during training). The **training** curve it logs is the raw mean cross-entropy loss in nats per token, via `model(x, y)`. Nats-per-token and bits-per-byte are not the same unit: bpb additionally weights by how many raw bytes each token decodes to, so it stays comparable across tokenizers/vocab sizes (this is exactly why the assignment asks for bpb instead of loss in the first place).

Recomputing an *exact* bytes-weighted bpb on every training step would mean switching the training forward pass to `loss_reduction='none'` and restructuring the gradient-accumulation/backward logic — a much bigger, riskier change for a metric that's only needed for logging. Instead, we reuse the "bytes per counted token" ratio that `evaluate_bpb` already computes internally during each periodic validation pass, and use it to convert the EMA training loss into an *approximate* training bpb:

```
train_bpb ≈ train_loss_nats / (ln(2) * bytes_per_token)
```

`bytes_per_token` is refreshed every time validation runs (`--eval-every` steps) and held constant between refreshes. This is an approximation (a single averaged conversion factor applied to the whole training curve between eval points, rather than an exact per-batch byte count), which is noted here so it can be flagged in the report if relevant.

### Change 1 of 3: `nanochat/loss_eval.py` — let `evaluate_bpb` also report `bytes_per_token`

```diff
 @torch.no_grad()
-def evaluate_bpb(model, batches, steps, token_bytes):
+def evaluate_bpb(model, batches, steps, token_bytes, return_bytes_per_token=False):
     ...
     # record the losses
     total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
     total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
+    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
     batch_iter = iter(batches)
     for _ in range(steps):
         ...
             total_nats += (loss2d * (num_bytes2d > 0)).sum()
             total_bytes += num_bytes2d.sum()
+            total_tokens += (num_bytes2d > 0).sum()
         else:
             ...
             total_nats += (loss2d * (num_bytes2d > 0)).sum()
             total_bytes += num_bytes2d.sum()
+            total_tokens += (num_bytes2d > 0).sum()
     # sum reduce across all ranks
     world_size = dist.get_world_size() if dist.is_initialized() else 1
     if world_size > 1:
         dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
         dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
+        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
     # move both to cpu, calculate bpb and return
     total_nats = total_nats.item()
     total_bytes = total_bytes.item()
+    total_tokens = total_tokens.item()
     if total_bytes == 0:
-        return float('inf')
+        return (float('inf'), 0.0) if return_bytes_per_token else float('inf')
     bpb = total_nats / (math.log(2) * total_bytes)
+    if return_bytes_per_token:
+        bytes_per_token = total_bytes / total_tokens if total_tokens > 0 else 0.0
+        return bpb, bytes_per_token
     return bpb
```

What each part of this diff does:

- **New `return_bytes_per_token` parameter, default `False`.** `evaluate_bpb` is also called from [`scripts/base_eval.py`](scripts/base_eval.py) (final-checkpoint bpb evaluation) without expecting the extra value. Defaulting the new behavior to off means every existing call site keeps working unmodified — only `scripts/base_train.py` opts in.
- **`total_tokens` accumulator.** The function already computes `num_bytes2d`, a per-token byte count that is `0` for tokens that should be excluded from the metric (special tokens like `<|bos|>`, and ignored/masked targets) and the real byte length otherwise. `(num_bytes2d > 0).sum()` counts how many *included* tokens were seen in this batch — added next to the existing `total_bytes` (sum of byte lengths) and `total_nats` (sum of losses) accumulators, in both the masked and fast code paths, so the count stays consistent with whichever tokens actually contributed to `total_bytes`.
- **`total_tokens` is all-reduced too**, for the same reason `total_nats`/`total_bytes` already are: under multi-GPU (DDP) training each rank only sees a shard of the batch, so the counts must be summed across ranks before dividing, otherwise a multi-GPU bpb would be computed from only one rank's fraction of the data.
- **`bytes_per_token = total_bytes / total_tokens`.** This is the average number of raw bytes each *counted* token decoded to, over the batches just evaluated — exactly the conversion factor described above, computed from real data rather than guessed from the tokenizer's vocabulary in the abstract (which would over/under-weight tokens by how often they actually appear in ClimbMix text).
- **Return shape only changes when opted in.** With `return_bytes_per_token=True`, the function returns `(bpb, bytes_per_token)` instead of just `bpb`; the `total_bytes == 0` edge case (a degenerate empty batch) is handled the same way, returning `(inf, 0.0)` instead of just `inf`, so callers that unpack a tuple never get a bare float when they least expect it.

### Change 2 of 3: `scripts/base_train.py` — local CSV logging and the approximate `train/bpb`

```diff
 import os
 os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
 import gc
+import csv
 import json
 ...
 base_dir = get_base_dir()
 output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
 checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
 resuming = args.resume_from_step != -1
+
+# Metrics CSV: fresh runs start a clean file (truncate any previous run's rows),
+# resumed runs append to the existing history instead of wiping it.
+os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
+metrics_path = os.path.join(base_dir, "logs", "metrics.csv")
+metrics_file = open(metrics_path, "a" if resuming else "w", newline="") if master_process else None
+metrics_writer = csv.writer(metrics_file) if master_process else None
+if master_process and metrics_file.tell() == 0:
+    metrics_writer.writerow(["step", "train_loss", "train_bpb", "val_bpb"])
 ...
 if not resuming:
     step = 0
     val_bpb = None # will be set if eval_every > 0
+    bytes_per_token = None # will be set once val bpb is first evaluated
     min_val_bpb = float("inf")
     smooth_train_loss = 0 # EMA of training loss
     total_training_time = 0 # total wall-clock time of training
 else:
     step = meta_data["step"]
     loop_state = meta_data["loop_state"]
     val_bpb = meta_data["val_bpb"]
+    bytes_per_token = loop_state.get("bytes_per_token")
     min_val_bpb = loop_state["min_val_bpb"]
     smooth_train_loss = loop_state["smooth_train_loss"]
     total_training_time = loop_state["total_training_time"]
 ...
         with disable_fp8(model):
-            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
+            val_bpb, bytes_per_token = evaluate_bpb(model, val_loader, eval_steps, token_bytes, return_bytes_per_token=True)
         print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
         if val_bpb < min_val_bpb:
             min_val_bpb = val_bpb
         wandb_run.log({
             "step": step,
             "total_training_flops": flops_so_far,
             "total_training_time": total_training_time,
             "val/bpb": val_bpb,
         })
+        if master_process:
+            metrics_writer.writerow([step, "", "", val_bpb])
+            metrics_file.flush()
         model.train()
 ...
     if step % 100 == 0:
+        train_bpb = debiased_smooth_loss / (math.log(2) * bytes_per_token) if bytes_per_token else None
         log_data = {
             "step": step,
             "total_training_flops": flops_so_far,
             "total_training_time": total_training_time,
             "train/loss": debiased_smooth_loss,
+            "train/bpb": train_bpb,
             "train/lrm": lrm,
             "train/dt": dt,
             "train/tok_per_sec": tok_per_sec,
             "train/mfu": mfu,
             "train/epoch": epoch,
         }
         wandb_run.log(log_data)
+        if master_process:
+            metrics_writer.writerow([step, debiased_smooth_loss, train_bpb if train_bpb is not None else "", ""])
+            metrics_file.flush()
 ...
                 "loop_state": { # all loop state (other than step) so that we can resume training
                     "min_val_bpb": min_val_bpb,
                     "smooth_train_loss": smooth_train_loss,
                     "total_training_time": total_training_time,
+                    "bytes_per_token": bytes_per_token,
                 },
 ...
 # cleanup
+if master_process:
+    metrics_file.close()
 wandb_run.finish() # wandb run finish
 compute_cleanup()
```

What each part of this diff does:

- **`metrics.csv` opened once, guarded by `master_process`.** In distributed (multi-GPU/DDP) training every rank runs this script, but only rank 0 (`master_process`) should own the log file, exactly like nanochat's own convention for checkpoint saving and sampling elsewhere in this file. It's opened in truncate mode (`"w"`) for a fresh run — so starting a new training run always begins from a clean `metrics.csv`, with no leftover rows from a previous/different run — and in append mode (`"a"`) when `--resume-from-step` is set, so resuming continues the same curve instead of losing its history. The header row (`step,train_loss,train_bpb,val_bpb`) is written whenever the file comes up empty (`metrics_file.tell() == 0`), which covers both the fresh-run case and a resume whose file didn't already exist.
- **`bytes_per_token` threaded through loop state.** It starts as `None` (no val eval has happened yet) and is set every time the periodic validation block runs, using the `return_bytes_per_token=True` flag added to `evaluate_bpb` in Change 1. It's also saved into and restored from the checkpoint's `loop_state`, so resuming a run from a checkpoint keeps a sensible conversion factor immediately, instead of leaving `train/bpb` unset until the next validation pass.
- **Val-eval block:** unpacks the new `(val_bpb, bytes_per_token)` return and appends a CSV row with just `val_bpb` filled in (the `train_loss`/`train_bpb` columns are left blank for this row, since no training step happened at this exact log point — the two curves are logged on their own independent cadences: `--eval-every` for validation, every 100 steps for training).
- **Train-loss block:** computes `train_bpb = train_loss_nats / (ln(2) * bytes_per_token)`, using whatever `bytes_per_token` was last measured by validation. This is only computed and logged once `bytes_per_token` is available (i.e., after the first validation pass) — before that, `train/bpb` is `None` and the CSV cell is left blank rather than written as a wrong/undefined number.
- **`wandb_run.log(...)` calls are unchanged in kind** (still called even under `--run=dummy`, where they're a harmless no-op via nanochat's `DummyWandb`) — the CSV write is purely additive, so this diff doesn't remove or depend on wandb in any way.
- **File closed at the very end**, after the training loop but before `wandb_run.finish()`/`compute_cleanup()`, so the CSV is always flushed and closed cleanly when the script exits normally.

### Change 3 of 3: `nanochat/flash_attention.py` — let `NANOCHAT_ATTN_IMPL` force the SDPA fallback

nanochat auto-detects Flash Attention 3 by checking whether a matching *kernel package* can be downloaded for the GPU's compute-capability major version (`kernels.has_kernel(...)`), then trusts that package to actually run. On an RTX 5050 Laptop GPU (Blackwell, SM 120), `has_kernel("kernels-community/flash-attn3")` returns `True` — a package exists — but that package has no compiled binary for SM 120 specifically, so the very first attention call crashes with `CUDA error: no kernel image is available for execution on the device` instead of nanochat falling back to its SDPA implementation the way it does on genuinely unsupported hardware (e.g. CPU/MPS). This is a real gap in nanochat's detection logic, not a training bug.

There was already an `_override_impl` variable in the file for exactly this kind of situation, but nothing ever set it to anything other than `None` — no CLI flag or env var wired it up. The fix makes it configurable:

```diff
+import os
 import torch
 import torch.nn.functional as F
 ...
-# Override for testing: set to 'fa3', 'sdpa', or None (auto)
-_override_impl = None
+# Override for testing, and for GPUs where has_kernel() finds a matching kernel package but that
+# package has no compiled binary for this specific compute capability (e.g. Blackwell/sm120 consumer
+# GPUs with the community FA3 kernel), which otherwise crashes at runtime instead of falling back.
+# Set NANOCHAT_ATTN_IMPL=fa3 or NANOCHAT_ATTN_IMPL=sdpa to force a choice; unset = auto-detect.
+_override_impl = os.environ.get("NANOCHAT_ATTN_IMPL")
```

`task2/script.bash` sets `export NANOCHAT_ATTN_IMPL="sdpa"` before running `scripts.base_train`/`scripts.base_eval`, which makes `_resolve_use_fa3()` return `False` unconditionally — every `flash_attn_func`/`flash_attn_with_kvcache` call then takes the `SDPA fallback` branch (`nanochat/flash_attention.py`'s existing `if USE_FA3: ... ` checks) and never touches the broken kernel package at all.

### Plotting the bpb curves

Once `task2/script.bash` has finished (or partway through a long run — `metrics.csv` is flushed after every logged row), run:

```bash
python -m task2.plot_bpb
```

[`task2/plot_bpb.py`](task2/plot_bpb.py) reads `task2/logs/metrics.csv` and saves `task2/bpb_curve.png` with matplotlib — plotting `train_bpb` and `val_bpb` against `step` on the same axes. This is the "training and validation set bits-per-byte curves" deliverable.

## Quick start

For a fresh Linux or WSL installation, the complete sequence is:

```bash
git clone https://git.liacs.nl/s5107911/agenticllm-assignment1-group3.git
cd agenticllm-assignment1-group3
uv sync --extra gpu  # use --extra cpu for CPU-only execution
source .venv/bin/activate
bash task1/script.bash
bash task2/script.bash
python -m task2.plot_bpb
```

Run all commands from the WSL/Linux terminal. If you chose the CPU setup, replace `gpu` with `cpu` in the `uv sync` command.

## Repository layout

```text
task1/
|-- base_data_climbmix/  # downloaded ClimbMix shards
`-- script.bash          # dataset download script
task2/
|-- base_checkpoints/d4/ # saved model + optimizer checkpoint
|-- logs/                # metrics.csv (train/val bpb) and eval_d4.txt (final bpb + samples)
|-- plot_bpb.py          # renders bpb_curve.png from logs/metrics.csv
`-- script.bash          # dataset download + pre-training + eval driver
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
