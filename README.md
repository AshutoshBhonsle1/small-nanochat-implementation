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

## Task 1: Tokenization

Task 1 trains two BPE tokenizers with nanochat's Rust tokenizer trainer on a 500 MB sample of ClimbMix — one with a vocabulary of 8,192 tokens and one with 32,768 — and compares them on compression ratio (bytes of text per token) and sequence length (tokens per text). Everything lives under `task1/`.

### Running Task 1

From the repository root:

```bash
bash task1/train_tokenizers.bash   # download data, train both tokenizers, compare them
bash task1/eval_tokenizers.bash    # rerun only the comparison, using the committed tokenizers
```

[`task1/train_tokenizers.bash`](task1/train_tokenizers.bash):

- **Lines 3–4** activate the virtual environment, so the script also works if you forgot to `source .venv/bin/activate` first.
- **Line 7** sets `NANOCHAT_BASE_DIR="./task1"`. nanochat resolves every data path (dataset shards, tokenizer output) relative to this directory, which otherwise defaults to `~/.cache/nanochat`. Setting it keeps Task 1's data and tokenizers inside `task1/`.
- **Line 10** runs `python -m nanochat.dataset -s 500` to download 500 MB of ClimbMix shards into `task1/base_data_climbmix/`. The `-s` flag didn't exist upstream and was added for this task — see Change 1 below.
- **Lines 16–18** run `python -m scripts.tok_train --vocab-size 32768` and then move the result to `task1/tokenizer_32k`. `tok_train.py` always saves to `<NANOCHAT_BASE_DIR>/tokenizer`, so each tokenizer has to be renamed before the next one is trained over it. The `rm -rf` before the `mv` matters: if `task1/tokenizer_32k` already exists from an earlier run, `mv task1/tokenizer task1/tokenizer_32k` would move the new tokenizer _inside_ the old directory instead of replacing it, leaving a stale tokenizer in place.
- **Lines 21–23** do the same for the 8k tokenizer (`--vocab-size 8192`), saved to `task1/tokenizer_8k`.
- **Lines 26–40** run the four `tok_eval.py --text ...` comparisons. These are the same commands as in `eval_tokenizers.bash`, described below.

[`task1/eval_tokenizers.bash`](task1/eval_tokenizers.bash) reruns only the comparison, without downloading or training anything, so it can be used with the committed tokenizers straight after cloning:

- **Lines 3–7** activate the virtual environment and set `NANOCHAT_BASE_DIR="./task1"` exactly as above, so `tok_eval.py` finds `task1/tokenizer_32k` and `task1/tokenizer_8k`.
- **Line 10** compares the tokenizers on an English paragraph describing sandwiches.
- **Line 13** compares them on a Latin translation of the same paragraph: the same content and alphabet, but a language other than English. The inner `"apertum"` quotes close and reopen the bash string, so the text actually passed to Python reads `genus apertum vocatur` without the quotation marks.
- **Line 16** compares them on the first 100 decimal places of π (numbers).
- **Lines 19–24** compare them on a short Python snippet (source code, including indentation and newlines) — the token-split printing loop from Change 2.

### Change 1 of 2: `nanochat/dataset.py` — download a dataset size in MB

Upstream, `nanochat.dataset` only accepts a number of shards (`-n`). The assignment asks for a 500 MB sample, so a `-s/--size-dataset` argument was added that works out the number of shards from a size in MB:

```diff
     parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
     parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
     parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
+    parser.add_argument("-s", "--size-dataset", type=int, default=0, help="Size of the dataset in MB that you want to download.")
     args = parser.parse_args()
 ...
-    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
+
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
+
     ids_to_download = list(range(num_train_shards))
     ids_to_download.append(MAX_SHARD) # always download the validation shard
```

A single HTTP `HEAD` request reads the size of the first shard without downloading it. Shards are all roughly the same size, so dividing the requested size by it and rounding **up** gives enough shards to cover at least that many MB. Without `-s` (the default `0`), the upstream `-n` behaviour is unchanged. Task 2 reuses the same flag.

### Change 2 of 2: `scripts/tok_eval.py` — compare our two tokenizers, on any text

Upstream, `tok_eval.py` compares nanochat's default tokenizer (`get_tokenizer()`, loaded from `<NANOCHAT_BASE_DIR>/tokenizer`) against the pretrained GPT-2 and GPT-4 tokenizers, on a fixed set of built-in samples. That didn't fit Task 1 for three reasons, and each one led to a change:

1. **It loads the wrong tokenizers.** Task 1 produces `tokenizer_32k` and `tokenizer_8k` and never saves anything to the plain `tokenizer/` directory, and the question is how _our two_ vocabulary sizes compare, not how either compares to GPT-2/GPT-4. The script now loads both directories with `RustBPETokenizer.from_directory` and compares them with each other. `print_comparison` is generalised from "baseline vs Ours" to any two named tokenizers.
2. **It can only evaluate built-in text.** A `--text` argument was added. When it is passed, only that text is evaluated, and the ClimbMix train/val shards are no longer read at all (upstream reads them unconditionally, which is slow and requires the dataset to be downloaded).
3. **A compression ratio doesn't show _where_ the extra tokens come from.** With `--text`, the script also prints each tokenizer's actual token split, decoding every token id back to its text with `id_to_token` and joining the pieces with `|`.

```diff
-from nanochat.tokenizer import get_tokenizer, RustBPETokenizer
+import argparse
+import os
+
+from nanochat.tokenizer import RustBPETokenizer
 from nanochat.dataset import parquets_iter_batched
+from nanochat.common import get_base_dir
+
+parser = argparse.ArgumentParser(description="Evaluate tokenizer compression ratio and sequence length")
+parser.add_argument("--text", type=str, default=None, help="Custom text to evaluate instead of the built-in samples")
+args = parser.parse_args()
 ...
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
-
-# Try out current default compared to GPT-2 and GPT-4 tokenizers
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
+
+# Compare our 32k and 8k vocabulary tokenizers against each other
+base_dir = get_base_dir()
+tokenizer_dirs = {
+    "32k": os.path.join(base_dir, "tokenizer_32k"),
+    "8k": os.path.join(base_dir, "tokenizer_8k"),
+}
+
+tokenizers = {}
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
+    tokenizers[tokenizer_name] = tokenizer

     vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
     tokenizer_results[tokenizer_name] = {}
 ...
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
 ...
-        ours_data = ours_results[name]
+        other_data = other_results[name]
 ...
-        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
+        relative_diff = ((baseline_data['tokens'] - other_data['tokens']) / baseline_data['tokens']) * 100
 ...
-        if baseline_data['ratio'] > ours_data['ratio']:
+        if baseline_data['ratio'] > other_data['ratio']:
 ...
-        elif ours_data['ratio'] > baseline_data['ratio']:
+        elif other_data['ratio'] > baseline_data['ratio']:
             baseline_color, ours_color = RED, GREEN
-            better = "Ours"
+            better = other_name
 ...
-              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
-              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
+              f"{ours_color}{other_data['tokens']:<7}{RESET} "
+              f"{ours_color}{other_data['ratio']:<7.2f}{RESET} "
 ...
-# Print comparisons
-print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
-print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)
+# Print comparison
+print_comparison("32k", tokenizer_results['32k'], "8k", tokenizer_results['8k'], all_text)
+
+# When evaluating custom text, show the actual token split for each tokenizer
+if args.text is not None:
+    for tokenizer_name, tokenizer in tokenizers.items():
+        encoded = tokenizer.encode(args.text)
+        pieces = [tokenizer.id_to_token(token_id) for token_id in encoded]
+        print(f"\n{tokenizer_name} tokenizer split ({len(pieces)} tokens):")
+        print("|".join(pieces))
```

In the comparison table, the 32k tokenizer is the baseline: `Relative Diff %` is `(tokens_32k − tokens_8k) / tokens_32k`, so a negative number means the 8k tokenizer needs that many percent more tokens than the 32k one for the same text.

Without `--text`, the script still evaluates all the built-in samples plus the ClimbMix shards, as upstream does, but for our two tokenizers:

```bash
export NANOCHAT_BASE_DIR="./task1"
python -m scripts.tok_eval
```

### Output

**Tokenizers.** Both are committed, so Task 2 onwards can use them without retraining:

```text
task1/
├── tokenizer_32k/
│   ├── tokenizer.pkl     # 404 KB
│   └── token_bytes.pt    # 132 KB
└── tokenizer_8k/
    ├── tokenizer.pkl     #  96 KB
    └── token_bytes.pt    #  36 KB
```

## Task 2: Pre-Training

Task 2 pre-trains a `--depth 4` nanochat base model on ClimbMix, using the 32k tokenizer from Task 1. It logs bits-per-byte (bpb) for both the training and the validation set, plots the two curves, and samples completions from the finished model. Everything lives under `task2/`.

Training was run on a laptop with an NVIDIA RTX 5050 Laptop GPU (Blackwell architecture), which is the reason for Change 2 below.

### Running Task 2

From the repository root, after `task1/train_tokenizers.bash` has been run at least once (or with the committed `task1/tokenizer_32k`):

```bash
bash task2/pretrain_model.bash   # download data, pre-train the depth-4 model
bash task2/eval_model.bash       # plot the bpb curves, sample completions
```

[`task2/pretrain_model.bash`](task2/pretrain_model.bash):

- **Line 2** sets `set -e`, so the script stops at the first failing command instead of, for example, starting training without a tokenizer.
- **Lines 4–5** activate the virtual environment.
- **Line 8** sets `NANOCHAT_BASE_DIR="./task2"`, so the dataset, tokenizer, checkpoints and logs all live under `task2/`, separate from Task 1's files.
- **Line 14** sets `NANOCHAT_ATTN_IMPL="sdpa"`, which forces PyTorch's SDPA attention instead of Flash Attention 3. nanochat wrongly detects Flash Attention 3 as usable on Blackwell GPUs and crashes on the first attention call ([Dao-AILab/flash-attention#1987](https://github.com/Dao-AILab/flash-attention/issues/1987)). The environment variable is new — see Change 2 below.
- **Lines 21–27** copy `task1/tokenizer_32k` to `task2/tokenizer`. `base_train.py` always loads the tokenizer from `<NANOCHAT_BASE_DIR>/tokenizer`, and Task 1 only saves tokenizers under `tokenizer_32k` and `tokenizer_8k`, so a copy under the plain name is needed. The script exits with an error if the Task 1 tokenizer doesn't exist. `rm -rf` runs before `cp -r` for the same reason as the `mv` in Task 1: copying onto an existing directory would nest the new copy inside the old one. The 32k tokenizer is used because it is nanochat's default vocabulary size; the 8k tokenizer was only trained as a comparison point.
- **Line 32** downloads 500 MB of ClimbMix into `task2/base_data_climbmix/` with the `-s` flag from Task 1's Change 1. At startup, `base_train.py` prints how many tokens the run needs. If 500 MB weren't enough, the dataloader would start a second pass over the data and `-s` should be increased. For depth 4 it was enough: training finished within the first pass.
- **Lines 38–41** run `python -m scripts.base_train` with three arguments; everything else is left at nanochat's defaults, which derive width, heads, learning rates and training horizon from `--depth`:
  - `--depth=4` is the model size asked for in the assignment.
  - `--eval-every=100` evaluates validation bpb every 100 steps (the default is 250). With Change 1, the training bpb is evaluated at the same moments, so this sets how many points the bpb curves have.
  - `--window-pattern=L` gives every layer full-context attention. nanochat's default, `SSSL`, uses sliding-window attention on three out of four layers, which SDPA doesn't support; `base_train.py` itself warns about this and recommends `L` when Flash Attention 3 isn't in use.

[`task2/eval_model.bash`](task2/eval_model.bash):

- **Lines 2–14** are the same as in `pretrain_model.bash`: stop on errors, activate the environment, set `NANOCHAT_BASE_DIR="./task2"` and force SDPA (sampling runs the model too, so it needs the same attention fix).
- **Line 17** runs [`task2/plot_bpb.py`](task2/plot_bpb.py), which reads `task2/logs/metrics.csv` (written by Change 1) and saves the training and validation bpb curves against training step to `task2/logs/bpb_curve.png`.
- **Line 19** runs `python -m scripts.base_eval --eval sample --model-tag d4` with five `--prompt` arguments, sampling a completion of each prompt from the final `d4` checkpoint with no system prompt. The `--prompt` flag is new — see Change 3 below.

### Change 1 of 3: `scripts/base_train.py` — training-set bpb, and logging to a CSV file

The assignment asks for training and validation bpb curves. Upstream `base_train.py` has two gaps:

1. **It never measures bpb on the training set.** It only evaluates bpb on the validation split. The training number it logs is a smoothed cross-entropy loss in nats per token, which is a different unit and depends on the tokenizer, so it can't be plotted against validation bpb. The function that computes bpb, `nanochat.loss_eval.evaluate_bpb`, isn't tied to the validation split, though: it evaluates the model on whatever batches it is given, and `scripts/base_eval.py` already calls it on both splits. So the fix is a second `evaluate_bpb` call, on a data loader for the training split, next to the existing validation call.
2. **It only logs to Weights & Biases.** Training ran offline on a separate machine without a wandb account, and with `--run=dummy` (the default) wandb logging is a no-op, so the numbers would only exist in the terminal output. The script now also writes every logged value to `<NANOCHAT_BASE_DIR>/logs/metrics.csv`.

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
+    metrics_writer.writerow(["step", "total_training_flops", "total_training_time", "train_loss", "train_bpb", "val_bpb"])
 ...
 train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
 build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
+build_train_bpb_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device)
 ...
 while True:
     last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
     flops_so_far = num_flops_per_token * total_batch_size * step
+    # Metrics for this step are gathered from separate blocks below (eval, loss logging)
+    # and written as a single CSV row once all of them have had a chance to run.
+    has_pending_metrics = False
+    pending_metrics = {"train_loss": "", "train_bpb": "", "val_bpb": ""}

-    # once in a while: evaluate the val bpb (all ranks participate)
+    # once in a while: evaluate the val and train bpb (all ranks participate)
     if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
         model.eval()
         val_loader = build_val_loader()
+        train_bpb_loader = build_train_bpb_loader()
         eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
         with disable_fp8(model):
             val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
-        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
+            train_bpb = evaluate_bpb(model, train_bpb_loader, eval_steps, token_bytes)
+        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f} | Train bpb: {train_bpb:.6f}")
         if val_bpb < min_val_bpb:
             min_val_bpb = val_bpb
         wandb_run.log({
             "step": step,
             "total_training_flops": flops_so_far,
             "total_training_time": total_training_time,
             "val/bpb": val_bpb,
+            "train/bpb": train_bpb,
         })
+        has_pending_metrics = True
+        pending_metrics["train_bpb"] = train_bpb
+        pending_metrics["val_bpb"] = val_bpb
         model.train()
 ...
     # termination conditions (TODO: possibly also add loss explosions etc.)
     if last_step:
+        if master_process and has_pending_metrics:
+            metrics_writer.writerow([step, flops_so_far, total_training_time, pending_metrics["train_loss"], pending_metrics["train_bpb"], pending_metrics["val_bpb"]])
+            metrics_file.flush()
         break
 ...
         wandb_run.log(log_data)
+        has_pending_metrics = True
+        pending_metrics["train_loss"] = debiased_smooth_loss
+
+    if master_process and has_pending_metrics:
+        metrics_writer.writerow([step, flops_so_far, total_training_time, pending_metrics["train_loss"], pending_metrics["train_bpb"], pending_metrics["val_bpb"]])
+        metrics_file.flush()
 ...
 # cleanup
+if master_process:
+    metrics_file.close()
 wandb_run.finish() # wandb run finish
 compute_cleanup()
```

How the pieces fit together:

- **Training-set bpb.** `build_train_bpb_loader` is a copy of nanochat's `build_val_loader` with `split="train"`. It creates a new loader each time, separate from `train_loader`, so measuring bpb never takes batches away from training. Both bpb evaluations run in the same block, at the same `--eval-every` steps and on the same number of tokens (`--eval-tokens`), so the two curves are measured the same way.
- **The CSV file.** Only the main process (`master_process`) writes it, the same convention nanochat uses for saving checkpoints. A fresh run overwrites any earlier `metrics.csv`; a run resumed with `--resume-from-step` appends to it instead, so no history is lost. The header is written whenever the file starts empty.
- **One row per step.** The bpb evaluation (every `--eval-every` steps) and the loss logging (every 100 steps) can happen on the same step. Instead of each writing its own row, both fill in a shared `pending_metrics` dict, and one row is written once both have run. Columns that weren't measured on that step are left empty. The `last_step` branch writes its row before the loop exits, because the normal write at the end of the loop is never reached on the final step.
- **wandb is untouched**, apart from also receiving `train/bpb`. Every column in the CSV is one that is already sent to wandb.

### Change 2 of 3: `nanochat/flash_attention.py` — force SDPA with `NANOCHAT_ATTN_IMPL`

On GPUs other than Hopper, nanochat decides Flash Attention 3 is available if the `kernels-community/flash-attn3` kernel package exists (`has_kernel`). It doesn't check that the package actually contains a build for the GPU it's running on. On a Blackwell GPU like the RTX 5050, the check passes, but the first attention call then crashes instead of falling back to SDPA ([Dao-AILab/flash-attention#1987](https://github.com/Dao-AILab/flash-attention/issues/1987)).

`flash_attention.py` already has an `_override_impl` variable for exactly this, but upstream it is hard-coded to `None` with no way to set it. It now reads the `NANOCHAT_ATTN_IMPL` environment variable:

```diff
+import os
 import torch
 import torch.nn.functional as F
 ...
-# Override for testing: set to 'fa3', 'sdpa', or None (auto)
-_override_impl = None
+# Override auto-detection: set NANOCHAT_ATTN_IMPL=fa3 or =sdpa to force a choice; unset = auto-detect.
+_override_impl = os.environ.get("NANOCHAT_ATTN_IMPL")
```

With `NANOCHAT_ATTN_IMPL=sdpa`, `_resolve_use_fa3()` returns `False` before it looks at the detection result, so every attention call uses SDPA. If the variable is unset, behaviour is exactly as upstream. An environment variable was chosen over a command-line flag because `flash_attention.py` is imported by the model code used by every script (`base_train.py`, `base_eval.py`, `chat_sft.py`, …), so one exported variable covers all of them.

### Change 3 of 3: `scripts/base_eval.py` — sample your own prompts with `--prompt`

The report asks for five completions from the pre-trained model. `base_eval.py --eval sample` only samples a fixed, hard-coded list of prompts, followed by eight unconditioned samples (generated from an empty prompt). A repeatable `--prompt` argument was added so the prompts can be chosen on the command line:

```diff
     parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
+    parser.add_argument('--prompt', type=str, action='append', default=None, help='Custom prompt to sample from (can be passed multiple times); replaces the default prompt list for --eval sample')
     parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
 ...
         if ddp_rank == 0:
-            prompts = [
+            prompts = args.prompt if args.prompt is not None else [
                 "The capital of France is",
                 "The chemical symbol of gold is",
                 "If yesterday was Friday, then tomorrow will be",
 ...
-            print0("\nUnconditioned samples:")
-            tokens = tokenizer("", prepend="<|bos|>")
-            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
-            for sample in uncond:
-                sample_str = tokenizer.decode(sample)
-                print0("-" * 80)
-                print0(sample_str)
-                unconditioned_samples.append(sample_str)
+            if args.prompt is None:
+                print0("\nUnconditioned samples:")
+                tokens = tokenizer("", prepend="<|bos|>")
+                uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
+                for sample in uncond:
+                    sample_str = tokenizer.decode(sample)
+                    print0("-" * 80)
+                    print0(sample_str)
+                    unconditioned_samples.append(sample_str)
```

When `--prompt` is given, it replaces the built-in prompt list and the unconditioned samples are skipped. Without it, the script behaves exactly as upstream. For example:

```bash
export NANOCHAT_BASE_DIR="./task2"
export NANOCHAT_ATTN_IMPL="sdpa"
python -m scripts.base_eval --eval sample --model-tag d4 --prompt "Once upon a time,"
```

### Output

The trained model and its training log are committed, so Tasks 3 and 4 can use them without retraining:

```text
task2/
├── base_checkpoints/d4/
│   ├── model_000528.pt          # model weights after the final step (528)
│   ├── optim_000528_rank0.pt    # optimizer state, used to warm-start Task 3
│   └── meta_000528.json         # model config, training arguments, final val bpb
└── logs/
    ├── metrics.csv              # step, FLOPs, time, train loss, train bpb, val bpb
    ├── bpb_curve.png            # training and validation bpb curves (eval_model.bash)
    └── chat_eval_base.txt       # chat_eval.py scores of the base model (see Task 3, Evaluation)
```

## Task 3: Mid-Training and Supervised Fine-Tuning

nanochat has only two training stages out of the box: pre-training (`base_train.py`) and SFT (`chat_sft.py`). Its SFT stage trains on one mixed dataset of general conversation (SmolTalk), knowledge questions (MMLU) and grade-school math (GSM8K). Task 3 splits that single run into two stages, starting from the Task 2 `d4` base checkpoint (step 528):

1. **Stage 1 — mid-training** ([`scripts/chat_midtrain.py`](scripts/chat_midtrain.py)): reasoning and knowledge data only (MMLU + GSM8K).
2. **Stage 2 — SFT** ([`scripts/chat_sft.py`](scripts/chat_sft.py)): instruction-following data only (SmolTalk), starting from the Stage 1 checkpoint instead of the base model.

Both stages were run on a single NVIDIA RTX 2080 Ti on the `tritanium` server, not on the laptop used for Task 2.

### Change 1 of 3: `scripts/chat_midtrain.py` — a new mid-training script

The assignment describes Stage 1 as editing `train_tasks` in `chat_sft.py` and Stage 2 as editing it again. Doing that in place would mean the repository only ever contains one of the two stages, and rerunning Stage 1 would require undoing the Stage 2 edit. Instead, Stage 1 gets its own script. `chat_midtrain.py` is a copy of the original `chat_sft.py` with four changes:

- **Training data:** SmolTalk is removed from `train_tasks`, leaving MMLU and GSM8K, as the assignment asks.
- **Validation data:** SmolTalk is removed from the validation mixture as well, so Stage 1's validation bpb only measures the kind of data that stage trains on.
- **Output directory:** the checkpoint is saved to `chatmidtrain_checkpoints/` instead of `chatsft_checkpoints/`. Otherwise Stage 2 would overwrite it, and Stage 2 couldn't tell the two checkpoints apart.
- **wandb project:** runs are logged under `nanochat-midtrain` instead of `nanochat-sft`, so the two stages appear as separate projects.

Diff against the original `chat_sft.py`:

```diff
-parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
+parser = argparse.ArgumentParser(description="Midtraining the model")
 ...
-wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)
+wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-midtrain", name=args.run, config=user_config)
 ...
 # SFT data mixture and DataLoader
 train_tasks = [
-    SmolTalk(split="train"), # 460K rows of general conversations
     *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
     *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
 ]
 train_dataset = TaskMixture(train_tasks)
 print0(f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")
 val_dataset = TaskMixture([
-    SmolTalk(split="test"), # 24K rows in test set
     MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
     GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
 ]) # total: 24K + 5.2K + 0.42K ~= 29.6K rows
 ...
-        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
+        checkpoint_dir = os.path.join(base_dir, "chatmidtrain_checkpoints", output_dirname)
```

Everything else is unchanged from `chat_sft.py`. The script still loads the `"base"` checkpoint and its optimizer state, and keeps all of the original command-line arguments, including `--mmlu-epochs` (default 3) and `--gsm8k-epochs` (default 4).

### Change 2 of 3: `nanochat/checkpoint_manager.py` — a `"midtrain"` checkpoint source

Scripts don't load checkpoints by path. They call `load_model(source, ...)` with a source name, and `load_model` looks the name up in a fixed table to find the checkpoint directory under `NANOCHAT_BASE_DIR`. The table only knew `"base"`, `"sft"` and `"rl"`, so there was no way to load the new `chatmidtrain_checkpoints/` directory. A `"midtrain"` entry is added:

```diff
 def load_model(source, *args, **kwargs):
     model_dir = {
         "base": "base_checkpoints",
+        "midtrain": "chatmidtrain_checkpoints",
         "sft": "chatsft_checkpoints",
         "rl": "chatrl_checkpoints",
     }[source]
```

Stage 2 uses this to load the Stage 1 checkpoint (Change 3). Because `scripts/chat_eval.py` loads models through the same `load_model`, the Stage 1 checkpoint can also be evaluated with `chat_eval.py -i midtrain`.

### Change 3 of 3: `scripts/chat_sft.py` — SmolTalk only, loaded from the Stage 1 checkpoint

For Stage 2, the assignment asks for SFT on SmolTalk only, starting from the Stage 1 checkpoint instead of the base model. That takes two edits to `chat_sft.py`: the model is loaded from the `"midtrain"` source added in Change 2, and MMLU and GSM8K are removed from `train_tasks`.

```diff
 # Load the model and tokenizer
-model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)
+model, tokenizer, meta = load_model("midtrain", device, phase="train", model_tag=args.model_tag, step=args.model_step)
 ...
 train_tasks = [
     SmolTalk(split="train"), # 460K rows of general conversations
-    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
-    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
 ]
```

The validation mixture was left unchanged (SmolTalk + MMLU + GSM8K test splits). Because the `load_model` source is now hard-coded to `"midtrain"`, `chat_sft.py` can no longer start from a base checkpoint: it always loads from `chatmidtrain_checkpoints`.

### Output

Each stage produced a checkpoint and a wandb run directory. Both were copied into the repository from the server:

```text
task3_midtrain/                              # Stage 1: mid-training
├── chat_eval_midtrain.txt                   # chat_eval.py scores of this checkpoint (see Evaluation)
├── chatmidtrain_checkpoints/d4/
│   ├── model_000464.pt                      # model weights after the final step (464)
│   ├── optim_000464_rank0.pt                # optimizer state
│   └── meta_000464.json                     # model config, training arguments, final val bpb
└── midtrain_logs/                           # wandb run directory
    ├── files/
    │   ├── output.log                       # full terminal output: per-step loss, val bpb, final evaluation
    │   ├── config.yaml                      # all command-line arguments of the run
    │   ├── wandb-summary.json               # final value of every logged metric
    │   ├── wandb-metadata.json              # exact command, git commit, hardware
    │   └── requirements.txt                 # installed Python packages
    ├── logs/                                # wandb's own debug logs
    └── run-tty968p7.wandb                   # binary wandb run file (all logged metrics)
task3_sft/                                   # Stage 2: SFT
├── chat_eval_sft.txt                        # chat_eval.py scores of this checkpoint (see Evaluation)
├── chatsft_checkpoints/d4/
│   ├── model_000718.pt                      # model weights after the final step (718)
│   ├── optim_000718_rank0.pt
│   └── meta_000718.json
└── chatsft_logs/                            # wandb run directory, same layout as above
    ├── files/
    ├── logs/
    └── run-hki626fq.wandb
```

The checkpoints were produced by these commands (taken from each run's `wandb-metadata.json`). Both ran with nanochat's default `NANOCHAT_BASE_DIR`, `~/.cache/nanochat`:

As with Task 2, the `.pt` files are stored with Git LFS.

### Evaluation

The pre-trained, mid-trained and SFT checkpoints were each evaluated with nanochat's [`scripts/chat_eval.py`](scripts/chat_eval.py) on the full test sets of ARC-Easy, ARC-Challenge, MMLU, GSM8K (1,319 problems) and HumanEval (164 problems). These scores are the ones in the report's Task 3 evaluation table. Each result file sits next to the checkpoint it evaluates:

| File | Checkpoint evaluated |
|---|---|
| [`task2/logs/chat_eval_base.txt`](task2/logs/chat_eval_base.txt) | pre-trained base model (`task2/base_checkpoints/d4`, step 528) |
| [`task3_midtrain/chat_eval_midtrain.txt`](task3_midtrain/chat_eval_midtrain.txt) | Stage 1, mid-training (`chatmidtrain_checkpoints/d4`, step 464) |
| [`task3_sft/chat_eval_sft.txt`](task3_sft/chat_eval_sft.txt) | Stage 2, SFT (`chatsft_checkpoints/d4`, step 718) |

Each file is the raw terminal output of one run: the progress lines of each benchmark, a `<task> accuracy: …` line for all five benchmarks, and the final `ChatCORE metric`: each task's accuracy rescaled so that 0 is random guessing and 1 is perfect, averaged over the five tasks.

The runs were done on the RTX 5050 laptop, with the same environment variables as Task 2 and `NANOCHAT_BASE_DIR` pointing at the folder that holds each checkpoint. nanochat loads the tokenizer from `<NANOCHAT_BASE_DIR>/tokenizer`, so the Task 1 tokenizer is first copied there, as in `pretrain_model.bash`:

```bash
source .venv/bin/activate
export NANOCHAT_ATTN_IMPL="sdpa"
NANOCHAT_BASE_DIR=./task2 python -m scripts.chat_eval -i base -g d4 2>&1 | tee task2/logs/chat_eval_base.txt
rm -rf task3_midtrain/tokenizer && cp -r task1/tokenizer_32k task3_midtrain/tokenizer
NANOCHAT_BASE_DIR=./task3_midtrain python -m scripts.chat_eval -i midtrain -g d4 2>&1 | tee task3_midtrain/chat_eval_midtrain.txt
rm -rf task3_sft/tokenizer && cp -r task1/tokenizer_32k task3_sft/tokenizer
NANOCHAT_BASE_DIR=./task3_sft python -m scripts.chat_eval -i sft -g d4 2>&1 | tee task3_sft/chat_eval_sft.txt
```

## Repository layout

```text
task1/
|-- base_data_climbmix/  # downloaded ClimbMix shards
|-- tokenizer_32k/       # trained 32k-vocab tokenizer
|-- tokenizer_8k/        # trained 8k-vocab tokenizer
|-- train_tokenizers.bash # dataset download + tokenizer training + comparison
`-- eval_tokenizers.bash  # tokenizer comparison only
task2/
|-- base_checkpoints/d4/ # saved model + optimizer checkpoint
|-- logs/                # metrics.csv (train/val bpb), bpb_curve.png, chat_eval_base.txt
|-- plot_bpb.py          # renders logs/bpb_curve.png from logs/metrics.csv
|-- pretrain_model.bash  # dataset download + pre-training
`-- eval_model.bash      # bpb plot + sample completions
task3_midtrain/
|-- chat_eval_midtrain.txt        # chat_eval.py scores of the Stage 1 checkpoint
|-- chatmidtrain_checkpoints/d4/  # Stage 1 (mid-training) model + optimizer checkpoint
`-- midtrain_logs/                # wandb run files; files/output.log is the full training log
task3_sft/
|-- chat_eval_sft.txt             # chat_eval.py scores of the Stage 2 checkpoint
|-- chatsft_checkpoints/d4/       # Stage 2 (SFT) model + optimizer checkpoint
`-- chatsft_logs/                 # wandb run files; files/output.log is the full training log
nanochat/                # project source code
scripts/                 # executable Python scripts
tests/                   # test suite
pyproject.toml           # project and dependency configuration
uv.lock                 # locked dependency versions
```
