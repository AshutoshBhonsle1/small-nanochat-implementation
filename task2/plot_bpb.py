"""
Plot the training and validation bits-per-byte (bpb) curves from task2/script.bash's run.

Reads task2/logs/metrics.csv (written by the local-logging diff in scripts/base_train.py)
and saves task2/bpb_curve.png.

Usage:
    python -m task2.plot_bpb
"""
import csv
import os

import matplotlib.pyplot as plt

METRICS_PATH = os.path.join("task2", "logs", "metrics.csv")
OUTPUT_PATH = os.path.join("task2", "logs", "bpb_curve.png")


def load_curves(metrics_path):
    train_steps, train_bpb = [], []
    val_steps, val_bpb = [], []
    with open(metrics_path, newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if row["train_bpb"]:
                train_steps.append(step)
                train_bpb.append(float(row["train_bpb"]))
            if row["val_bpb"]:
                val_steps.append(step)
                val_bpb.append(float(row["val_bpb"]))
    return train_steps, train_bpb, val_steps, val_bpb


def main():
    train_steps, train_bpb, val_steps, val_bpb = load_curves(METRICS_PATH)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_steps, train_bpb, label="train bpb", marker="o")
    ax.plot(val_steps, val_bpb, label="val bpb", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("bits per byte")
    ax.set_title("Task 2: depth-4 base model bpb curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
