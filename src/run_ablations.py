"""Ablation-study driver.

Runs a list of configurations as separate processes (so that a crash in one
configuration cannot take the whole study down) and collects the resulting
``metrics.json`` files into a single CSV table that can be pasted into the
report.

Usage
-----
    python src/run_ablations.py --task 1            # FVSBN ablations
    python src/run_ablations.py --task 2            # autoregressive ablations
    python src/run_ablations.py --task all
    python src/run_ablations.py --collect-only      # only rebuild the tables
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# --------------------------------------------------------------------------------------
# Task 1 (FVSBN) -- the training set-up is identical in every run, only one factor changes
# --------------------------------------------------------------------------------------
TASK1 = [
    # id,               extra CLI arguments
    ("t1_base",        []),
    ("t1_rank50",      ["--rank", "50"]),
    ("t1_rank200",     ["--rank", "200"]),
    ("t1_block1",      ["--block-size", "1"]),
    ("t1_block784",    ["--block-size", "784"]),
    ("t1_sweeps20",    ["--n-sweeps", "20"]),
    ("t1_sweeps50",    ["--n-sweeps", "50"]),
    ("t1_binar_static", ["--binarize", "static"]),
    ("t1_binar_mean",  ["--binarize", "mean"]),
]

# --------------------------------------------------------------------------------------
# Task 2 (neural autoregressive model)
# --------------------------------------------------------------------------------------
TASK2 = [
    ("t2_hidden100", ["--hidden", "100", "--fixed-order"]),
    ("t2_nhidden1",  ["--n-hidden", "1", "--fixed-order"]),
    ("t2_no_V",      ["--no-input-weights", "--fixed-order"]),
    ("t2_nade_oa",   []),
    ("t2_gruar",     ["--model", "gruar"]),
]


def run(cmd: list) -> int:
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    return r.returncode


def collect(task: int) -> list:
    """Read every metrics.json of the ablation folders and build table rows."""
    rows = []
    base = os.path.join(ROOT, "results")
    if task == 1:
        folders = sorted(d for d in os.listdir(base) if d.startswith("ablation_t1"))
        for d in folders:
            p = os.path.join(base, d, "metrics.json")
            if not os.path.exists(p):
                continue
            m = json.load(open(p, encoding="utf-8"))
            rows.append({
                "run": d.replace("ablation_", ""),
                "binarization": m["binarization"],
                "rank": m["rank"],
                "params": m["params"],
                "epochs": m["epochs"],
                "test_bits_per_dim": round(m["test_pseudo_nll_bits_per_dim"], 4),
                "gibbs_sweeps": m["gibbs_sweeps"],
                "gibbs_block": m["gibbs_block_size"],
                "train_time_s": round(m["training_time_s"], 1),
                "gen_time_batch64_s": round(m["inference_time_batch64_s"], 3),
                "gen_time_per_sample_s": round(m["inference_time_per_sample_s"], 3),
            })
    else:
        folders = sorted(d for d in os.listdir(base) if d.startswith("ablation_t2"))
        for d in folders:
            p = os.path.join(base, d, "metrics.json")
            if not os.path.exists(p):
                continue
            m = json.load(open(p, encoding="utf-8"))
            rows.append({
                "run": d.replace("ablation_", ""),
                "model": m["model"],
                "hidden": m["hidden"],
                "n_hidden": m["n_hidden"],
                "input_weights": m["input_weights"],
                "params": m["params"],
                "epochs": m["epochs"],
                "test_bits_per_dim": round(m["test_nll_bits_per_dim"], 4),
                "train_time_s": round(m["training_time_s"], 1),
                "gen_time_batch64_s": round(m["inference_time_batch64_s"], 3),
            })
    return rows


def write_table(rows: list, path: str) -> None:
    if not rows:
        print(f"[ablation] no results for {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ablation] wrote {path} ({len(rows)} rows)")
    for r in rows:
        print("   ", r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="all", choices=["1", "2", "all"])
    ap.add_argument("--epochs1", type=int, default=40)
    ap.add_argument("--epochs2", type=int, default=25)
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    if not args.collect_only:
        if args.task in ("1", "all"):
            for run_id, extra in TASK1:
                out = os.path.join("results", f"ablation_{run_id}")
                cmd = [PY, "src/train_task1.py", "--model", "full",
                       "--epochs", str(args.epochs1), "--out", out] + extra
                run(cmd)
        if args.task in ("2", "all"):
            for run_id, extra in TASK2:
                out = os.path.join("results", f"ablation_{run_id}")
                cmd = [PY, "src/train_task2.py", "--epochs", str(args.epochs2),
                       "--batch-size", "64", "--out", out] + extra
                run(cmd)

    if args.task in ("1", "all"):
        write_table(collect(1), os.path.join(ROOT, "results", "ablation_task1.csv"))
    if args.task in ("2", "all"):
        write_table(collect(2), os.path.join(ROOT, "results", "ablation_task2.csv"))


if __name__ == "__main__":
    main()
