"""Training / evaluation script for Task 1: Fully Visible Sigmoid Belief Network.

Usage
-----
    python train_task1.py --epochs 30 --rank 0 --n-sweeps 200

Outputs (under ``results/task1``)
--------------------------------
    samples_*.png          sample grids during training (progress showcase)
    final_samples.png      final generation grid
    log.csv                per-epoch training / validation pseudo-NLL
    metrics.json           summary metrics + wall-clock timings
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

LOG2 = math.log(2)
from common import CSVLogger, Timer, count_params, ensure_dir, sample_grid_from_rows, save_image_grid, save_json, set_seed
from data import MNISTBinarized
from model_fvsbn import FVSBN, FVSBNOrdered


def parse_args():
    p = argparse.ArgumentParser(description="Task 1 - FVSBN on binarized MNIST")
    p.add_argument("--model", type=str, default="ordered", choices=["ordered", "full"],
                   help="'ordered' = autoregressive conditionals (exact likelihood, "
                        "exact ancestral sampling); 'full' = unordered conditionals "
                        "trained by pseudo-likelihood and sampled with Gibbs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--rank", type=int, default=0, help="0 = full rank (784x784), else low-rank inner dim")
    p.add_argument("--n-sweeps", type=int, default=200, help="Gibbs sweeps at generation time")
    p.add_argument("--block-size", type=int, default=32, help="pixels updated in parallel per Gibbs step")
    p.add_argument("--binarize", type=str, default="random", choices=["random", "static", "mean"])
    p.add_argument("--train-size", type=int, default=60000)
    p.add_argument("--test-size", type=int, default=10000)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="results/task1")
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    out_dir = ensure_dir(args.out)
    print(f"[task1] device={device}  out={out_dir}")

    # ------------------------------------------------------------------ data
    ds = MNISTBinarized(
        data_dir="data",
        mode=args.binarize,
        dynamic=True,
        device=device,
        seed=args.seed,
        limit_train=args.train_size,
    )
    print(f"[task1] train={ds.n_train} test={ds.n_test} binarize={args.binarize}(dynamic)")

    # deterministic binarization of the test set -> reproducible reported NLL
    test_x_fixed = ds.fixed_test_set(seed=12345)[: args.test_size]
    n_test_eval = test_x_fixed.shape[0]

    # ----------------------------------------------------------------- model
    if args.model == "ordered":
        model = FVSBNOrdered(dim=784).to(device)
        nll_is_exact = True
        model_name = "FVSBN with ordered (autoregressive) conditionals"
    else:
        model = FVSBN(dim=784, rank=args.rank if args.rank > 0 else None).to(device)
        nll_is_exact = False
        model_name = "FVSBN with unordered conditionals (pseudo-likelihood + Gibbs)"
    print(f"[task1] {model_name}")
    print(f"[task1] FVSBN params={count_params(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger = CSVLogger(os.path.join(out_dir, "log.csv"),
                       ["epoch", "train_pll", "test_pll", "train_nll_bits", "test_nll_bits", "epoch_time_s"])

    # ----------------------------------------------------------------- train
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    rows = []
    best_test = float("inf")
    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        tot, nb = 0.0, 0
        for _ in range(max(1, ds.n_train // args.batch_size)):
            x, _ = ds.get_batch(args.batch_size, "train", generator=gen)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(x)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        train_pll = tot / nb

        model.eval()
        with torch.no_grad():
            vals = []
            for _ in range(20):
                x, _ = ds.get_batch(512, "test", generator=gen)
                vals.append(model.nll(x).item())
            test_nll = sum(vals) / len(vals)

            # deterministic evaluation on the fixed test binarization (reported)
            fixed_vals = []
            for i in range(0, n_test_eval, 2000):
                fixed_vals.append(model.nll(test_x_fixed[i : i + 2000]).item())
            test_nll_fixed = sum(fixed_vals) / len(fixed_vals)

        dt = time.perf_counter() - t0
        logger.append({
            "epoch": epoch,
            "train_pll": f"{-train_pll:.4f}",
            "test_pll": f"{-test_nll_fixed:.4f}",
            "train_nll_bits": f"{train_pll / (784 * LOG2):.4f}",
            "test_nll_bits": f"{test_nll_fixed / (784 * LOG2):.4f}",
            "epoch_time_s": f"{dt:.2f}",
        })
        print(f"[task1] epoch {epoch:3d}  train PLL={-train_pll:.3f}  test PLL={-test_nll_fixed:.3f}  "
              f"({test_nll_fixed / (784 * LOG2):.4f} bits/dim)  {dt:.1f}s")
        train_time = time.perf_counter() - train_start

        # sample grid at regular intervals (progress showcase) -------------------
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                g = model.sample_grid(args.n_samples, n_sweeps=args.n_sweeps,
                                      block_size=args.block_size, device=device, generator=gen)
            rows.append(g.cpu())
            save_image_grid(g.cpu(), os.path.join(out_dir, f"samples_epoch{epoch:03d}.png"), nrow=8)
            torch.save(g.cpu(), os.path.join(out_dir, f"samples_epoch{epoch:03d}.pt"))

    total_train_time = time.perf_counter() - train_start

    # ---------------------------------------------------------- final results
    model.eval()
    with torch.no_grad():
        final_grid = model.sample_grid(args.n_samples, n_sweeps=args.n_sweeps,
                                        block_size=args.block_size, device=device, generator=gen)
    save_image_grid(final_grid.cpu(), os.path.join(out_dir, "final_samples.png"), nrow=8)
    if rows:
        sample_grid_from_rows(rows, os.path.join(out_dir, "training_progress.png"), nrow=4)

    # ------------------------------------------------------------- inference
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model.sample(args.n_samples, n_sweeps=args.n_sweeps, block_size=args.block_size,
                         device=device, generator=gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0

    # per-sample average inference time
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model.sample(1, n_sweeps=args.n_sweeps, block_size=args.block_size,
                         device=device, generator=gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_time_one = time.perf_counter() - t0

    metrics = {
        "task": "1 - Fully Visible Sigmoid Belief Network (FVSBN)",
        "model": model_name,
        "model_type": args.model,
        "nll_is_exact": nll_is_exact,
        "dataset": "binarized MNIST (random/dynamic thresholding)",
        "train_size": ds.n_train,
        "test_size": ds.n_test,
        "binarization": args.binarize,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "rank": args.rank if args.rank > 0 else "full",
        "params": count_params(model),
        "gibbs_sweeps": args.n_sweeps,
        "gibbs_block_size": args.block_size,
        "test_pseudo_nll_nats_per_dim": test_nll_fixed / 784,
        "test_pseudo_nll_bits_per_dim": test_nll_fixed / (784 * LOG2),
        "test_nll_bits_per_dim": test_nll_fixed / (784 * LOG2),
        "test_pseudo_nll_bits_per_dim_dynamic_eval": test_nll / (784 * LOG2),
        "n_test_eval": n_test_eval,
        "training_time_s": total_train_time,
        "inference_time_batch64_s": gen_time,
        "inference_time_per_sample_s": gen_time_one,
        "device": str(device),
    }
    save_json(metrics, os.path.join(out_dir, "metrics.json"))
    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(out_dir, "fvsbn.pt"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
