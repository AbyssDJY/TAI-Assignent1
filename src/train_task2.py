"""Training / evaluation script for Task 2: neural autoregressive model.

Trains a NADE (order-agnostic MADE) on binarized MNIST and evaluates:
    * test NLL in bits/dim (the standard metric for binarized MNIST),
    * samples from exact ancestral sampling.

Usage
-----
    python train_task2.py --model nade --epochs 40 --hidden 500 --n-hidden 2
    python train_task2.py --model gruar --epochs 30 --hidden 512
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from common import CSVLogger, count_params, ensure_dir, sample_grid_from_rows, save_image_grid, save_json, set_seed
from data import MNISTBinarized
from model_nade import GRUAR, NADE

LOG2 = math.log(2)


def parse_args():
    p = argparse.ArgumentParser(description="Task 2 - autoregressive model on binarized MNIST")
    p.add_argument("--model", type=str, default="nade", choices=["nade", "gruar"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=500)
    p.add_argument("--n-hidden", type=int, default=2)
    p.add_argument("--no-input-weights", action="store_true",
                   help="disable NADE's direct input->output weights (ablation)")
    p.add_argument("--fixed-order", action="store_true",
                   help="use the natural raster order instead of a fresh random "
                        "ordering per example (ablation: NADE -> fixed-order MADE)")
    p.add_argument("--binarize", type=str, default="random", choices=["random", "static", "mean"])
    p.add_argument("--train-size", type=int, default=60000)
    p.add_argument("--test-size", type=int, default=10000)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--eval-orders", type=int, default=5, help="random orderings averaged for NLL")
    p.add_argument("--resume", action="store_true",
                   help="continue from ckpt.pt in --out (survives interruptions)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="results/task2")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    out_dir = ensure_dir(args.out)
    print(f"[task2] model={args.model} device={device} out={out_dir}")

    ds = MNISTBinarized(data_dir="data", mode=args.binarize, dynamic=True,
                        device=device, seed=args.seed, limit_train=args.train_size)

    # A *deterministic* binarization of the test set, used for the reported
    # likelihood so that the numbers are reproducible across runs.
    test_x_fixed = ds.fixed_test_set(seed=12345)
    n_test_eval = min(args.test_size, test_x_fixed.shape[0])
    test_x_fixed = test_x_fixed[:n_test_eval]

    if args.model == "nade":
        model = NADE(dim=784, hidden=args.hidden, n_hidden=args.n_hidden,
                     use_input_weights=not args.no_input_weights,
                     order_agnostic=not args.fixed_order).to(device)
    else:
        model = GRUAR(dim=784, hidden=args.hidden).to(device)
    print(f"[task2] params={count_params(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger = CSVLogger(os.path.join(out_dir, "log.csv"),
                       ["epoch", "train_nll", "test_nll", "test_nll_fixed",
                        "train_bits", "test_bits", "test_bits_fixed", "epoch_time_s"])

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    # sanity check: the model must be causal (no information flow from future pixels)
    if isinstance(model, NADE):
        x_chk = torch.rand(4, 784, device=device).round()
        ok = model.test_causality(x_chk)
        print(f"[task2] causality check (no future-pixel leakage): {'PASS' if ok else 'FAIL'}")
        assert ok, "NADE causality check failed"

    # ------------------------------------------------ optional resume
    start_epoch = 0
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = int(ck["epoch"]) + 1
        logger.keep_before("epoch", start_epoch)
        print(f"[task2] resumed from {ckpt_path} at epoch {start_epoch}")

    rows = []
    train_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
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
        train_nll = tot / nb

        model.eval()
        with torch.no_grad():
            vals = []
            for _ in range(5):
                x, _ = ds.get_batch(512, "test", generator=gen)
                vals.append(model.nll(x, n_orders=args.eval_orders).item())
            test_nll = sum(vals) / len(vals)

        # deterministic evaluation on the fixed test binarization with the
        # *natural* (raster) order -- this is the number reported in the paper
        with torch.no_grad():
            fixed_vals = []
            # NB: keep this batch small -- NADE materialises (B, 784, 784) masks
            for i in range(0, n_test_eval, 256):
                xb = test_x_fixed[i : i + 256]
                fixed_vals.append(model.nll(xb, n_orders=1, natural=True).item())
            test_nll_fixed = sum(fixed_vals) / len(fixed_vals)

        dt = time.perf_counter() - t0
        logger.append({
            "epoch": epoch,
            "train_nll": f"{train_nll:.2f}",
            "test_nll": f"{test_nll:.2f}",
            "test_nll_fixed": f"{test_nll_fixed:.2f}",
            "train_bits": f"{train_nll / (784 * LOG2):.4f}",
            "test_bits": f"{test_nll / (784 * LOG2):.4f}",
            "test_bits_fixed": f"{test_nll_fixed / (784 * LOG2):.4f}",
            "epoch_time_s": f"{dt:.2f}",
        })
        print(f"[task2] epoch {epoch:3d}  train NLL={train_nll:.2f}  test NLL={test_nll:.2f}  "
              f"fixed={test_nll_fixed:.2f} ({test_nll_fixed / (784 * LOG2):.4f} bits/dim)  {dt:.1f}s")

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch},
                   ckpt_path)

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                g = model.sample_grid(args.n_samples, generator=gen)
            rows.append(g.cpu())
            save_image_grid(g.cpu(), os.path.join(out_dir, f"samples_epoch{epoch:03d}.png"), nrow=8)

    total_train_time = time.perf_counter() - train_start

    model.eval()
    with torch.no_grad():
        final_grid = model.sample_grid(args.n_samples, generator=gen)
    save_image_grid(final_grid.cpu(), os.path.join(out_dir, "final_samples.png"), nrow=8)
    if rows:
        sample_grid_from_rows(rows, os.path.join(out_dir, "training_progress.png"), nrow=4)

    # ---------------------------------------------------------- inference time
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model.sample(args.n_samples, generator=gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0

    metrics = {
        "task": "2 - Neural autoregressive model",
        "model": ("NADE (order-agnostic MADE)" if not args.fixed_order
                  else "MADE (fixed raster order)") if args.model == "nade"
                 else "GRU autoregressive",
        "dataset": "binarized MNIST (random/dynamic thresholding)",
        "train_size": ds.n_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "n_hidden": args.n_hidden if args.model == "nade" else 1,
        "input_weights": (not args.no_input_weights) if args.model == "nade" else None,
        "params": count_params(model),
        "test_nll_nats_per_dim": test_nll_fixed / 784,
        "test_nll_bits_per_dim": test_nll_fixed / (784 * LOG2),
        "test_nll_bits_per_dim_dynamic_eval": test_nll / (784 * LOG2),
        "n_test_eval": n_test_eval,
        "training_time_s": total_train_time,
        "inference_time_batch64_s": gen_time,
        "device": str(device),
    }
    save_json(metrics, os.path.join(out_dir, "metrics.json"))
    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(out_dir, "model.pt"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
