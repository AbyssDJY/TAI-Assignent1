"""Reference baselines for binarized MNIST likelihood.

Two trivial baselines are computed on exactly the same deterministic test
binarisation that is used to evaluate the autoregressive models:

``constant``          every pixel is predicted with the global mean frequency
                      (the entropy of the data, i.e. no model at all);
``independent-pixels`` every pixel gets its *own* marginal frequency estimated
                      on the training set.  This is the real competitor: it
                      knows the per-pixel bias of MNIST (borders are empty) but
                      it assumes the pixels are conditionally independent, so
                      it cannot use any spatial structure.

A good autoregressive model must beat ``independent-pixels`` -- that is the
evidence that it actually exploits the correlation between neighbouring
pixels.

Usage:  python src/baselines.py --out results/baselines.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from common import ensure_dir, save_json, set_seed
from data import MNISTBinarized

LOG2 = math.log(2)


def bernoulli_nll(probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """NLL in nats of Bernoulli(x; probs), summed over the 784 pixels."""
    p = probs.clamp(1e-7, 1 - 1e-7)
    return -(x * torch.log(p) + (1 - x) * torch.log(1 - p)).sum(dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binarize", type=str, default="random", choices=["random", "static", "mean"])
    ap.add_argument("--train-size", type=int, default=60000)
    ap.add_argument("--test-size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/baselines.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MNISTBinarized(data_dir="data", mode=args.binarize, dynamic=True,
                        device=device, seed=args.seed, limit_train=args.train_size)

    train = ds.fixed_train_set(seed=12345)
    test = ds.fixed_test_set(seed=12345)[: args.test_size]

    # ---- baseline 1: constant predictor (global mean) -----------------------------
    g = float(train.mean())
    nll_const = bernoulli_nll(torch.full_like(test, g), test).mean().item()

    # ---- baseline 2: independent pixels (per-position marginal) --------------------
    p_pos = train.mean(dim=0)                                     # (784,)
    nll_indep = bernoulli_nll(p_pos.unsqueeze(0).expand_as(test), test).mean().item()

    res = {
        "dataset": f"binarized MNIST ({args.binarize} threshold), "
                   f"train={train.shape[0]}, test={test.shape[0]}",
        "seed": 12345,
        "global_mean_foreground": g,
        "constant_bits_per_dim": nll_const / (784 * LOG2),
        "independent_pixels_bits_per_dim": nll_indep / (784 * LOG2),
        "constant_nats_per_dim": nll_const / 784,
        "independent_pixels_nats_per_dim": nll_indep / 784,
        "note": "the autoregressive models must beat 'independent_pixels' to "
                "show that they exploit spatial structure",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_json(res, args.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
