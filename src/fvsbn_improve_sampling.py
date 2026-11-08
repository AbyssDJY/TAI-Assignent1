"""Improving the FVSBN sampler: annealing and mode selection.

The Gibbs chain of a pseudo-likelihood-trained fully visible model settles
(the flip rate drops to ~3%) but it settles into a *noisy local mode* rather
than into a digit.  This is expected: the conditionals were fitted on real
images only, so configurations the chain visits during sampling are far outside
the training distribution and the chain is easily trapped.

Two standard remedies are compared here:

``anneal``   start the chain at a high temperature and cool it down to T=1.
             High temperature flattens the landscape, which lets the chain
             move between modes before it commits to one.
``best-of-K`` draw K independent chains and keep the sample with the highest
             *pseudo-log-likelihood* under the model, i.e. the most
             model-typical configuration found.  This is a cheap form of mode
             selection and it is what we use for the reported samples.

Usage:
    python src/fvsbn_improve_sampling.py --ckpt results/task1/fvsbn.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from common import ensure_dir, save_image_grid, set_seed
from model_fvsbn import FVSBN


@torch.no_grad()
def annealed_gibbs(model: FVSBN, n: int, n_sweeps: int, block_size: int,
                   t_start: float, t_end: float, device, generator,
                   init: str = "random"):
    if init == "random":
        x = (torch.rand(n, model.dim, device=device, generator=generator) < 0.5).float()
    elif init == "sparse":
        x = (torch.rand(n, model.dim, device=device, generator=generator) < 0.1).float()
    else:
        raise ValueError(init)

    W = model.effective_W()
    for sweep in range(n_sweeps):
        frac = sweep / max(n_sweeps - 1, 1)
        temp = t_start * (t_end / t_start) ** frac if t_start > 0 else 1.0
        perm = torch.randperm(model.dim, device=device, generator=generator)
        for start in range(0, model.dim, block_size):
            idx = perm[start:start + block_size]
            z = x @ W[idx].t() + model.b[idx]
            p = torch.sigmoid(z / temp)
            u = torch.rand(n, idx.numel(), device=device, generator=generator)
            x[:, idx] = (u < p).float()
    return x


@torch.no_grad()
def pseudo_loglik(model: FVSBN, x: torch.Tensor) -> torch.Tensor:
    """Sum of log p(x_i | x_-i) for every sample (higher = more model-typical)."""
    z = model.logits(x)
    return -torch.nn.functional.binary_cross_entropy_with_logits(
        z, x, reduction="none").sum(dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="results/task1/fvsbn.pt")
    ap.add_argument("--out", type=str, default="results/task1/sampling_improved")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--sweeps", type=int, default=400)
    ap.add_argument("--t-start", type=float, default=2.5)
    ap.add_argument("--n-chains", type=int, default=16, help="best-of-K chains")
    args = ap.parse_args()

    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    a = ck["args"]
    model = FVSBN(dim=784, rank=a["rank"] if a.get("rank", 0) > 0 else None).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    out_dir = ensure_dir(args.out)
    gen = torch.Generator(device=device)
    gen.manual_seed(12345)

    report = {}
    for tag, t0, t1, init in [("plain", 1.0, 1.0, "random"),
                              ("anneal", args.t_start, 1.0, "random"),
                              ("anneal_sparse", args.t_start, 1.0, "sparse")]:
        t = time.perf_counter()
        x = annealed_gibbs(model, args.n, args.sweeps, args.block, t0, t1,
                           device, gen, init=init)
        dt = time.perf_counter() - t
        pll = pseudo_loglik(model, x)
        save_image_grid(x.view(-1, 1, 28, 28), os.path.join(out_dir, f"{tag}.png"), nrow=8)
        report[tag] = {"mean_pseudo_loglik": round(pll.mean().item(), 2),
                       "foreground_rate": round(x.mean().item(), 4),
                       "time_s": round(dt, 1)}
        print(f"[improve] {tag:14s} pll={pll.mean().item():8.2f}  "
              f"fg={x.mean().item():.3f}  {dt:5.1f}s")

        # ---- best-of-K: draw K*n chains, keep the top-n by pseudo-log-likelihood
        t = time.perf_counter()
        xs = annealed_gibbs(model, args.n * args.n_chains, args.sweeps, args.block,
                            t0, t1, device, gen, init=init)
        pll_all = pseudo_loglik(model, xs)
        order = torch.argsort(pll_all, descending=True)[: args.n]
        xb = xs[order].clone()
        dt2 = time.perf_counter() - t
        save_image_grid(xb.view(-1, 1, 28, 28),
                        os.path.join(out_dir, f"{tag}_bestof{args.n_chains}.png"), nrow=8)
        report[f"{tag}_bestof{args.n_chains}"] = {
            "mean_pseudo_loglik": round(pseudo_loglik(model, xb).mean().item(), 2),
            "foreground_rate": round(xb.mean().item(), 4),
            "time_s": round(dt2, 1),
            "n_chains_per_kept": args.n_chains,
        }
        print(f"[improve] {tag+'_bestof':14s} pll={report[f'{tag}_bestof{args.n_chains}']['mean_pseudo_loglik']:8.2f}"
              f"  fg={xb.mean().item():.3f}  {dt2:5.1f}s")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[improve] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
