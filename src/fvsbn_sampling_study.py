"""Study of the Gibbs sampler of the FVSBN (Task 1).

Why this script exists
----------------------
A fully visible model trained with the pseudo-likelihood is *specified by a set
of conditionals*, not by a joint distribution.  If those conditionals are not
mutually consistent, the Gibbs chain has no stationary distribution: it keeps
flipping pixels and the "samples" look like noise even though the
pseudo-likelihood is very low.  Two things therefore have to be measured, not
assumed:

1. **Does the chain settle?**  We run extra sweeps after the burn-in and count
   the fraction of pixels that change value per sweep (the *flip rate*).  A
   healthy chain settles to a small flip rate; a frustrated one keeps flipping.
2. **Does the block size matter?**  Updating ``B`` pixels at once from the same
   stale state is an approximation: the larger ``B``, the less the chain
   resembles true Gibbs.

Results are written to ``<out>/`` together with a flip-rate curve and one
sample grid per configuration.

Usage:
    python src/fvsbn_sampling_study.py --ckpt results/task1/fvsbn.pt
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
def gibbs_with_trace(model: FVSBN, n: int, n_sweeps: int, block_size: int,
                     temperature: float, device, generator):
    """Run Gibbs and return (samples, flips_per_sweep, wall_time)."""
    x = (torch.rand(n, model.dim, device=device, generator=generator) < 0.5).float()
    W = model.effective_W()
    flips = []
    t0 = time.perf_counter()
    for _ in range(n_sweeps):
        perm = torch.randperm(model.dim, device=device, generator=generator)
        n_flip = 0
        for start in range(0, model.dim, block_size):
            idx = perm[start:start + block_size]
            z = x @ W[idx].t() + model.b[idx]
            p = torch.sigmoid(z / temperature)
            u = torch.rand(n, idx.numel(), device=device, generator=generator)
            new = (u < p).float()
            n_flip += (new != x[:, idx]).sum().item()
            x[:, idx] = new
        flips.append(n_flip / (n * model.dim))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return x, flips, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="results/task1/fvsbn.pt")
    ap.add_argument("--out", type=str, default="results/task1/sampling_study")
    ap.add_argument("--n", type=int, default=64)
    # block:sweeps:temperature
    ap.add_argument("--configs", type=str,
                    default="64:300:1.0,16:300:1.0,4:300:1.0,1:300:1.0,1:600:1.0,4:300:0.7")
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

    summary, traces = [], {}
    for spec in args.configs.split(","):
        block, sweeps, temp = spec.split(":")
        block, sweeps, temp = int(block), int(sweeps), float(temp)
        x, flips, dt = gibbs_with_trace(model, args.n, sweeps, block, temp, device, gen)
        name = f"b{block}_s{sweeps}_t{temp:.1f}.png"
        save_image_grid(x.view(-1, 1, 28, 28), os.path.join(out_dir, name), nrow=8)
        # flip rate over the last 10% of the chain = "did it settle?"
        tail = flips[int(0.9 * len(flips)):]
        rec = {"block_size": block, "sweeps": sweeps, "temperature": temp,
               "final_flip_rate": round(tail[-1], 5),
               "mean_flip_rate_last10pct": round(sum(tail) / len(tail), 5),
               "foreground_rate": round(x.mean().item(), 4),
               "time_batch_s": round(dt, 2),
               "time_per_sample_s": round(dt / args.n, 4),
               "active_pixels": int(x.sum().item())}
        summary.append(rec)
        traces[f"b{block}_s{sweeps}_t{temp:.1f}"] = [round(f, 5) for f in flips]
        print(f"[study] block={block:3d} sweeps={sweeps:4d} T={temp:.1f}  "
              f"flip={rec['mean_flip_rate_last10pct']:.4f}  fg={rec['foreground_rate']:.3f}  "
              f"{dt:6.1f}s")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"configs": summary, "flip_traces": traces}, f, indent=2)
    print(f"[study] wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
