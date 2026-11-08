"""Training / evaluation script for Task 3: GAN on the *original* MNIST.

Supports two objectives:
    --objective dcgan   original DCGAN (non-saturating logistic loss + label smoothing)
    --objective wgan    WGAN-GP (Wasserstein loss with gradient penalty)

The report needs a quantitative statement about training stability / mode
collapse, so every ``--eval-every`` epochs we:
    * save a sample grid (progress showcase),
    * label 2000 generated images with an independently trained MNIST
      classifier and record the digit histogram + collapse metrics.

Usage
-----
    python train_task3.py --objective dcgan --epochs 50
    python train_task3.py --objective wgan  --epochs 50
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from common import CSVLogger, count_params, ensure_dir, sample_grid_from_rows, save_image_grid, save_json, set_seed
from data import load_mnist
from evaluator import collapse_metrics, label_histogram, train_classifier
from model_gan import Discriminator, Generator, d_loss_dcgan, g_loss_dcgan, wgan_gp_loss


def parse_args():
    p = argparse.ArgumentParser(description="Task 3 - GAN on original MNIST")
    p.add_argument("--objective", type=str, default="dcgan", choices=["dcgan", "wgan"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.5, 0.999))
    p.add_argument("--n-critic", type=int, default=5, help="critic iterations per generator step (WGAN)")
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--train-size", type=int, default=60000,
                   help="size of the MNIST training subset used for the GAN")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-n", type=int, default=2000)
    p.add_argument("--resume", action="store_true",
                   help="continue from ckpt.pt in --out (survives interruptions)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="results/task3")
    p.add_argument("--skip-classifier", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    out_dir = ensure_dir(args.out)
    print(f"[task3] objective={args.objective} device={device} out={out_dir}")

    # ------------------------------------------------------------ real data
    raw = load_mnist("data")
    tr = raw["train_x"][: args.train_size].astype(np.float32) / 255.0  # [N,28,28] in [0,1]
    real_all = torch.from_numpy(tr).unsqueeze(1).to(device) * 2 - 1  # -> [-1,1]
    n_real = real_all.shape[0]
    print(f"[task3] training images: {n_real}")

    # ---------------------------------------------------------- models
    G = Generator(latent_dim=args.latent_dim).to(device)
    D = Discriminator().to(device)
    print(f"[task3] G params={count_params(G):,}  D params={count_params(D):,}")

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=tuple(args.betas))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=tuple(args.betas))

    # ------------------------------------------- evaluator classifier (mode collapse)
    clf = None
    ckpt = os.path.join(out_dir, "mnist_classifier.pt")
    if not args.skip_classifier:
        clf = train_classifier("data", epochs=3, device=device, seed=args.seed, ckpt_path=ckpt)

    logger = CSVLogger(
        os.path.join(out_dir, "log.csv"),
        ["epoch", "d_loss", "g_loss", "gp", "modes_covered", "mode_entropy", "top1_share", "epoch_time_s"],
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    # ------------------------------------------------ optional resume
    start_epoch = 0
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ck["G"])
        D.load_state_dict(ck["D"])
        optG.load_state_dict(ck["optG"])
        optD.load_state_dict(ck["optD"])
        start_epoch = int(ck["epoch"]) + 1
        logger.keep_before("epoch", start_epoch)
        print(f"[task3] resumed from {ckpt_path} at epoch {start_epoch}")

    rows = []
    best_entropy = -1.0
    train_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        perm = torch.randperm(n_real, device=device)
        d_tot = g_tot = gp_tot = 0.0
        nb = 0

        for i in range(0, n_real, args.batch_size):
            idx = perm[i : i + args.batch_size]
            real = real_all[idx]
            if real.shape[0] < 2:
                continue
            b = real.shape[0]

            # ------------------------------------------------------ critic step
            if args.objective == "dcgan":
                # one critic update and one generator update per batch
                z = torch.randn(b, args.latent_dim, device=device, generator=gen)
                fake = G(z)
                d_real, d_fake = D(real), D(fake.detach())
                loss_d = d_loss_dcgan(d_real, d_fake)
                optD.zero_grad(set_to_none=True)
                loss_d.backward()
                optD.step()
                d_tot += loss_d.item()

                # generator step (non-saturating logistic loss)
                z = torch.randn(b, args.latent_dim, device=device, generator=gen)
                fake = G(z)
                loss_g = g_loss_dcgan(D(fake))
                optG.zero_grad(set_to_none=True)
                loss_g.backward()
                optG.step()
                g_tot += loss_g.item()
                nb += 1
            else:  # WGAN-GP
                for _ in range(args.n_critic):
                    z = torch.randn(b, args.latent_dim, device=device, generator=gen)
                    fake = G(z).detach()
                    loss_d, _, gp = wgan_gp_loss(D, real, fake, z, args.lambda_gp, device)
                    optD.zero_grad(set_to_none=True)
                    loss_d.backward()
                    optD.step()
                    d_tot += loss_d.item()
                    gp_tot += gp.item()
                z = torch.randn(b, args.latent_dim, device=device, generator=gen)
                fake = G(z)
                loss_g = -D(fake).mean()
                optG.zero_grad(set_to_none=True)
                loss_g.backward()
                optG.step()
                g_tot += loss_g.item()
                nb += 1

        d_avg, g_avg, gp_avg = d_tot / max(nb, 1), g_tot / max(nb, 1), gp_tot / max(nb, 1)
        dt = time.perf_counter() - t0

        # ------------------------------------------------------ evaluation
        G.eval()
        with torch.no_grad():
            g = G(torch.randn(args.n_samples, args.latent_dim, device=device, generator=gen))
            g01 = (g + 1) / 2
            rows.append(g01.cpu())
            save_image_grid(g01.cpu(), os.path.join(out_dir, f"samples_epoch{epoch:03d}.png"), nrow=8)
            torch.save(g01.cpu(), os.path.join(out_dir, f"samples_epoch{epoch:03d}.pt"))

            if clf is not None and (epoch % args.eval_every == 0 or epoch == args.epochs - 1):
                big = G(torch.randn(args.eval_n, args.latent_dim, device=device, generator=gen))
                counts = label_histogram(clf, big, device=device)
                cm = collapse_metrics(counts)
            else:
                cm = {"num_modes_covered": "", "mode_entropy": "", "mode_chi2_vs_uniform": "",
                      "top1_share": "", "histogram": ""}
        G.train()

        logger.append({
            "epoch": epoch,
            "d_loss": f"{d_avg:.4f}",
            "g_loss": f"{g_avg:.4f}",
            "gp": f"{gp_avg:.4f}",
            "modes_covered": cm["num_modes_covered"],
            "mode_entropy": cm["mode_entropy"],
            "top1_share": cm["top1_share"],
            "epoch_time_s": f"{dt:.2f}",
        })
        torch.save({"G": G.state_dict(), "D": D.state_dict(),
                    "optG": optG.state_dict(), "optD": optD.state_dict(),
                    "epoch": epoch}, ckpt_path)

        extra = ""
        if cm["num_modes_covered"] != "":
            extra = (f"  modes={cm['num_modes_covered']}/10  entropy={cm['mode_entropy']:.3f}"
                     f"  top1={cm['top1_share']:.3f}")
            if cm["mode_entropy"] > best_entropy:
                best_entropy = cm["mode_entropy"]
        print(f"[task3] epoch {epoch:3d}  D={d_avg:.4f}  G={g_avg:.4f}  gp={gp_avg:.3f}  {dt:.1f}s{extra}")

    total_train_time = time.perf_counter() - train_start

    # ---------------------------------------------------------- final results
    G.eval()
    with torch.no_grad():
        final = (G(torch.randn(args.n_samples, args.latent_dim, device=device, generator=gen)) + 1) / 2
        big = G(torch.randn(10000, args.latent_dim, device=device, generator=gen))
    save_image_grid(final.cpu(), os.path.join(out_dir, "final_samples.png"), nrow=8)
    if rows:
        sample_grid_from_rows(rows, os.path.join(out_dir, "training_progress.png"), nrow=4)

    final_metrics = {}
    if clf is not None:
        counts = label_histogram(clf, big, device=device)
        final_metrics = collapse_metrics(counts)
        print(f"[task3] final collapse metrics: {final_metrics}")

    # ---------------------------------------------------------- inference time
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = G(torch.randn(args.n_samples, args.latent_dim, device=device, generator=gen))
        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0

    metrics = {
        "task": "3 - GAN on original MNIST",
        "objective": args.objective,
        "architecture": "DCGAN-style conv generator/discriminator (28x28x1)",
        "dataset": "original (grayscale) MNIST, no binarisation",
        "train_size": n_real,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "lr": args.lr,
        "n_critic": args.n_critic if args.objective == "wgan" else 1,
        "G_params": count_params(G),
        "D_params": count_params(D),
        "training_time_s": total_train_time,
        "inference_time_batch64_s": gen_time,
        "collapse_metrics_final": final_metrics,
        "device": str(device),
    }
    save_json(metrics, os.path.join(out_dir, "metrics.json"))
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "args": vars(args)},
               os.path.join(out_dir, "gan.pt"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
