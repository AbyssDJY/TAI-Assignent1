"""A small MNIST classifier used as an *evaluator* for the GAN (Task 3).

Why: the assignment asks for a report on training stability, specifically
whether **mode collapse** occurred.  "Mode collapse" is not directly observable
from the images alone, so we quantify it with an independent, supervised
reference:

    1. train a small CNN on the *real* MNIST training set,
    2. label a large batch of generated images with it,
    3. look at the histogram of predicted labels.

A healthy GAN produces a roughly uniform histogram over the 10 digits.  A
collapsed GAN concentrates almost all mass on one or two digits.  We report

    * ``num_modes_covered``  -- number of distinct digits with >= 1% of the mass,
    * ``mode_entropy``        -- entropy of the label histogram (max = log2 10),
    * ``mode_chi2``           -- chi-square distance to the uniform histogram.

This is a standard and reproducible proxy for mode collapse.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """Small convolutional classifier for 28x28 grayscale digits.

    28x28x1 -> conv(32) -> conv(64) -> maxpool -> 14x14 -> dense(128) -> 10
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 28x28x64
        self.pool = nn.MaxPool2d(2)                   # 14x14x64
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        h = self.drop1(h)
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        h = self.drop2(h)
        return self.fc2(h)


def train_classifier(
    data_dir: str = "data",
    epochs: int = 3,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
    lr: float = 1e-3,
    seed: int = 0,
    ckpt_path: Optional[str] = None,
) -> MnistCNN:
    """Train (or load) the evaluator classifier.  Returns a model in eval mode."""
    from data import load_mnist
    import numpy as np

    device = torch.device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    raw = load_mnist(data_dir)
    x = torch.from_numpy(raw["train_x"].astype(np.float32) / 255.0).unsqueeze(1)
    y = torch.from_numpy(raw["train_y"].astype(np.int64))

    if ckpt_path and os.path.exists(ckpt_path):
        model = MnistCNN().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        return model

    model = MnistCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, corr, lsum = 0, 0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x[idx].to(device), y[idx].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            lsum += loss.item() * xb.shape[0]
            corr += (out.argmax(1) == yb).sum().item()
            tot += xb.shape[0]
        print(f"[classifier] epoch {ep}: loss={lsum / tot:.4f} acc={corr / tot:.4f}")

    model.eval()
    if ckpt_path:
        torch.save(model.state_dict(), ckpt_path)
    return model


@torch.no_grad()
def label_histogram(
    classifier: MnistCNN,
    images: torch.Tensor,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
) -> torch.Tensor:
    """Count how many generated images the classifier assigns to each digit."""
    device = torch.device(device)
    counts = torch.zeros(10, dtype=torch.long)
    for i in range(0, images.shape[0], batch_size):
        xb = images[i : i + batch_size].to(device)
        if xb.min() < 0:
            xb = (xb + 1) / 2
        pred = classifier(xb).argmax(1).cpu()
        counts += torch.bincount(pred, minlength=10)
    return counts


def collapse_metrics(counts: torch.Tensor) -> dict:
    """Turn a label histogram into quantitative mode-collapse statistics."""
    import math

    counts = counts.float()
    total = counts.sum().item()
    p = counts / max(total, 1.0)
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum() / math.log(10))  # normalised to [0,1]
    modes = int((p >= 0.01).sum().item())
    chi2 = float((((p - 0.1) ** 2) / 0.1).sum().item() * total)
    top1 = float(p.max().item())
    return {
        "num_modes_covered": modes,
        "mode_entropy": round(entropy, 4),
        "mode_chi2_vs_uniform": round(chi2, 3),
        "top1_share": round(top1, 4),
        "histogram": [int(c) for c in counts.tolist()],
    }
