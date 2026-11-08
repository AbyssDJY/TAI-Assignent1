"""Shared utilities for CS5494 Assignment 1.

Contains: seeding, device selection, CSV/JSON logging, image-grid helpers and
small timing utilities used by every experiment script.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
def set_seed(seed: int = 0) -> None:
    """Seed python / numpy / torch RNGs (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    """Resolve a torch device. ``pref='auto'`` prefers CUDA when available."""
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


class CSVLogger:
    """Append-only CSV writer used for training curves (loss / NLL per epoch)."""

    def __init__(self, path: str, fieldnames: Iterable[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(",".join(self.fieldnames) + "\n")

    def append(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(",".join(str(row.get(k, "")) for k in self.fieldnames) + "\n")

    def keep_before(self, column: str, value: int) -> None:
        """Drop every row whose ``column`` is >= ``value``.

        Used when resuming from a checkpoint: the epochs that will be re-run
        must not appear twice in the log.
        """
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        if not lines:
            return
        header, rows = lines[0], lines[1:]
        kept = []
        for r in rows:
            parts = r.split(",")
            try:
                ep = int(parts[self.fieldnames.index(column)])
            except (ValueError, IndexError):
                continue
            if ep < value:
                kept.append(r)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join([header] + kept) + "\n")


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# --------------------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------------------
def to_uint8_grid(images: torch.Tensor) -> np.ndarray:
    """Convert a batch of images in [0,1] (N,C,H,W) to a uint8 numpy grid."""
    x = images.detach().cpu().clamp(0, 1).numpy()
    if x.shape[1] == 1:
        x = np.repeat(x, 3, axis=1)
    x = (x * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    return x


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
) -> None:
    """Save a grid of images to ``path`` (PNG)."""
    from torchvision.utils import save_image

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    x = images.detach().float().cpu()
    if x.min() < 0:  # assume [-1, 1] tanh output
        x = (x + 1) / 2
    save_image(x, path, nrow=nrow, padding=padding, normalize=normalize)


def sample_grid_from_rows(
    rows: List[torch.Tensor], path: str, nrow: int = 4, padding: int = 2
) -> None:
    """Stack per-training-stage image batches vertically into one grid PNG.

    ``rows[i]`` is a (n, C, H, W) tensor of samples produced at training stage ``i``;
    every row is laid out with ``nrow`` columns so the figure reads as
    "rows = training progress, columns = samples".
    """
    assert len(rows) > 0, "no rows to save"
    n_cols = min(nrow, min(r.shape[0] for r in rows))
    strips = []
    for r in rows:
        r = r[:n_cols]
        if r.shape[1] == 1:
            r = r.repeat(1, 3, 1, 1)
        strips.append(r)
    grid = torch.cat(strips, dim=0)
    save_image_grid(grid, path, nrow=n_cols, padding=padding)


# --------------------------------------------------------------------------------------
# Parameter counting
# --------------------------------------------------------------------------------------
def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
