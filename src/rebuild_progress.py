"""Rebuild the "training progress" showcase grid (5 rows x 4 columns) from the
per-stage sample grids that the training scripts wrote.

The assignment asks for a generation-progress figure in which *each row is one
training stage*; the canonical layout is 5 stages x 4 samples.  The per-stage
files (``samples_epochXXX.png``) are 8x8 grids written by ``torchvision.utils.
save_image`` with ``nrow=8`` and ``padding=2``, so a cell of 28x28 pixels starts
at ``(2 + i*30, 2 + j*30)``.  This script crops the first ``cols`` cells of the
first row of every stage and assembles them into one 5x4 figure.

Usage:  python src/rebuild_progress.py results/task2
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from PIL import Image

CELL = 28
PAD = 2
STRIDE = CELL + PAD


def rebuild(folder: str, cols: int = 4, rows: int = 5) -> str:
    files = sorted(glob.glob(os.path.join(folder, "samples_epoch*.png")))
    if not files:
        raise SystemExit(f"no sample grids found in {folder}")
    if len(files) > rows:
        # pick `rows` stages evenly spaced over the whole run, always including
        # the first and the last one
        idx = [round(i * (len(files) - 1) / (rows - 1)) for i in range(rows)]
        stages = [files[i] for i in sorted(set(idx))]
    else:
        stages = files

    first = np.array(Image.open(stages[0]).convert("L"))
    H, W = first.shape
    # sanity: the grid must be an 8-column layout
    assert (W - PAD) % STRIDE == 0, f"unexpected grid width {W}"

    out = np.zeros((rows * STRIDE + PAD, cols * STRIDE + PAD), dtype=np.uint8)
    for r, f in enumerate(stages):
        img = np.array(Image.open(f).convert("L"))
        for c in range(cols):
            y0 = PAD + r * STRIDE
            x0 = PAD + c * STRIDE
            out[y0:y0 + CELL, x0:x0 + CELL] = img[PAD:PAD + CELL, PAD:PAD + CELL]
    # the source grids were 8 wide -> we cropped only the first 4 columns
    dst = os.path.join(folder, "training_progress.png")
    Image.fromarray(out, mode="L").convert("RGB").save(dst)
    print(f"[progress] {dst}  ({len(stages)} stages x {cols} samples)")
    return dst


if __name__ == "__main__":
    folders = sys.argv[1:] or ["results/task1", "results/task2", "results/task3_dcgan"]
    for f in folders:
        if os.path.isdir(f):
            try:
                rebuild(f)
            except SystemExit as e:
                print(f"[progress] skip {f}: {e}")
