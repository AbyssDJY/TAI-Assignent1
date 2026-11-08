"""MNIST loading + binarization utilities (no torchvision dataset dependency).

The raw IDX files are downloaded once from a public mirror (so the same code runs
unchanged on Colab / Kaggle / a local machine) and parsed here.

Binarization schemes implemented
-------------------------------
``static``  : x_i = 1 iff intensity_i > 0.5 * 255           (deterministic)
``mean``    : x_i = 1 iff intensity_i > mean(intensity)     (per-image threshold)
``random``  : x_i = 1 iff intensity_i > t,  t ~ U(0,1)     (per-image threshold,
              re-drawn every epoch -> the "dynamic binarization" of Larochelle &
              Murray 2011, which is the standard protocol for binarized MNIST)

**Important detail.**  The random threshold is drawn *once per image*, not per
pixel.  Using a per-pixel threshold would turn every pixel into an independent
Bernoulli variable and destroy all spatial correlation, which makes the
binarized data almost trivially easy and destroys the value of the task.
With a per-image threshold the gray levels of a single digit are thresholded
together, so the resulting binary image still carries the stroke structure.

Task 1 & 2 use binarized MNIST, Task 3 uses the original grayscale MNIST.
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np
import torch

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# --------------------------------------------------------------------------------------
# Raw IO
# --------------------------------------------------------------------------------------
def _download(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for fname in FILES.values():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"[data] downloading {fname} ...")
            urllib.request.urlretrieve(MIRROR + fname, path)


def _read_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "not an image file"
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)


def _read_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, "not a label file"
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8)


def load_mnist(data_dir: str = "data") -> Dict[str, np.ndarray]:
    """Return dict with train/test images (uint8 [N,28,28]) and labels (uint8 [N])."""
    _download(data_dir)
    return {
        "train_x": _read_images(os.path.join(data_dir, FILES["train_images"])),
        "train_y": _read_labels(os.path.join(data_dir, FILES["train_labels"])),
        "test_x": _read_images(os.path.join(data_dir, FILES["test_images"])),
        "test_y": _read_labels(os.path.join(data_dir, FILES["test_labels"])),
    }


# --------------------------------------------------------------------------------------
# Binarization (numpy reference implementation)
# --------------------------------------------------------------------------------------
def binarize(
    images: np.ndarray,
    mode: str = "random",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Binarize uint8 images [N,28,28] -> float32 {0,1} [N,784].

    ``mode='random'`` draws ONE threshold per image.
    """
    rng = rng or np.random.default_rng()
    x = images.astype(np.float32) / 255.0
    flat = x.reshape(x.shape[0], -1)
    if mode == "static":
        thr = np.full((flat.shape[0], 1), 0.5, dtype=np.float32)
    elif mode == "mean":
        thr = flat.mean(axis=1, keepdims=True)
    elif mode == "random":
        thr = rng.random((flat.shape[0], 1)).astype(np.float32)
    else:
        raise ValueError(f"unknown binarization mode: {mode}")
    return (flat > thr).astype(np.float32)


class MNISTBinarized:
    """Binarized MNIST held entirely on the GPU for fast training.

    The *intensities* are stored; binarization happens on the fly so that the
    dynamic (per-epoch re-randomised) threshold is cheap.

    Parameters
    ----------
    mode   : 'static' | 'mean' | 'random'
    dynamic: if True (and mode == 'random') a fresh random threshold is drawn
             every time ``get_batch`` is called (data augmentation).
    """

    def __init__(
        self,
        data_dir: str = "data",
        mode: str = "random",
        dynamic: bool = True,
        device: torch.device | str = "cpu",
        seed: int = 0,
        limit_train: int | None = None,
    ):
        raw = load_mnist(data_dir)
        self.mode = mode
        self.dynamic = dynamic and mode == "random"
        self.device = torch.device(device)

        tr = raw["train_x"]
        if limit_train is not None:
            tr = tr[:limit_train]
        te = raw["test_x"]

        self.train_int = torch.from_numpy(
            tr.astype(np.float32) / 255.0
        ).reshape(len(tr), -1).to(self.device)
        self.test_int = torch.from_numpy(
            te.astype(np.float32) / 255.0
        ).reshape(len(te), -1).to(self.device)

        self.train_y = torch.from_numpy(raw["train_y"][: len(tr)].astype(np.int64)).to(self.device)
        self.test_y = torch.from_numpy(raw["test_y"].astype(np.int64)).to(self.device)
        self.n_train = len(tr)
        self.n_test = len(te)

    # -- core ------------------------------------------------------------------------
    def _threshold(self, intensities: torch.Tensor, mode: str,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Per-image threshold of shape (B, 1)."""
        if mode == "static":
            return torch.full((intensities.shape[0], 1), 0.5, device=intensities.device)
        if mode == "mean":
            return intensities.mean(dim=1, keepdim=True)
        if mode == "random":
            return torch.rand(intensities.shape[0], 1, device=intensities.device,
                              generator=generator)
        raise ValueError(mode)

    def _binarize(self, intensities: torch.Tensor, mode: str,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        thr = self._threshold(intensities, mode, generator)
        return (intensities > thr).float()

    # -- accessors -------------------------------------------------------------------
    def get_batch(self, batch_size: int, split: str = "train",
                  generator: Optional[torch.Generator] = None):
        """Return (x [B,784] float, y [B] long)."""
        if split == "train":
            n = self.n_train
            idx = torch.randint(0, n, (batch_size,), device=self.device, generator=generator)
            inten = self.train_int[idx]
            mode = self.mode if self.dynamic else self.mode
            x = self._binarize(inten, mode, generator)
            return x, self.train_y[idx]
        else:
            n = self.n_test
            idx = torch.randint(0, n, (batch_size,), device=self.device, generator=generator)
            inten = self.test_int[idx]
            x = self._binarize(inten, self.mode, generator)
            return x, self.test_y[idx]

    def fixed_test_set(self, mode: Optional[str] = None, seed: int = 12345) -> torch.Tensor:
        """A **deterministic** binarization of the test set.

        Used for likelihood evaluation: the model is trained on dynamically
        re-thresholded images (data augmentation), but the reported test NLL
        must be reproducible, so we materialise one fixed random-threshold
        draw of the test set with a dedicated seed.
        """
        mode = mode or self.mode
        gen = torch.Generator(device=self.device).manual_seed(seed)
        return self._binarize(self.test_int, mode, gen)

    def fixed_train_set(self, mode: Optional[str] = None, seed: int = 12345) -> torch.Tensor:
        """A deterministic binarization of the *training* set (same purpose)."""
        mode = mode or self.mode
        gen = torch.Generator(device=self.device).manual_seed(seed)
        return self._binarize(self.train_int, mode, gen)

    def real_grid(self, n: int = 64, mode: Optional[str] = None) -> torch.Tensor:
        """A batch of real (binarized) images reshaped to [n,1,28,28] for plotting."""
        mode = mode or self.mode
        x = self._binarize(self.train_int[:n], mode)
        return x.view(-1, 1, 28, 28)
