"""Evaluate the evaluator CNN on the real MNIST test set.

The accuracy of this classifier is the reference against which the GAN
mode-collapse statistics are interpreted, so it is reported in the paper.
The checkpoint is written by ``train_task3.py``.

Usage:  python src/eval_classifier.py --ckpt results/task3_dcgan/mnist_classifier.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from data import load_mnist
from evaluator import MnistCNN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_mnist("data")
    x = torch.from_numpy(raw["test_x"].astype(np.float32) / 255.0).unsqueeze(1).to(device)
    y = torch.from_numpy(raw["test_y"].astype(np.int64)).to(device)

    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    correct = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], 512):
            correct += (model(x[i:i + 512]).argmax(1) == y[i:i + 512]).sum().item()
    acc = correct / x.shape[0]
    print(f"[classifier] {args.ckpt}: test accuracy = {acc:.4f}")

    out = args.out or os.path.join(os.path.dirname(args.ckpt), "classifier_accuracy.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.ckpt, "test_accuracy": acc,
                   "n_test": int(x.shape[0])}, f, indent=2)


if __name__ == "__main__":
    main()
