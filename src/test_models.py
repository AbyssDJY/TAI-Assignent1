"""Causality tests for every autoregressive model in the project.

A masked autoregressive network can be *silently* wrong: a transposed mask
still trains, still gives a finite loss and still produces plausible-looking
samples, but the model is then reading the future and its likelihood is
meaningless.  This module checks, with automatic differentiation, that the
output for pixel ``k`` has exactly zero gradient with respect to every pixel
that comes *after* ``k`` in the model's own ordering.

It is used by the training scripts (an assertion) and can be run standalone:

    python src/test_models.py

Both checks earned their keep during development: the ordered-FVSBN mask was
transposed at first, which inflated its apparent likelihood.
"""
from __future__ import annotations

import sys

import torch

from model_fvsbn import FVSBNOrdered
from model_nade import GRUAR, NADE


# --------------------------------------------------------------------------------------
def check_ordered_fvsbn(dim: int = 784, n_probe: int = 12, seed: int = 0) -> bool:
    torch.manual_seed(seed)
    m = FVSBNOrdered(dim=dim)
    # make the weights non-trivial, otherwise every gradient is zero anyway
    with torch.no_grad():
        m.W.normal_(0, 0.1)
    x = torch.randint(0, 2, (4, dim)).float().requires_grad_(True)
    logits = m.logits(x)

    g = torch.Generator().manual_seed(seed)
    for i in torch.randperm(dim, generator=g)[:n_probe].tolist():
        grad = torch.autograd.grad(logits[:, i].sum(), x, retain_graph=True)[0]
        # pixel i itself and everything after it must not influence logits[:, i]
        if grad[:, i:].abs().max().item() > 1e-6:
            print(f"  [FVSBNOrdered] LEAK at pixel {i}")
            return False
    return True


def check_nade(dim: int = 784, n_probe: int = 12, seed: int = 0) -> bool:
    torch.manual_seed(seed)
    m = NADE(dim=dim, hidden=64, n_hidden=2)
    with torch.no_grad():
        m.W1.normal_(0, 0.1)
        m.W3.normal_(0, 0.1)
        m.V.normal_(0, 0.1)
    x = torch.randint(0, 2, (4, dim)).float()
    return m.test_causality(x, n_probe=n_probe, seed=seed)


def check_gru(dim: int = 784, n_probe: int = 8, seed: int = 0) -> bool:
    torch.manual_seed(seed)
    m = GRUAR(dim=dim, hidden=32)
    x = torch.randint(0, 2, (2, dim)).float().requires_grad_(True)
    logits = m(x)
    g = torch.Generator().manual_seed(seed)
    for t in torch.randperm(dim, generator=g)[:n_probe].tolist():
        grad = torch.autograd.grad(logits[:, t].sum(), x, retain_graph=True)[0]
        # logits[:, t] may depend on x[:, <t] only
        if grad[:, t:].abs().max().item() > 1e-6:
            print(f"  [GRUAR] LEAK at step {t}")
            return False
    return True


def main() -> None:
    checks = [("FVSBNOrdered", check_ordered_fvsbn),
              ("NADE", check_nade),
              ("GRUAR", check_gru)]
    ok = True
    for name, fn in checks:
        good = fn()
        ok &= good
        print(f"[causality] {name:14s} {'PASS' if good else 'FAIL'}")
    print("\nALL CAUSALITY CHECKS PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
