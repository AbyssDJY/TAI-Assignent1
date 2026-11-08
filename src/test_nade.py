"""Correctness tests for the NADE implementation.

Two independent references are used:

1. ``brute_forward`` -- the literal definition of the NADE conditionals,
   computed with explicit python loops (no masks, no vectorisation).
2. ``brute_sample``  -- the literal definition of ancestral sampling, again
   with explicit loops.

The vectorised implementation in ``model_nade.py`` must agree with both, and
the causality check must pass.
"""
from __future__ import annotations

import torch

from model_nade import NADE


def _brute_hidden(model: NADE, x_row, rank, upto_rank):
    """Explicit hidden layer stack for one example.

    Unit j of layer l sees the pixels whose rank is <= cap_l(j); layer l+1
    receives unit j1 only if cap_l(j1) <= cap_{l+1}(j2).
    """
    h = []
    for j in range(model.hidden):
        s = model.b1[j].item()
        for q in range(model.dim):
            if rank[q] <= model.caps[j].item():
                s += model.W1[q, j].item() * x_row[q].item()
        h.append(torch.tanh(torch.tensor(s)))
    for l in range(1, model.n_hidden):
        h2 = []
        for j2 in range(model.hidden):
            s = model.bh[l - 1][j2].item()
            for j1 in range(model.hidden):
                if model.caps[j1].item() <= model.caps[j2].item():
                    s += model.Wh[l - 1][j2, j1].item() * h[j1].item()
            h2.append(torch.tanh(torch.tensor(s)))
        h = h2
    return h


def brute_forward(model: NADE, x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Explicit per-rank reference implementation (slow but unambiguous)."""
    b, d = x.shape
    caps = model.caps
    out = torch.zeros(b, d)
    for bi in range(b):
        pi = perm[bi].tolist()                       # pi[i] = spatial pixel at rank i
        rank = [0] * d
        for i, k in enumerate(pi):
            rank[k] = i
        for i in range(d):
            k = pi[i]                                # spatial pixel we predict
            h = _brute_hidden(model, x[bi], rank, i)
            s = model.b3[k].item()
            for j in range(model.hidden):
                if caps[j].item() < i:
                    s += model.W3[k, j].item() * h[j].item()
            if model.V is not None:
                for q in range(d):
                    if rank[q] < i:
                        s += model.V[k, q].item() * x[bi, q].item()
            out[bi, k] = s
    return out


def brute_sample(model: NADE, n: int, gen: torch.Generator, device) -> torch.Tensor:
    """Explicit ancestral sampling reference.

    The uniform deviates are drawn *per step for all examples at once* so that
    the RNG stream matches the vectorised implementation exactly.
    """
    d = model.dim
    perm = torch.argsort(torch.rand(n, d, device=device, generator=gen), dim=1)
    x = torch.zeros(n, d, device=device)
    rank = torch.empty(n, d, dtype=torch.long)
    for bi in range(n):
        pi = perm[bi].tolist()
        for i, k in enumerate(pi):
            rank[bi, k] = i
    for i in range(d):
        logits = torch.zeros(n)
        for bi in range(n):
            k = int(perm[bi, i].item())
            h = _brute_hidden(model, x[bi], rank[bi].tolist(), i)
            lg = model.b3[k].item()
            for j in range(model.hidden):
                if model.caps[j].item() < i:
                    lg += model.W3[k, j].item() * h[j].item()
            if model.V is not None:
                for q in range(d):
                    if rank[bi, q].item() < i:
                        lg += model.V[k, q].item() * x[bi, q].item()
            logits[bi] = lg
        u = torch.rand(n, device=device, generator=gen)
        vals = (u < torch.sigmoid(logits)).float()
        for bi in range(n):
            k = int(perm[bi, i].item())
            x[bi, k] = vals[bi]
    return x


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    ok_all = True

    # ---------------------------------------------------------------- forward test
    for use_v in (True, False):
        for n_hidden in (1, 2, 3):
            m = NADE(dim=12, hidden=5, n_hidden=n_hidden, use_input_weights=use_v)
            x = torch.randint(0, 2, (3, 12)).float()
            perm = torch.stack([torch.randperm(12) for _ in range(3)])
            got = m._logits_from_ranks(x, m._ranks(perm))
            exp = brute_forward(m, x, perm)
            err = (got - exp).abs().max().item()
            good = err < 1e-4
            ok_all &= good
            print(f"[forward] use_V={use_v} n_hidden={n_hidden}: max|diff|={err:.2e} {'OK' if good else 'FAIL'}")

    # ---------------------------------------------------------------- sampler test
    m = NADE(dim=10, hidden=4, n_hidden=2)
    g1 = torch.Generator(device=device).manual_seed(7)
    fast = m.sample(3, device=device, generator=g1)
    g2 = torch.Generator(device=device).manual_seed(7)
    slow = brute_sample(m, 3, g2, device)
    err = (fast - slow).abs().max().item()
    ok_all &= err == 0
    print(f"[sample ] vectorised vs. brute force: max|diff|={err:.2e} {'OK' if err == 0 else 'FAIL'}")

    # ---------------------------------------------------------------- causality
    m = NADE(dim=64, hidden=16, n_hidden=2)
    x = torch.randint(0, 2, (4, 64)).float()
    ok = m.test_causality(x, n_probe=12)
    ok_all &= ok
    print(f"[causal ] no future leakage: {'PASS' if ok else 'FAIL'}")

    # ------------------------------------------------- likelihood sanity (dim 784)
    m = NADE(dim=784, hidden=500, n_hidden=2)
    x = torch.randint(0, 2, (8, 784)).float()
    l1 = m.loss(x).item()
    l2 = m.loss(x, perm=torch.arange(784).unsqueeze(0).repeat(8, 1)).item()
    print(f"[nll    ] random order={l1:.2f}  natural order={l2:.2f}  (both ~{784*0.693:.1f} for random bits)")
    print(f"[params ] {sum(p.numel() for p in m.parameters()):,}")
    print("\nALL TESTS PASS" if ok_all else "\nSOME TESTS FAILED")


if __name__ == "__main__":
    main()
