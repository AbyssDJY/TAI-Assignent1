"""Task 1: Fully Visible Sigmoid Belief Network (FVSBN) for binarized MNIST.

Model
-----
A *fully visible* sigmoid belief network is a directed graphical model whose
**hidden layers are deterministic functions of the visible layer** -- i.e. there
is no stochastic hidden unit at all.  The visible units are Bernoulli and the
model is specified by a single conditional distribution

    p(x_i = 1 | x_{-i}) = sigmoid( w_i^T x_{-i} + b_i )

This is exactly a (multivariate) **logistic regression** of each pixel on all
the other pixels.  The joint distribution is defined implicitly by the
collection of these conditionals, which is why generation is done by **Gibbs
sampling**: repeatedly resample each pixel from its conditional given the
current values of all the others.

Training
--------
The log-likelihood of a fully visible model is intractable in general, so we
maximise the **pseudo-likelihood** (a.k.a. *conditional* / *pseudo* likelihood),

    L(x) = sum_i  log p(x_i | x_{-i}) ,

which is fully differentiable and is the standard surrogate for a fully visible
SBN.  We minimise the binary cross entropy of every pixel given the rest,
which is exactly the pseudo-likelihood objective above.

Note on efficiency
------------------
Computing ``w_i^T x_{-i}`` for every i would be O(d^2) per sample with a
separate weight vector per pixel.  Instead we exploit the identity

    w_i^T x_{-i} + b_i  =  (W x)_i + b_i - W_ii x_i

so a *single* matrix multiply with a shared weight matrix W (with the diagonal
explicitly masked out) gives all pixel logits at once.  W is therefore a
784x784 matrix -- 615k parameters -- shared across pixels, which is what makes
the model trainable in minutes on one GPU.

We additionally offer an optional low-rank factorisation W = U V^T to control
capacity (used for the ablation study).

Sampling
--------
Gibbs sampling with a random pixel ordering.  Each sweep touches all 784
pixels once.  We use *parallel* (block) Gibbs updates over a random subset of
pixels at a time (a "checkerboard-free" random-block scheme): sampling pixels
one at a time is exact but needs 784 sequential GPU steps per sample, which is
slow; blocking keeps the sampler approximate but fast.  We make the block size
configurable so the exactness/speed trade-off can be reported.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FVSBN(nn.Module):
    """Fully visible sigmoid belief network (logistic regression over pixels).

    Parameters
    ----------
    dim : number of visible units (784 for 28x28).
    rank : if not None, W is factorised as ``U @ V.T`` with inner dimension
        ``rank`` (low-rank capacity control / ablation).
    """

    def __init__(self, dim: int = 784, rank: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        if rank is None:
            self.W = nn.Parameter(torch.zeros(dim, dim))
            self.U = None
            self.V = None
        else:
            self.U = nn.Parameter(torch.randn(dim, rank) * (1.0 / math.sqrt(rank)))
            self.V = nn.Parameter(torch.randn(rank, dim) * (1.0 / math.sqrt(dim)))
            self.W = None
        self.b = nn.Parameter(torch.zeros(dim))
        # mask that removes the self-connection W_ii
        self.register_buffer("eye", torch.eye(dim))

    # ---------------------------------------------------------------- core
    def effective_W(self) -> torch.Tensor:
        """Weight matrix with the diagonal masked to zero (no self-dependency)."""
        if self.rank is None:
            return self.W * (1.0 - self.eye)
        W = self.U @ self.V
        return W * (1.0 - self.eye)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Logit of every pixel given the *other* pixels: (B, D).

        The diagonal of W is masked to zero, so pixel i never sees itself and
        the self-term needs no separate correction.
        """
        W = self.effective_W()
        return x @ W.t() + self.b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    # ------------------------------------------------------------- training
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Pseudo-log-likelihood (as a *negative* loss, averaged over the batch).

        BCE over all pixels, summed over pixels and averaged over the batch.
        """
        z = self.logits(x)
        return F.binary_cross_entropy_with_logits(z, x, reduction="none").sum(dim=1).mean()

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        """Pseudo-log-likelihood in nats per dimension (for evaluation)."""
        z = self.logits(x)
        nll = F.binary_cross_entropy_with_logits(z, x, reduction="none").sum(dim=1)
        return nll.mean()

    # ------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(
        self,
        n: int,
        n_sweeps: int = 200,
        block_size: int = 32,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        temperature: float = 1.0,
        init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gibbs sampling.  Returns binary samples of shape (n, D).

        ``block_size`` pixels are updated simultaneously from the *same* current
        state (approximate / parallel Gibbs).  ``block_size = 1`` recovers exact
        single-site Gibbs at the cost of ``dim`` sequential steps per sweep.
        """
        device = device or next(self.parameters()).device
        if init is None:
            x = (torch.rand(n, self.dim, device=device, generator=generator) < 0.5).float()
        else:
            x = init.clone()

        W = self.effective_W()
        eye = self.eye
        for _ in range(n_sweeps):
            perm = torch.randperm(self.dim, device=device, generator=generator)
            for start in range(0, self.dim, block_size):
                idx = perm[start : start + block_size]
                # logits of the selected pixels given the current state
                z = x @ W[idx].t() + self.b[idx]          # (n, |idx|)
                # remove the (masked-out) self term -- W's diagonal is already 0
                p = torch.sigmoid(z / temperature)
                u = torch.rand(n, idx.numel(), device=device, generator=generator)
                x[:, idx] = (u < p).float()
        return x

    @torch.no_grad()
    def sample_grid(
        self,
        n: int,
        n_sweeps: int = 200,
        block_size: int = 32,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Samples reshaped to (n, 1, 28, 28) for plotting."""
        x = self.sample(n, n_sweeps, block_size, device, generator, temperature)
        return x.view(-1, 1, 28, 28)


# --------------------------------------------------------------------------------------
# Ordered (autoregressive) formulation -- the sampling-friendly FVSBN
# --------------------------------------------------------------------------------------
class FVSBNOrdered(nn.Module):
    """Fully visible sigmoid belief network with **ordered** conditionals.

    This is the formulation used as the "FVSBN" baseline in the NADE literature
    (Larochelle & Murray, 2011).  The visible units are put in a fixed order and
    each unit is a logistic regression on the units that precede it:

        p(x_i = 1 | x_1, ..., x_{i-1}) = sigmoid( w_i^T x_{<i} + b_i )

    Why the ordering matters
    ------------------------
    The *unordered* variant (``FVSBN`` above) specifies a separate conditional
    for every pixel given all the others and can only be trained with a
    pseudo-likelihood.  Those 784 conditionals need not be mutually consistent,
    so no joint distribution has to exist and Gibbs sampling on them is not
    guaranteed to converge to anything meaningful -- in our experiments it
    settles into noisy local modes.  Ordering the conditionals makes the model
    a proper directed graphical model:

    * the joint is exactly ``prod_i p(x_i | x_{<i})``, so the likelihood is
      **exact** (no pseudo-likelihood approximation),
    * generation is **exact ancestral sampling**, one pixel at a time, with no
      MCMC and no burn-in.

    The model has the same number of parameters as the unordered version
    (a full ``784 x 784`` weight matrix, restricted by a strictly lower
    triangular mask) and is therefore the cleanest possible ablation of "does
    ordering the conditionals help?".
    """

    def __init__(self, dim: int = 784, order: Optional[torch.Tensor] = None):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.zeros(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        # mask[i, j] = 1 iff pixel j precedes pixel i in the ordering
        if order is None:
            order = torch.arange(dim)
        self.register_buffer("order", order.long())
        rank = torch.empty(dim, dtype=torch.long)
        rank[order] = torch.arange(dim)
        self.register_buffer("rank", rank)
        # mask[i, j] = 1 iff pixel j precedes pixel i in the ordering, i.e.
        # rank(j) < rank(i).  Getting this transposed would let every pixel see
        # the future, which silently inflates the likelihood.
        mask = (rank.unsqueeze(1) > rank.unsqueeze(0)).float()   # [i, j] = rank_i > rank_j
        self.register_buffer("mask", mask)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    # ---------------------------------------------------------------- core
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) logits; ``logits[:, i]`` depends only on pixels before ``i``."""
        return x @ (self.W * self.mask).t() + self.b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    # ------------------------------------------------------------ training
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Exact negative log-likelihood (nats per image)."""
        return F.binary_cross_entropy_with_logits(
            self.logits(x), x, reduction="none").sum(dim=1).mean()

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        return self.loss(x)

    # ------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(
        self,
        n: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Exact ancestral sampling: one pixel at a time, in the model's order."""
        device = device or next(self.parameters()).device
        x = torch.zeros(n, self.dim, device=device)
        Wm = (self.W * self.mask)                    # rows already restricted to x_{<i}
        for k in range(self.dim):
            i = int(self.order[k])
            logit = x @ Wm[i] + self.b[i]            # (n,)
            u = torch.rand(n, device=device, generator=generator)
            x[:, i] = (u < torch.sigmoid(logit / temperature)).float()
        return x

    @torch.no_grad()
    def sample_grid(self, n: int, **kw) -> torch.Tensor:
        return self.sample(n, **kw).view(-1, 1, 28, 28)
