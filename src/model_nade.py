"""Task 2: Neural autoregressive model for binarized MNIST.

Two architectures are implemented:

1. ``NADE`` -- Neural Autoregressive Distribution Estimator (Larochelle &
   Murray, 2011), i.e. a MADE (Germain et al., 2015) trained in an
   *order-agnostic* fashion: for every training example a fresh random
   ordering of the 784 pixels is drawn.  The joint density factorises exactly

       p(x) = prod_k p(x_{pi(k)} | x_{pi(0)}, ..., x_{pi(k-1)})

   and the model is *permutation invariant*: the same weights are used for
   every ordering, which is what makes NADE a strong generative model on
   binarized MNIST.

   **How the ordering is applied** (this is the subtle part).  The random
   ordering must *not* be implemented by permuting the data and then applying
   a fixed MADE mask: doing so makes every weight act on a *different* pixel in
   every example, the model can no longer tell which spatial pixel it is
   predicting, and the likelihood collapses to that of a constant predictor.
   Instead the ordering is applied to the *spatial index*:

   * let ``r[b, k]`` be the rank of spatial pixel ``k`` in example ``b``;
   * hidden unit ``j`` of layer 0 may see spatial pixel ``k`` iff
     ``r[b, k] <= cap_j``;
   * the output for spatial pixel ``k`` may use hidden unit ``j`` iff
     ``cap_j < r[b, k]``;
   * the direct input->output term may connect spatial pixel ``q`` to spatial
     pixel ``k`` iff ``r[b, q] < r[b, k]``.

   The *weights* therefore stay anchored to the spatial identity of the
   pixels, while the *connectivity* is re-drawn per example.  All masks are
   boolean tensors of shape ``(B, 784, H)`` / ``(B, 784, 784)``.

   ``test_causality`` verifies numerically that output pixel ``k`` never
   depends on a pixel whose rank is larger than ``r[k]``.

2. ``GRUAR`` -- a recurrent autoregressive baseline (GRU with teacher forcing).
   Used as a comparison point in the ablation study.

Generation for both is exact ancestral sampling (no MCMC required).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# NADE
# --------------------------------------------------------------------------------------
class NADE(nn.Module):
    """Order-agnostic Neural Autoregressive Distribution Estimator.

    Parameters
    ----------
    dim              : number of visible pixels (784).
    hidden           : width of each hidden layer.
    n_hidden         : number of hidden layers (>= 1).
    use_input_weights: include NADE's direct input->output connections ("V").
    order_agnostic   : if True a fresh random ordering is drawn per example
                       (NADE); if False the natural raster order is always
                       used, which degenerates to a fixed-order MADE.
    """

    def __init__(
        self,
        dim: int = 784,
        hidden: int = 500,
        n_hidden: int = 2,
        use_input_weights: bool = True,
        order_agnostic: bool = True,
    ):
        super().__init__()
        assert n_hidden >= 1
        self.dim = dim
        self.hidden = hidden
        self.n_hidden = n_hidden
        self.use_input_weights = use_input_weights
        self.order_agnostic = order_agnostic

        # ---- "capacity" of every hidden unit -----------------------------------------
        # cap_j is the largest *rank* whose information may reach hidden unit j.
        # The caps are spread evenly over [0, dim-1] and include both 0 and
        # dim-1, so that
        #   * every output of rank k >= 1 has at least one usable hidden unit
        #     (the one with cap 0), and
        #   * the last input rank dim-1 is reachable by the last unit.
        if hidden == 1:
            caps = torch.tensor([dim - 1], dtype=torch.long)
        else:
            caps = torch.round(
                torch.arange(hidden, dtype=torch.float32) * (dim - 1) / (hidden - 1)
            ).long()
        caps = caps.clamp(0, dim - 1)
        self.register_buffer("caps", caps)                    # (H,)

        # ---- weights -----------------------------------------------------------------
        # W1[k, j] : spatial input k -> hidden unit j          (D, H)
        self.W1 = nn.Parameter(torch.zeros(dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        # Wh[l][j2, j1] : hidden unit j1 -> hidden unit j2 of the next layer
        self.Wh = nn.ParameterList()
        self.bh = nn.ParameterList()
        for _ in range(n_hidden - 1):
            self.Wh.append(nn.Parameter(torch.zeros(hidden, hidden)))
            self.bh.append(nn.Parameter(torch.zeros(hidden)))
        # W3[k, j] : hidden unit j -> spatial output k          (D, H)
        self.W3 = nn.Parameter(torch.zeros(dim, hidden))
        self.b3 = nn.Parameter(torch.zeros(dim))
        # V[k, q]  : direct spatial input q -> spatial output k (D, D)
        if use_input_weights:
            self.V = nn.Parameter(torch.zeros(dim, dim))
        else:
            self.register_parameter("V", None)

        # ---- static hidden->hidden connectivity --------------------------------------
        # mh[j2, j1] = 1 iff cap_{l-1}(j1) <= cap_l(j2), i.e. the information seen
        # by j2 stays inside j2's own capacity.
        for l in range(1, n_hidden):
            cap_prev = caps.unsqueeze(0)      # (1,H) indexed by j1
            cap_cur = caps.unsqueeze(1)       # (H,1) indexed by j2
            self.register_buffer(f"mh{l - 1}", (cap_prev <= cap_cur))

        self.reset_parameters()

    # ------------------------------------------------------------------ init
    def reset_parameters(self) -> None:
        for w in [self.W1, self.W3] + list(self.Wh):
            nn.init.xavier_uniform_(w)
        if self.V is not None:
            nn.init.xavier_uniform_(self.V)
        for b in [self.b1, self.b3] + list(self.bh):
            nn.init.zeros_(b)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _random_perm(b: int, dim: int, device, generator=None) -> torch.Tensor:
        """(B, dim): row b lists the *spatial* pixel indices in the sampled order.

        ``perm[b, i]`` is the spatial index of the pixel revealed at step ``i``.
        """
        return torch.argsort(torch.rand(b, dim, device=device, generator=generator), dim=1)

    def _ranks(self, perm: torch.Tensor) -> torch.Tensor:
        """(B, dim) rank of every spatial pixel: ``r[b, k] = i`` iff ``perm[b, i] == k``."""
        return torch.argsort(perm, dim=1)

    def _perms(self, b: int, device, generator=None) -> torch.Tensor:
        if self.order_agnostic:
            return self._random_perm(b, self.dim, device, generator)
        return torch.arange(self.dim, device=device).unsqueeze(0).repeat(b, 1)

    # ---------------------------------------------------------------- forward
    def _logits_from_ranks(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Logits for every spatial pixel given the per-example rank matrix ``r``."""
        b = x.shape[0]
        caps = self.caps                                     # (H,)

        # --- layer 0: hidden unit j sees spatial pixel k iff r[k] <= cap_j
        m1 = r.unsqueeze(2) <= caps.view(1, 1, self.hidden)  # (B, D, H) bool
        h = torch.tanh(
            torch.matmul((m1 * self.W1.unsqueeze(0)).transpose(1, 2), x.unsqueeze(2)).squeeze(2)
            + self.b1
        )
        # --- hidden layers (static capacity connectivity)
        for l in range(1, self.n_hidden):
            mask = getattr(self, f"mh{l - 1}")
            h = torch.tanh(
                torch.matmul((mask * self.Wh[l - 1]).unsqueeze(0), h.unsqueeze(2)).squeeze(2)
                + self.bh[l - 1]
            )
        # --- output layer: spatial pixel k may use hidden unit j iff cap_j < r[k]
        m3 = caps.view(1, 1, self.hidden) < r.unsqueeze(2)    # (B, D, H) bool
        out = torch.matmul(m3 * self.W3.unsqueeze(0), h.unsqueeze(2)).squeeze(2) + self.b3
        # --- direct input->output path: q may feed k iff r[q] < r[k]
        if self.V is not None:
            mv = r.unsqueeze(1) < r.unsqueeze(2)             # (B, D, D) bool: mv[k,q]=1[r[q]<r[k]]
            out = out + torch.matmul(mv * self.V.unsqueeze(0),
                                     x.unsqueeze(2)).squeeze(2)
        return out

    def forward(self, x: torch.Tensor, perm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-pixel Bernoulli *logits*, returned in the original (spatial) order."""
        if perm is None:
            perm = self._perms(x.shape[0], x.device)
        return self._logits_from_ranks(x, self._ranks(perm))

    def loss(self, x: torch.Tensor, perm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Exact negative log-likelihood (nats per image) -- no sampling needed."""
        if perm is None:
            perm = self._perms(x.shape[0], x.device)
        logits = self._logits_from_ranks(x, self._ranks(perm))
        return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()

    @torch.no_grad()
    def nll(self, x: torch.Tensor, n_orders: int = 1, natural: bool = False) -> torch.Tensor:
        """Average NLL over ``n_orders`` orderings.

        ``natural=True`` uses the fixed raster order, which makes the value
        deterministic and therefore directly comparable / reproducible.
        """
        if natural:
            return self.loss(x, perm=self._perms(x.shape[0], x.device))
        total = 0.0
        for _ in range(n_orders):
            total = total + self.loss(x, perm=self._perms(x.shape[0], x.device))
        return total / n_orders

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(
        self,
        n: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        temperature: float = 1.0,
        order_agnostic: Optional[bool] = None,
    ) -> torch.Tensor:
        """Exact ancestral sampling.

        The pixels are revealed one at a time in the sampled order.  Two
        running accumulators make this cheap:
          * ``acc1``  -- the (masked) input contribution of every hidden unit,
          * ``outV``  -- the direct input->output contribution of every pixel.
        """
        device = device or next(self.parameters()).device
        oa = self.order_agnostic if order_agnostic is None else order_agnostic
        D, H = self.dim, self.hidden
        perm = (self._random_perm(n, D, device, generator) if oa
                else torch.arange(D, device=device).unsqueeze(0).repeat(n, 1))

        x = torch.zeros(n, D, device=device)
        caps = self.caps
        acc1 = torch.zeros(n, H, device=device)
        outV = torch.zeros(n, D, device=device)
        Vt = self.V.t().contiguous() if self.V is not None else None   # Vt[q, k] = V[k, q]

        for i in range(D):
            q = perm[:, i]                                     # spatial pixel revealed now

            # hidden states from the pixels revealed so far
            h = torch.tanh(acc1 + self.b1)
            for l in range(1, self.n_hidden):
                mask = getattr(self, f"mh{l - 1}")
                h = torch.tanh(
                    torch.matmul(mask * self.Wh[l - 1], h.unsqueeze(2)).squeeze(2) + self.bh[l - 1]
                )

            # logit of the pixel we are about to reveal
            w3 = self.W3[q]                                    # (n, H)
            m = (caps < i).to(w3.dtype)                        # only finished units
            logit = (w3 * m.unsqueeze(0) * h).sum(1) + self.b3[q]
            if self.V is not None:
                logit = logit + outV.gather(1, q.unsqueeze(1)).squeeze(1)

            u = torch.rand(n, device=device, generator=generator)
            val = (u < torch.sigmoid(logit / temperature)).float()
            x = x.scatter(1, q.unsqueeze(1), val.unsqueeze(1))

            # feed the new pixel into the accumulators (only into units whose
            # capacity has not been exceeded yet)
            acc1 = acc1 + self.W1[q] * (caps >= i).to(w3.dtype).unsqueeze(0) * val.unsqueeze(1)
            if self.V is not None:
                outV = outV + Vt[q] * val.unsqueeze(1)
        return x

    @torch.no_grad()
    def sample_grid(self, n: int, **kw) -> torch.Tensor:
        return self.sample(n, **kw).view(-1, 1, 28, 28)

    # ------------------------------------------------------------ diagnostics
    def test_causality(self, x: torch.Tensor, n_probe: int = 16, seed: int = 0) -> bool:
        """Check that output pixel ``k`` never depends on a *later* pixel.

        For a random subset of output pixels we compute the gradient of that
        output with respect to all inputs and verify that the gradient
        vanishes for every input whose rank is larger than the rank of the
        output.  A violation would mean the model can see the future, which
        would make the likelihood meaningless.
        """
        x = x.clone()
        b = x.shape[0]
        perm = self._perms(b, x.device)
        r = self._ranks(perm)
        x = x.detach().requires_grad_(True)
        logits = self._logits_from_ranks(x, r)

        g = torch.Generator(device="cpu").manual_seed(seed)
        ks = torch.randperm(self.dim, generator=g)[:n_probe].tolist()
        for k in ks:
            grad = torch.autograd.grad(logits[:, k].sum(), x, retain_graph=True)[0]
            later = r > r[:, k].unsqueeze(1)                   # (B, D) bool
            if (grad.abs() * later.float()).max().item() > 1e-6:
                return False
        return True


# --------------------------------------------------------------------------------------
# Recurrent autoregressive baseline
# --------------------------------------------------------------------------------------
class GRUAR(nn.Module):
    """Recurrent (GRU) autoregressive model with teacher forcing.

    The RNN consumes the pixel sequence one pixel at a time.  The state that
    has already seen pixels ``0..k-1`` predicts pixel ``k``, so the likelihood
    is an exact product of 784 Bernoulli conditionals.

    The first pixel in the ordering has no prefix, so its conditional is a
    learned constant (``first_logit``); the same constant is used during
    sampling, which keeps training and generation consistent.
    """

    def __init__(self, dim: int = 784, hidden: int = 512, n_layers: int = 1):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        self.first_logit = nn.Parameter(torch.zeros(1))

    # -- teacher-forced forward ------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, dim) -> logits (B, dim); logits[:, t] predicts x[:, t] from x[:, <t]."""
        b = x.shape[0]
        # the state that has seen x_0..x_{t-1} is produced by feeding x_0..x_{t-2}
        seq = x[:, :-1].unsqueeze(-1)                       # (B, dim-1, 1)
        out, _ = self.rnn(seq)                             # (B, dim-1, H)
        raw = self.out(out).squeeze(-1)                    # (B, dim-1) -> predicts x_1..x_{dim-1}
        first = self.first_logit.expand(b, 1)
        return torch.cat([first, raw], dim=1)               # (B, dim)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()

    @torch.no_grad()
    def nll(self, x: torch.Tensor, n_orders: int = 1, natural: bool = False) -> torch.Tensor:
        return self.loss(x)

    # -- ancestral sampling ----------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        n: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device
        x = torch.zeros(n, self.dim, device=device)
        h = None
        for t in range(self.dim):
            if t == 0:
                # nothing has been consumed yet -> constant conditional
                logit = self.first_logit.expand(n).clone()
            else:
                # feed the pixel generated in the previous step; the resulting
                # state has seen x_0..x_{t-1} and therefore predicts x_t
                inp = x[:, t - 1].view(n, 1, 1)
                _, h = self.rnn(inp, h)
                logit = self.out(h[-1]).squeeze(-1)
            u = torch.rand(n, device=device, generator=generator)
            x[:, t] = (u < torch.sigmoid(logit / temperature)).float()
        return x

    @torch.no_grad()
    def sample_grid(self, n: int, **kw) -> torch.Tensor:
        return self.sample(n, **kw).view(-1, 1, 28, 28)
