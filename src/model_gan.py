"""Task 3: GAN for handwritten digit generation on the *original* (grayscale) MNIST.

Two training objectives are implemented so that the stability analysis in the
report can be quantitative rather than anecdotal:

* ``dcgan`` -- the original DCGAN recipe (Radford et al., 2016): transposed
  convolution generator, strided convolution discriminator, ReLU / LeakyReLU,
  BatchNorm in G, and the non-saturating logistic loss with label smoothing.

* ``wgan``  -- Wasserstein GAN with gradient penalty (WGAN-GP, Arjovsky et al.,
  2016).  The discriminator is a 1-Lipschitz "critic"; the gradient penalty
  enforces that constraint explicitly, which is what makes WGAN-GP markedly
  more stable than vanilla DCGAN.

Architectures
-------------
Generator: latent z (100-d) -> dense 7x7x256 -> (ConvT 4x4 s2) -> 14x14x128
           -> (ConvT 4x4 s2) -> 28x28x1, tanh output in [-1, 1].
Discriminator: 28x28x1 -> (Conv 4x4 s2) -> 14x14x64 -> (Conv 4x4 s2) -> 7x7x128
           -> (Conv 3x3 s1) -> 5x5x128 -> flatten -> dense -> 1 logit.

The images are *not* binarised for this task (assignment 1.3): real images are
normalised to [-1, 1] to match the tanh generator output.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------------------
class Generator(nn.Module):
    """DCGAN generator: z -> 28x28x1 image in [-1, 1]."""

    def __init__(self, latent_dim: int = 100, base_channels: int = 128, img_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.base = base_channels

        self.fc = nn.Linear(latent_dim, base_channels * 7 * 7)
        self.bn0 = nn.BatchNorm1d(base_channels * 7 * 7)

        self.conv_t1 = nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(base_channels // 2)
        self.conv_t2 = nn.ConvTranspose2d(base_channels // 2, 1, 4, 2, 1)
        # (base//2) x 14 x 14 -> 1 x 28 x 28
        self.act = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = self.bn0(h).view(-1, self.base, 7, 7)
        h = self.act(self.bn1(self.conv_t1(h)))
        img = self.conv_t2(h)
        return torch.tanh(img)


class Discriminator(nn.Module):
    """DCGAN discriminator: 28x28x1 image -> one logit."""

    def __init__(self, base_channels: int = 64, img_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(1, base_channels, 4, 2, 1)  # 28 -> 14
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)  # 14 -> 7
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 0)  # 7 -> 5
        self.fc = nn.Linear(base_channels * 2 * 5 * 5, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.drop(h).flatten(1)
        return self.fc(h).squeeze(1)


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------
def d_loss_dcgan(d_real: torch.Tensor, d_fake: torch.Tensor, real_label: float = 0.9) -> torch.Tensor:
    """DCGAN discriminator loss with label smoothing."""
    loss_real = F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, real_label))
    loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    return 0.5 * (loss_real + loss_fake)


def g_loss_dcgan(d_fake: torch.Tensor) -> torch.Tensor:
    """DCGAN generator loss (non-saturating)."""
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


def wgan_gp_loss(
    critic: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    z: torch.Tensor,
    lambda_gp: float = 10.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """WGAN-GP losses.

    Returns ``(d_loss, g_loss, gp_term)``.
    """
    device = device or real.device
    d_real = critic(real).mean()
    d_fake = critic(fake.detach()).mean()

    # gradient penalty on the straight line between real and fake samples
    eps = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
    d_inter = critic(interpolated)
    grads = torch.autograd.grad(
        outputs=d_inter, inputs=interpolated,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    # NOTE: flatten to (B, D) *before* taking the norm.  Taking .norm(2, dim=1)
    # on the (B, 1, 28, 28) tensor would only normalise the channel axis, which
    # penalises the per-pixel gradient magnitude instead of the full gradient
    # norm and leaves the critic essentially unconstrained.
    grad_norm = grads.reshape(grads.size(0), -1).norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()

    d_loss = d_fake - d_real + lambda_gp * gp
    g_loss = -critic(fake).mean()
    return d_loss, g_loss, gp


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def sample_latent(n: int, latent_dim: int, device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    return torch.randn(n, latent_dim, device=device, generator=generator)


@torch.no_grad()
def generate_samples(
    G: Generator,
    n: int,
    latent_dim: int,
    device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate ``n`` images in [-1, 1], shape (n, 1, 28, 28)."""
    z = sample_latent(n, latent_dim, device, generator)
    return G(z)
