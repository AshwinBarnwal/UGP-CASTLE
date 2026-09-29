from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import CastleConfig


@dataclass
class BatchResult:
    loss: Tensor
    metrics: dict[str, float]
    z_intrinsic: Tensor
    z_niche: Tensor
    mu: Tensor
    purified: Tensor
    own_responsibility: Tensor
    eta: Tensor
    niche_effect: Tensor


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _kl_standard_normal(mean: Tensor, logvar: Tensor) -> Tensor:
    return 0.5 * torch.mean(torch.sum(mean.square() + logvar.exp() - logvar - 1.0, dim=1))


def _nb_nll(x: Tensor, mean: Tensor, dispersion: Tensor) -> Tensor:
    mean = mean.clamp_min(1e-8)
    dispersion = dispersion.clamp_min(1e-4)
    log_prob = (
        torch.lgamma(x + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(x + 1.0)
        + dispersion * (torch.log(dispersion) - torch.log(dispersion + mean))
        + x * (torch.log(mean) - torch.log(dispersion + mean))
    )
    return -log_prob.sum(dim=1).mean()


def _rbf_hsic(x: Tensor, y: Tensor, max_points: int = 256) -> Tensor:
    if x.shape[0] < 4:
        return x.new_zeros(())
    if x.shape[0] > max_points:
        take = torch.linspace(0, x.shape[0] - 1, max_points, device=x.device).long()
        x, y = x[take], y[take]
    dx = torch.cdist(x, x).square()
    dy = torch.cdist(y, y).square()
    positive_x = dx[dx > 0]
    positive_y = dy[dy > 0]
    sx = positive_x.median().detach().clamp_min(1e-6) if positive_x.numel() else x.new_tensor(1.0)
    sy = positive_y.median().detach().clamp_min(1e-6) if positive_y.numel() else y.new_tensor(1.0)
    k = torch.exp(-dx / sx)
    l = torch.exp(-dy / sy)
    k = k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()
    l = l - l.mean(0, keepdim=True) - l.mean(1, keepdim=True) + l.mean()
    return (k * l).sum() / max((x.shape[0] - 1) ** 2, 1)


class CastleModel(nn.Module):
    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        geometry_dim: int,
        median_total: float,
        cfg: CastleConfig,
    ) -> None:
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.n_genes = n_genes
        self.median_total = float(max(median_total, 1.0))
        self.encoder = _mlp(n_genes, m.hidden_dim, m.hidden_dim, m.dropout)
        self.zc_mean = nn.Linear(m.hidden_dim, m.intrinsic_dim)
        self.zc_logvar = nn.Linear(m.hidden_dim, m.intrinsic_dim)
        self.niche_encoder = _mlp(m.hidden_dim * 3, m.hidden_dim, m.hidden_dim, m.dropout)
        self.zn_mean = nn.Linear(m.hidden_dim, m.niche_dim)
        self.zn_logvar = nn.Linear(m.hidden_dim, m.niche_dim)
        self.intrinsic_decoder = _mlp(m.intrinsic_dim, m.hidden_dim, n_genes, m.dropout)
        self.niche_decoder = _mlp(
            m.intrinsic_dim + m.niche_dim, m.hidden_dim, n_genes, m.dropout
        )
        self.niche_reconstruction = nn.Linear(m.niche_dim, n_genes)
        self.eta_net = _mlp(geometry_dim, 32, 1, 0.0)
        self.log_library = nn.Embedding(n_cells, 1)
        self.log_dispersion = nn.Parameter(torch.zeros(n_genes))
        nn.init.constant_(self.log_library.weight, float(torch.log(torch.tensor(self.median_total))))

    def encode_intrinsic(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.encoder(torch.log1p(x / self.median_total))
        return h, self.zc_mean(h), self.zc_logvar(h).clamp(-8.0, 6.0)

    def encode_niche(self, niche_context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.niche_encoder(niche_context)
        return h, self.zn_mean(h), self.zn_logvar(h).clamp(-8.0, 6.0)

    @staticmethod
    def _sample(mean: Tensor, logvar: Tensor, stochastic: bool) -> Tensor:
        if not stochastic:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def decode(self, zc: Tensor, zn: Tensor, use_niche: bool) -> tuple[Tensor, Tensor]:
        intrinsic = self.intrinsic_decoder(zc)
        delta = self.niche_decoder(torch.cat([zc, zn], dim=1)) if use_niche else torch.zeros_like(intrinsic)
        return torch.softmax(intrinsic + delta, dim=1), delta

    @staticmethod
    def _incoming_rate(
        batch_size: int,
        n_genes: int,
        local_targets: Tensor,
        donor_mu: Tensor,
        donor_library: Tensor,
        donor_eta: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        incoming = donor_mu.new_zeros((batch_size, n_genes))
        if donor_mu.shape[0] == 0:
            return incoming
        contribution = donor_mu * (
            donor_library * donor_eta * edge_weight
        ).unsqueeze(1)
        incoming.index_add_(0, local_targets, contribution)
        return incoming

    def forward_batch(
        self,
        x: Tensor,
        cell_ids: Tensor,
        geometry: Tensor,
        niche_context: Tensor,
        neighbor_profile: Tensor,
        local_targets: Tensor,
        donor_mu: Tensor,
        donor_library: Tensor,
        donor_eta: Tensor,
        edge_weight: Tensor,
        background: Tensor,
        library_prior_log: Tensor,
        phase: str,
        beta_intrinsic: float,
        stochastic: bool = True,
    ) -> BatchResult:
        use_contamination = phase != "warmup" and self.cfg.model.contamination.enabled
        use_niche = phase == "joint"
        eta = torch.sigmoid(self.eta_net(geometry)).squeeze(1)
        eta = eta * self.cfg.model.contamination.eta_max if use_contamination else eta * 0.0
        library = torch.exp(self.log_library(cell_ids).squeeze(1)).clamp(1.0, 1e7)
        incoming = self._incoming_rate(
            x.shape[0], self.n_genes, local_targets, donor_mu, donor_library,
            donor_eta if use_contamination else donor_eta * 0.0, edge_weight,
        )

        h, zc_mean, zc_logvar = self.encode_intrinsic(x)
        _, zn_mean, zn_logvar = self.encode_niche(niche_context)
        purified = x
        own_resp = torch.ones_like(x)
        delta = torch.zeros_like(x)
        steps = self.cfg.model.contamination.purify_steps if use_contamination else 1
        for step in range(max(1, steps)):
            if step > 0:
                h, zc_mean, zc_logvar = self.encode_intrinsic(purified)
            zc = self._sample(zc_mean, zc_logvar, stochastic)
            zn = self._sample(zn_mean, zn_logvar, stochastic)
            mu, delta = self.decode(zc, zn, use_niche)
            own = (1.0 - eta).unsqueeze(1) * library.unsqueeze(1) * mu
            expected = own + incoming + background
            own_resp = own / expected.clamp_min(1e-8)
            purified = x * own_resp.clamp(0.0, 1.0)

        dispersion = F.softplus(self.log_dispersion).unsqueeze(0) + 1e-4
        reconstruction = _nb_nll(x, expected, dispersion)
        kl_c = _kl_standard_normal(zc_mean, zc_logvar)
        kl_n = _kl_standard_normal(zn_mean, zn_logvar) if use_niche else x.new_zeros(())
        sparse_niche = delta.abs().mean() if use_niche else x.new_zeros(())
        niche_prediction = torch.softmax(self.niche_reconstruction(zn), dim=1)
        niche_recon = F.mse_loss(niche_prediction, neighbor_profile) if use_niche else x.new_zeros(())
        hsic = _rbf_hsic(zc_mean, zn_mean) if use_niche else x.new_zeros(())
        eta_penalty = eta.mean()
        library_penalty = F.mse_loss(torch.log(library), library_prior_log)
        weights = self.cfg.loss
        loss = (
            reconstruction
            + beta_intrinsic * kl_c
            + weights.beta_niche * kl_n
            + weights.lambda_niche_sparse * sparse_niche
            + weights.lambda_niche_reconstruction * niche_recon
            + weights.lambda_hsic * hsic
            + weights.lambda_eta * eta_penalty
            + weights.lambda_library * library_penalty
        )
        metrics = {
            "loss": float(loss.detach()),
            "reconstruction": float(reconstruction.detach()),
            "kl_intrinsic": float(kl_c.detach()),
            "kl_niche": float(kl_n.detach()),
            "mean_eta": float(eta.mean().detach()),
            "mean_own_responsibility": float(own_resp.mean().detach()),
            "library_penalty": float(library_penalty.detach()),
        }
        return BatchResult(
            loss, metrics, zc_mean, zn_mean, mu, purified, own_resp, eta, delta
        )
