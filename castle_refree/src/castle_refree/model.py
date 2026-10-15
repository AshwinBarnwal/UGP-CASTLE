from __future__ import annotations

import math
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
    hidden: Tensor
    library: Tensor
    expected: Tensor | None = None


def _mlp(
    input_dim: int, hidden_dim: int, output_dim: int, dropout: float,
    n_layers: int = 1, norm: str = "layer",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(max(1, n_layers)):
        layers += [
            nn.Linear(width, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if norm == "batch" else nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        ]
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


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


def _zinb_nll(x: Tensor, mean: Tensor, dispersion: Tensor, zero_logits: Tensor) -> Tensor:
    """Zero-inflated NB. Inflation only adds mass at zero, so the source
    responsibilities of non-zero counts are unchanged."""
    mean = mean.clamp_min(1e-8)
    dispersion = dispersion.clamp_min(1e-4)
    log_theta_mu = torch.log(dispersion + mean)
    nb_zero = dispersion * (torch.log(dispersion) - log_theta_mu)
    log_zero = F.softplus(-zero_logits + nb_zero) - F.softplus(-zero_logits)
    log_nonzero = (
        -zero_logits - F.softplus(-zero_logits)
        + nb_zero
        + x * (torch.log(mean) - log_theta_mu)
        + torch.lgamma(x + dispersion) - torch.lgamma(dispersion) - torch.lgamma(x + 1.0)
    )
    log_prob = torch.where(x < 1e-8, log_zero, log_nonzero)
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


def _segment_logsumexp(values: Tensor, segment: Tensor, n_segments: int) -> Tensor:
    peak = values.new_full((n_segments,), -torch.inf).scatter_reduce(
        0, segment, values, reduce="amax", include_self=True
    )
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak)).detach()
    total = values.new_zeros(n_segments).index_add(0, segment, torch.exp(values - peak[segment]))
    return peak + torch.log(total.clamp_min(1e-30))


class CastleModel(nn.Module):
    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        geometry_dim: int,
        median_total: float,
        cfg: CastleConfig,
        median_edge_distance: float = 1.0,
        log_kappa_init: float = 0.0,
    ) -> None:
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.n_genes = n_genes
        self.median_total = float(max(median_total, 1.0))
        deep = {"n_layers": m.n_layers, "norm": m.norm}
        encoder_width = n_genes * (2 if m.encoder_input == "dual" else 1)
        self.encoder = _mlp(encoder_width, m.hidden_dim, m.hidden_dim, m.dropout, **deep)
        visible = torch.ones(n_genes)
        visible[list(m.hidden_genes)] = 0.0  # genes hidden from the encoder (held-out tuning)
        self.register_buffer("encoder_visible", visible)
        if m.prior_components > 0:
            # Learned Gaussian-mixture prior on z_c; sparse Dirichlet on the weights
            # lets unused components switch off.
            self.prior_means = nn.Parameter(0.5 * torch.randn(m.prior_components, m.intrinsic_dim))
            self.prior_logvars = nn.Parameter(torch.zeros(m.prior_components, m.intrinsic_dim))
            self.prior_logits = nn.Parameter(torch.zeros(m.prior_components))
        self.zc_mean = nn.Linear(m.hidden_dim, m.intrinsic_dim)
        self.zc_logvar = nn.Linear(m.hidden_dim, m.intrinsic_dim)
        self.n_cells = n_cells
        if m.niche_encoder == "contact_steps":
            # Per contact-step scale: attention over the scale's cells (own
            # expression excluded) plus azimuthal magnitudes m = 0, 1, 2.
            hdim, scales = m.hidden_dim, m.niche_steps
            in_dim = m.intrinsic_dim if m.niche_input == "zc" else hdim
            self.niche_key = nn.ModuleList(nn.Linear(in_dim, hdim) for _ in range(scales))
            self.niche_edge = nn.ModuleList(nn.Linear(3, hdim) for _ in range(scales))
            self.niche_score = nn.ModuleList(nn.Linear(hdim, 1) for _ in range(scales))
            self.niche_value = nn.ModuleList(nn.Linear(in_dim, hdim) for _ in range(scales))
            niche_input = (hdim + 3 * in_dim) * scales
        else:
            niche_input = (m.intrinsic_dim if m.niche_input == "zc" else m.hidden_dim) * 3
        self.niche_encoder = _mlp(niche_input, m.hidden_dim, m.hidden_dim, m.dropout)
        self.zn_mean = nn.Linear(m.hidden_dim, m.niche_dim)
        self.zn_logvar = nn.Linear(m.hidden_dim, m.niche_dim)
        self.intrinsic_decoder = _mlp(m.intrinsic_dim, m.hidden_dim, n_genes, m.dropout, **deep)
        self.niche_decoder = _mlp(
            m.intrinsic_dim + m.niche_dim, m.hidden_dim, n_genes, m.dropout
        )
        self.niche_reconstruction = nn.Linear(m.niche_dim, n_genes)
        self.niche_zc_head = nn.Linear(m.niche_dim, m.intrinsic_dim)
        self.eta_net = _mlp(geometry_dim, 32, 1, 0.0)
        self.log_library = nn.Embedding(n_cells, 1)
        self.log_dispersion = nn.Parameter(torch.zeros(n_genes))
        if m.likelihood == "zinb":
            self.zero_inflation = nn.Linear(m.intrinsic_dim, n_genes)
        nn.init.constant_(self.log_library.weight, float(torch.log(torch.tensor(self.median_total))))
        c = m.contamination
        if c.background == "controls_learned":
            # beta_g: background profile over genes, uniform at initialisation.
            self.background_logits = nn.Parameter(torch.zeros(n_genes))
        if c.kernel == "learned":
            # Initialised to the fixed distance kernel: tau = median edge distance.
            self.log_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(max(median_edge_distance, 1e-3))))))
            self.alpha_boundary = nn.Parameter(torch.zeros(()))
            self.alpha_size = nn.Parameter(torch.zeros(()))
        if c.library == "amortized":
            self.library_mean = nn.Linear(m.hidden_dim, 1)
            self.library_logvar = nn.Linear(m.hidden_dim, 1)
            self.log_kappa = nn.Parameter(torch.tensor(float(log_kappa_init)))
            nn.init.zeros_(self.library_mean.weight)
            nn.init.constant_(self.library_mean.bias, float(log_kappa_init))

    def encode_intrinsic(self, x: Tensor, purified: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """With ``encoder_input='dual'`` the encoder sees raw and purified counts."""
        inputs = torch.log1p(x / self.median_total) * self.encoder_visible
        if self.cfg.model.encoder_input == "dual":
            clean = x if purified is None else purified
            inputs = torch.cat([inputs, torch.log1p(clean / self.median_total) * self.encoder_visible], dim=1)
        h = self.encoder(inputs)
        return h, self.zc_mean(h), self.zc_logvar(h).clamp(-8.0, 6.0)

    def kl_intrinsic(self, mean: Tensor, logvar: Tensor, z: Tensor) -> Tensor:
        """KL(q(z_c|x) || p(z_c)); Monte Carlo against the mixture prior."""
        if self.cfg.model.prior_components <= 0:
            return _kl_standard_normal(mean, logvar)
        log_q = -0.5 * (((z - mean) ** 2) / logvar.exp() + logvar + math.log(2 * math.pi)).sum(dim=1)
        diff = z.unsqueeze(1) - self.prior_means.unsqueeze(0)
        log_component = -0.5 * ((diff**2) / self.prior_logvars.exp() + self.prior_logvars + math.log(2 * math.pi)).sum(dim=2)
        log_p = torch.logsumexp(torch.log_softmax(self.prior_logits, dim=0) + log_component, dim=1)
        return (log_q - log_p).mean()

    def prior_weight_penalty(self) -> Tensor:
        if self.cfg.model.prior_components <= 0:
            return self.log_dispersion.new_zeros(())
        log_pi = torch.log_softmax(self.prior_logits, dim=0)
        return -(self.cfg.loss.prior_weight_alpha - 1.0) * log_pi.sum() / self.n_cells

    def contact_niche_features(self, context: list[dict[str, Tensor]]) -> Tensor:
        """Per contact-step scale: attention message and azimuthal magnitudes."""
        features = []
        for k, scale in enumerate(context):
            hidden, mask = scale["hidden"], scale["mask"]  # [B, K, H], [B, K]
            maskf = mask.to(hidden.dtype)
            edge = torch.stack([scale["share"], scale["distance"], scale["steps"]], dim=-1)
            score = self.niche_score[k](
                torch.tanh(self.niche_key[k](hidden) + self.niche_edge[k](edge))
            ).squeeze(-1)
            attention = torch.softmax(score.masked_fill(~mask, -1e9), dim=1) * maskf
            message = (attention.unsqueeze(-1) * self.niche_value[k](hidden)).sum(dim=1)
            # Equal weight per cell in the scale; empty scales give zeros.
            weight = (maskf / maskf.sum(dim=1, keepdim=True).clamp_min(1.0)).unsqueeze(-1)
            angle = scale["angle"].unsqueeze(-1)
            magnitudes = [(weight * hidden).sum(dim=1).abs()]
            for order in (1, 2):
                real = (weight * hidden * torch.cos(order * angle)).sum(dim=1)
                imag = (weight * hidden * torch.sin(order * angle)).sum(dim=1)
                magnitudes.append(torch.sqrt(real.square() + imag.square() + 1e-12))
            features.extend([message, *magnitudes])
        return torch.cat(features, dim=1)

    def encode_niche(self, niche_context) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(niche_context, list):
            niche_context = self.contact_niche_features(niche_context)
        h = self.niche_encoder(niche_context)
        return h, self.zn_mean(h), self.zn_logvar(h).clamp(-8.0, 6.0)

    @staticmethod
    def _sample(mean: Tensor, logvar: Tensor, stochastic: bool) -> Tensor:
        if not stochastic:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def decode(self, zc: Tensor, zn: Tensor, use_niche: bool) -> tuple[Tensor, Tensor]:
        intrinsic = self.intrinsic_decoder(zc)
        zc_for_niche = zc.detach() if self.cfg.loss.block_delta_grad else zc
        delta = self.niche_decoder(torch.cat([zc_for_niche, zn], dim=1)) if use_niche else torch.zeros_like(intrinsic)
        return torch.softmax(intrinsic + delta, dim=1), delta

    def eta_from_geometry(self, geometry: Tensor, has_out: Tensor) -> Tensor:
        eta = torch.sigmoid(self.eta_net(geometry)).squeeze(1)
        return eta * self.cfg.model.contamination.eta_max * has_out.to(eta.dtype)

    def edge_logits(self, features: Tensor, shift: Tensor) -> Tensor:
        """features columns: distance, boundary fraction, recipient size share."""
        return (
            -features[:, 0] / torch.exp(self.log_tau)
            + self.alpha_boundary * features[:, 1]
            + self.alpha_size * features[:, 2]
            + shift
        )

    def learned_in_weights(
        self, in_features: Tensor, in_shift: Tensor, in_donor: Tensor,
        out_features: Tensor, out_shift: Tensor, out_donor: Tensor, n_donors: int,
    ) -> Tensor:
        """w_{j->i} for incoming edges, normalised over each donor's full contact list."""
        normaliser = _segment_logsumexp(self.edge_logits(out_features, out_shift), out_donor, n_donors)
        return torch.exp(self.edge_logits(in_features, in_shift) - normaliser[in_donor])

    def forward_batch(
        self,
        x: Tensor,
        cell_ids: Tensor,
        geometry: Tensor,
        own_has_out: Tensor,
        niche_context: Tensor,
        neighbor_profile: Tensor,
        local_targets: Tensor,
        in_donor: Tensor,
        in_features: Tensor,
        in_shift: Tensor,
        in_weight: Tensor,
        donor_geometry: Tensor,
        donor_has_out: Tensor,
        donor_mu: Tensor,
        donor_library: Tensor,
        donor_eta: Tensor,
        out_donor: Tensor,
        out_features: Tensor,
        out_shift: Tensor,
        background: Tensor,
        library_prior_log: Tensor,
        phase: str,
        beta_intrinsic: float,
        stochastic: bool = True,
        own_eta: Tensor | None = None,
        donor_purified: Tensor | None = None,
        donor_zn: Tensor | None = None,
        neighbor_zc: Tensor | None = None,
    ) -> BatchResult:
        c = self.cfg.model.contamination
        use_contamination = phase != "warmup" and c.enabled
        use_niche = phase == "joint"
        # own_eta fixes the leakage fractions (e.g. null-calibrated values); the
        # donors' fractions then come from the same fixed cache.
        eta = own_eta if own_eta is not None else self.eta_from_geometry(geometry, own_has_out)
        if not use_contamination:
            eta = eta * 0.0

        # Spillover received by each batch cell from its donors.
        incoming = x.new_zeros(x.shape)
        if use_contamination and local_targets.numel():
            n_donors = donor_mu.shape[0]
            if c.kernel == "learned":
                weight = self.learned_in_weights(
                    in_features, in_shift, in_donor, out_features, out_shift, out_donor, n_donors
                )
            else:
                weight = in_weight
            if c.donor_gradient and own_eta is None:
                eta_donor = self.eta_from_geometry(donor_geometry, donor_has_out)
            else:
                eta_donor = donor_eta
            if c.donor_live_profiles and donor_purified is not None:
                # Spec Sec. 7.1: donors' profiles decoded inside the batch, so the
                # spillover term also trains the encoder and decoder. Donors are
                # encoded from their cached purified counts (second hop cached).
                _, donor_zc, _ = self.encode_intrinsic(donor_purified)
                donor_mu, _ = self.decode(donor_zc, donor_zn, use_niche)
            rate = (donor_library * eta_donor)[in_donor] * weight
            incoming = incoming.index_add(0, local_targets, donor_mu[in_donor] * rate.unsqueeze(1))

        h, zc_mean, zc_logvar = self.encode_intrinsic(x)
        if niche_context is None:
            # Niche inputs are only built for the niche phase; z_n is unused before.
            zn_mean = x.new_zeros((x.shape[0], self.cfg.model.niche_dim))
            zn_logvar = torch.zeros_like(zn_mean)
        else:
            _, zn_mean, zn_logvar = self.encode_niche(niche_context)

        if c.library == "amortized":
            lib_mean = self.library_mean(h).squeeze(1)
            lib_logvar = self.library_logvar(h).squeeze(1).clamp(-8.0, 4.0)
            log_library = self._sample(lib_mean, lib_logvar, stochastic)
            library = torch.exp(log_library).clamp(1.0, 1e7)
            prior_mean = self.log_kappa + library_prior_log
            prior_var = c.library_prior_sd**2
            library_penalty = 0.5 * torch.mean(
                (lib_logvar.exp() + (lib_mean - prior_mean).square()) / prior_var
                - 1.0 - lib_logvar + math.log(prior_var)
            )
            library_weight = 1.0
        else:
            library = torch.exp(self.log_library(cell_ids).squeeze(1)).clamp(1.0, 1e7)
            library_penalty = F.mse_loss(torch.log(library), library_prior_log)
            library_weight = self.cfg.loss.lambda_library

        background_prior = x.new_zeros(())
        if c.background == "controls_learned":
            # Per-cell background total nu_i stays anchored to the controls (eq. 13);
            # only its spread over genes, beta_g, is learned (Dirichlet prior).
            beta = torch.softmax(self.background_logits, dim=0)
            # background holds nu_i / G per cell (uniform share); rescale to the
            # per-cell total and spread it over genes by the learned profile.
            background = background[:, :1] * self.n_genes * beta.unsqueeze(0)
            background_prior = -(c.background_dirichlet_alpha - 1.0) * torch.log(beta).sum() / self.n_cells

        def purify(zc: Tensor, zn: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            mu, delta = self.decode(zc, zn, use_niche)
            own = (1.0 - eta).unsqueeze(1) * library.unsqueeze(1) * mu
            expected = own + incoming + background
            own_resp = own / expected.clamp_min(1e-8)
            return mu, delta, expected, own_resp, x * own_resp.clamp(0.0, 1.0)

        if c.unroll_fix:
            # Spec eqs. 19-24: K purify-and-re-encode steps, then the posterior.
            zn = self._sample(zn_mean, zn_logvar, stochastic)
            for _ in range(c.purify_steps if use_contamination else 0):
                zc = self._sample(zc_mean, zc_logvar, stochastic)
                _, _, _, _, purified = purify(zc, zn)
                h, zc_mean, zc_logvar = self.encode_intrinsic(x, purified)
            zc = self._sample(zc_mean, zc_logvar, stochastic)
            mu, delta, expected, own_resp, purified = purify(zc, zn)
        else:
            steps = c.purify_steps if use_contamination else 1
            for step in range(max(1, steps)):
                if step > 0:
                    h, zc_mean, zc_logvar = self.encode_intrinsic(purified)
                zc = self._sample(zc_mean, zc_logvar, stochastic)
                zn = self._sample(zn_mean, zn_logvar, stochastic)
                mu, delta, expected, own_resp, purified = purify(zc, zn)

        dispersion = F.softplus(self.log_dispersion).unsqueeze(0) + 1e-4
        if self.cfg.model.likelihood == "zinb":
            reconstruction = _zinb_nll(x, expected, dispersion, self.zero_inflation(zc))
        else:
            reconstruction = _nb_nll(x, expected, dispersion)
        kl_c = self.kl_intrinsic(zc_mean, zc_logvar, zc)
        identity_recon = x.new_zeros(())
        if use_niche and self.cfg.loss.identity_recon_weight > 0:
            # z_c alone, without the niche effect, must also explain the cell.
            mu_identity = torch.softmax(self.intrinsic_decoder(zc), dim=1)
            own_identity = (1.0 - eta).unsqueeze(1) * library.unsqueeze(1) * mu_identity
            identity_recon = _nb_nll(x, own_identity + incoming + background, dispersion)
        niche_zc = x.new_zeros(())
        if use_niche and self.cfg.loss.lambda_niche_zc > 0 and neighbor_zc is not None:
            niche_zc = F.mse_loss(self.niche_zc_head(zn), neighbor_zc)
        kl_n = _kl_standard_normal(zn_mean, zn_logvar) if use_niche else x.new_zeros(())
        sparse_niche = delta.abs().mean() if use_niche else x.new_zeros(())
        if use_niche and self.cfg.loss.niche_group_weight > 0:
            # Spec eq. 29: only a minority of genes should be niche-responsive.
            sparse_niche = sparse_niche + self.cfg.loss.niche_group_weight * torch.sqrt(
                delta.square().mean(dim=0) + 1e-12
            ).mean()
        niche_prediction = torch.softmax(self.niche_reconstruction(zn), dim=1)
        niche_recon = F.mse_loss(niche_prediction, neighbor_profile) if use_niche else x.new_zeros(())
        hsic = _rbf_hsic(zc_mean, zn_mean) if use_niche else x.new_zeros(())
        eta_penalty = eta.mean()
        weights = self.cfg.loss
        loss = (
            reconstruction
            + beta_intrinsic * kl_c
            + weights.beta_niche * kl_n
            + weights.lambda_niche_sparse * sparse_niche
            + weights.lambda_niche_reconstruction * niche_recon
            + weights.lambda_hsic * hsic
            + weights.lambda_eta * eta_penalty
            + library_weight * library_penalty
            + background_prior
            + self.prior_weight_penalty()
            + weights.identity_recon_weight * identity_recon
            + weights.lambda_niche_zc * niche_zc
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
        if c.kernel == "learned":
            metrics.update(
                tau=float(torch.exp(self.log_tau).detach()),
                alpha_boundary=float(self.alpha_boundary.detach()),
                alpha_size=float(self.alpha_size.detach()),
            )
        return BatchResult(
            loss, metrics, zc_mean, zn_mean, mu, purified, own_resp, eta, delta, h, library, expected
        )
