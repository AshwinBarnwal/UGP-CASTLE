from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import Tensor
from tqdm.auto import tqdm

from .config import CastleConfig
from .data import SpatialCountData
from .graph import (
    SpilloverGraph,
    build_spatial_graph,
    contact_spillover_graph,
    contact_step_neighbours,
    full_geometry_features,
    geometry_features,
    knn_spillover_graph,
)
from .model import BatchResult, CastleModel


@dataclass
class ModelCache:
    hidden: np.ndarray
    mu: np.ndarray
    zc: np.ndarray
    zn: np.ndarray
    library: np.ndarray
    eta: np.ndarray
    edge_weights: np.ndarray  # per spillover edge, normalised over each donor
    edge_shift: np.ndarray  # expression-guided logit shift per spillover edge
    purified: np.ndarray  # cells x genes; raw counts before the contamination phase


class CastleTrainer:
    """Alternating cached-neighbor trainer suitable for a 16 GB workstation.

    Donor states are refreshed between epochs and treated as stop-gradient values
    inside an epoch. This is the usual scalable approximation for a graph-coupled
    likelihood; target-cell paths remain fully differentiable. With
    ``donor_gradient`` the donors' leakage fractions (and, with the learned kernel,
    the transfer weights) are recomputed live from geometry, so the leakage model
    also receives gradients from the cells that the leaked transcripts reach.
    """

    def __init__(self, data: SpatialCountData, cfg: CastleConfig) -> None:
        self.data = data
        self.cfg = cfg
        c = cfg.model.contamination
        self._set_seed(cfg.train.seed)
        if cfg.train.num_threads > 0:
            torch.set_num_threads(cfg.train.num_threads)
        self.device = self._select_device(cfg.train.device)
        # The kNN graph always drives the niche encoder.
        self.graph = build_spatial_graph(
            data.coordinates, cfg.graph.n_neighbors, cfg.graph.distance_quantile
        )
        self.spill = self._build_spillover_graph()
        self.has_out = self.spill.has_out_edges
        self.edge_features = np.column_stack(
            [self.spill.distance, self.spill.boundary_fraction, self.spill.size_share]
        ).astype(np.float32)
        if c.geometry_features in ("full", "spec"):
            if data.geometry is None:
                raise ValueError(f"geometry_features={c.geometry_features!r} needs geometry.csv in the input bundle")
            self.geometry = full_geometry_features(
                data.geometry, transcript_features=c.geometry_features == "spec"
            )
        else:
            self.geometry = geometry_features(self.graph, data.coordinates, data.areas)

        totals = np.asarray(data.counts.sum(axis=1)).ravel()
        self.median_total = float(np.median(totals[totals > 0]))
        areas_vary = float(np.std(np.log(np.maximum(data.areas, 1e-6)))) > 1e-3
        log_kappa_init = 0.0
        if c.library == "amortized":
            # Prior log l ~ N(log kappa + log area, sd^2); kappa is learned.
            if areas_vary:
                self.library_prior_log = np.log(data.areas).astype(np.float32)
                log_kappa_init = float(np.log(self.median_total / np.median(data.areas)))
            else:
                # No areas: an empirical prior over log totals, as in scVI.
                self.library_prior_log = np.zeros(data.n_cells, dtype=np.float32)
                log_totals = np.log(np.maximum(totals, 1.0))
                log_kappa_init = float(log_totals.mean())
                c.library_prior_sd = float(log_totals.std())
        else:
            if areas_vary:
                area_scale = self.median_total / max(float(np.median(data.areas)), 1e-6)
                library_prior = data.areas * area_scale
            else:
                # The compact object may not carry cell areas. In that case use a
                # weak observed-total anchor rather than pretending every cell has
                # identical true transcript content.
                library_prior = np.maximum(totals, 1.0)
            self.library_prior_log = np.log(np.maximum(library_prior, 1.0)).astype(np.float32)

        positive = self.spill.distance[self.spill.distance > 0]
        self.distance_scale = (
            cfg.graph.distance_scale if cfg.graph.distance_scale > 0
            else float(np.median(positive)) if positive.size else 1.0
        )
        self.step_neighbours = None
        if cfg.model.niche_encoder == "contact_steps":
            if data.contact_edges is None:
                raise ValueError("niche_encoder='contact_steps' needs edges_contact.csv in the input bundle")
            self.step_neighbours = contact_step_neighbours(
                data.contact_edges, data.coordinates, data.n_cells, cfg.model.niche_steps
            )
        self._raw_dense: np.ndarray | None = None
        self.model = CastleModel(
            data.n_cells, data.n_genes, self.geometry.shape[1], self.median_total, cfg,
            median_edge_distance=self.distance_scale, log_kappa_init=log_kappa_init,
        ).to(self.device)
        with torch.no_grad():
            self.model.log_library.weight[:, 0].copy_(
                torch.from_numpy(np.log(np.maximum(totals, 1.0))).to(self.device)
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.projection = self._make_projection()
        self.background = self._estimate_background()
        self.cache: ModelCache | None = None
        self.history: list[dict[str, float | int | str]] = []
        # Fixed per-cell leakage fractions (null calibration); None = learned.
        self.eta_override: np.ndarray | None = None
        self.active_phase = "warmup"  # phase whose batches are being built

    def _build_spillover_graph(self) -> SpilloverGraph:
        mode = self.cfg.model.contamination.spillover_graph
        if mode == "contact":
            if self.data.contact_edges is None:
                raise ValueError("spillover_graph='contact' needs edges_contact.csv in the input bundle")
            return contact_spillover_graph(self.data.contact_edges, self.data.areas, self.data.n_cells)
        if mode == "knn":
            return knn_spillover_graph(self.graph, self.data.areas)
        raise ValueError(f"Unknown spillover_graph {mode!r}")

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    def _make_projection(self) -> np.ndarray:
        e = self.cfg.model.contamination.expression_guided
        rng = np.random.default_rng(e.seed)
        matrix = rng.normal(size=(self.data.n_genes, e.projection_dim)).astype(np.float32)
        matrix /= np.sqrt(max(e.projection_dim, 1))
        return matrix

    def _estimate_background(self) -> np.ndarray:
        """Per-cell background per gene, one column (cells x 1).

        One value per cell is broadcast across genes in the likelihood, which
        avoids a dense cells x genes allocation on high-plex panels. The value is
        the uniform per-gene share nu_i / G; the learned-profile mode rescales it
        by G * beta_g inside the model.
        """
        mode = self.cfg.model.contamination.background
        zeros = np.zeros((self.data.n_cells, 1), dtype=np.float32)
        if mode == "none":
            return zeros
        if mode in ("controls", "controls_learned"):
            controls = self.data.control_counts
            n_controls = self.data.n_control_features
            if controls is None or n_controls == 0:
                raise ValueError("background='controls' needs control counts and control_features.tsv")
            # CASTLE eq. 13: background per unit area, rescaled from control to
            # real probes; spread uniformly over genes (beta = uniform).
            kappa = self.data.n_genes / n_controls * controls.sum() / self.data.areas.sum()
            nu = kappa * self.data.areas
            return (nu / self.data.n_genes)[:, None].astype(np.float32)
        # "auto": legacy behaviour. Negative controls are optional; with the
        # processed STexampleData object they are unavailable, so background is
        # deliberately zero instead of a free term that could absorb biology.
        if self.data.negative_control_counts is None:
            return zeros
        # If supplied, this column is interpreted as the already calibrated
        # expected total background per cell. It is spread uniformly because a
        # panel-specific background profile cannot be identified from totals.
        expected_total = np.maximum(
            self.data.negative_control_counts.astype(np.float32), 0.0
        )
        return (expected_total / self.data.n_genes)[:, None].astype(np.float32)

    def _dense_counts(self, ids: np.ndarray) -> np.ndarray:
        return self.data.counts[ids].toarray().astype(np.float32, copy=False)

    def _niche_context(self, ids: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        neighbors = self.graph.neighbors[ids]
        k = neighbors.shape[1]
        cuts = [max(1, int(round(k * f))) for f in (1 / 3, 2 / 3, 1.0)]
        blocks = [hidden[neighbors[:, :cut]].mean(axis=1) for cut in cuts]
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def _all_counts(self) -> np.ndarray:
        if self._raw_dense is None:
            self._raw_dense = self.data.counts.toarray().astype(np.float32)
        return self._raw_dense

    def _niche_needed(self) -> bool:
        """Contact-step niche inputs are costly; build them only in the niche phase."""
        return self.step_neighbours is None or self.active_phase == "joint"

    def _niche_input(self, ids: np.ndarray, hidden: np.ndarray, zc: np.ndarray | None = None):
        """Niche encoder input: averaged context, or per-step neighbour tensors.

        With ``niche_input='zc'`` neighbours are described by their (fixed) z_c.
        """
        if self.cfg.model.niche_input == "zc" and zc is not None:
            hidden = zc
        if self.step_neighbours is None:
            return self._tensor(self._niche_context(ids, hidden))
        scales = []
        for scale in self.step_neighbours:
            index = scale.index[ids]
            mask = index >= 0
            gathered = hidden[np.where(mask, index, 0)] * mask[..., None]
            scales.append({
                "hidden": self._tensor(gathered.astype(np.float32)),
                "mask": self._tensor(mask),
                "share": self._tensor(scale.share[ids]),
                "distance": self._tensor((scale.distance[ids] / max(self.distance_scale, 1e-6)).astype(np.float32)),
                "steps": self._tensor((scale.steps[ids] / self.cfg.model.niche_steps).astype(np.float32)),
                "angle": self._tensor(scale.angle[ids]),
            })
        return scales

    def _neighbor_zc(self, ids: np.ndarray) -> np.ndarray:
        """Mean z_c of the directly touching cells (k-NN without a contact graph)."""
        zc = self.cache.zc
        if self.step_neighbours is not None:
            index = self.step_neighbours[0].index[ids]
            mask = index >= 0
            summed = (zc[np.where(mask, index, 0)] * mask[..., None]).sum(axis=1)
            return (summed / np.maximum(mask.sum(axis=1, keepdims=True), 1)).astype(np.float32)
        return zc[self.graph.neighbors[ids]].mean(axis=1).astype(np.float32)

    def _neighbor_profile(self, ids: np.ndarray, mu: np.ndarray) -> np.ndarray:
        if self.cfg.loss.niche_target == "purified" and self.cache is not None:
            # Spec eq. 31: the neighbours' aggregate purified profile, here over
            # the directly touching cells (k-NN when there is no contact graph).
            profile = self.cache.purified / np.maximum(self.cache.purified.sum(axis=1, keepdims=True), 1e-9)
            if self.step_neighbours is not None:
                index = self.step_neighbours[0].index[ids]
                mask = index >= 0
                summed = (profile[np.where(mask, index, 0)] * mask[..., None]).sum(axis=1)
                return (summed / np.maximum(mask.sum(axis=1, keepdims=True), 1)).astype(np.float32)
            return profile[self.graph.neighbors[ids]].mean(axis=1).astype(np.float32)
        return mu[self.graph.neighbors[ids]].mean(axis=1).astype(np.float32)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    def _tensor(self, values: np.ndarray, dtype: torch.dtype | None = None) -> Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(values))
        return (tensor.to(dtype) if dtype is not None else tensor).to(self.device)

    def refresh_cache(self) -> ModelCache:
        self.model.eval()
        c = self.cfg.model.contamination
        n, g = self.data.n_cells, self.data.n_genes
        hdim = self.cfg.model.hidden_dim
        zcdim = self.cfg.model.intrinsic_dim
        zndim = self.cfg.model.niche_dim
        hidden = np.empty((n, hdim), dtype=np.float32)
        zc = np.empty((n, zcdim), dtype=np.float32)
        library = np.empty(n, dtype=np.float32)
        batch_size = self.cfg.train.cache_batch_size
        with torch.no_grad():
            for start in range(0, n, batch_size):
                ids = np.arange(start, min(start + batch_size, n))
                x = torch.from_numpy(self._dense_counts(ids)).to(self.device)
                h, mean, _ = self.model.encode_intrinsic(x)
                hidden[ids] = h.cpu().numpy()
                zc[ids] = mean.cpu().numpy()
                if c.library == "amortized":
                    library[ids] = torch.exp(self.model.library_mean(h).squeeze(1)).clamp(1.0, 1e7).cpu().numpy()

        zn = np.empty((n, zndim), dtype=np.float32)
        mu = np.empty((n, g), dtype=np.float32)
        phase = str(self.history[-1].get("phase")) if self.history else "warmup"
        use_niche = phase == "joint"

        self.active_phase = phase

        def decode_all() -> None:
            with torch.no_grad():
                for start in range(0, n, batch_size):
                    ids = np.arange(start, min(start + batch_size, n))
                    if self._niche_needed():
                        _, zn_mean, _ = self.model.encode_niche(self._niche_input(ids, hidden, zc))
                    else:
                        zn_mean = torch.zeros((ids.size, self.cfg.model.niche_dim), device=self.device)
                    zc_tensor = torch.from_numpy(zc[ids]).to(self.device)
                    profile, _ = self.model.decode(zc_tensor, zn_mean, use_niche)
                    zn[ids] = zn_mean.cpu().numpy()
                    mu[ids] = profile.cpu().numpy()

        decode_all()
        with torch.no_grad():
            if c.library != "amortized":
                all_ids = torch.arange(n, device=self.device)
                library = torch.exp(self.model.log_library(all_ids).squeeze(1)).cpu().numpy()
            eta = self.model.eta_from_geometry(
                self._tensor(self.geometry), self._tensor(self.has_out)
            ).cpu().numpy()
        if self.eta_override is not None:
            eta = (self.eta_override * self.has_out).astype(np.float32)

        edge_shift = self._edge_shift(mu)
        edge_weights = self._edge_weights(edge_shift)
        purified_counts = self._all_counts() if self.cache is None else self.cache.purified
        self.cache = ModelCache(hidden, mu, zc, zn, library, eta, edge_weights, edge_shift, purified_counts)

        # The spatial context must be built from purified, not raw, cell states.
        # Perform a stop-gradient fixed-point refresh after the warm-up phase.
        if phase != "warmup" and c.enabled:
            purified_hidden = np.empty_like(hidden)
            purified_zc = np.empty_like(zc)
            purified_counts = np.empty((n, g), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, n, batch_size):
                    ids = np.arange(start, min(start + batch_size, n))
                    tensors = self._batch_tensors(ids)
                    result = self.model.forward_batch(
                        **tensors,
                        phase=phase,
                        beta_intrinsic=self.cfg.loss.beta_intrinsic_end,
                        stochastic=False,
                    )
                    if c.unroll_fix:
                        # The trained posterior itself, not one extra encoding pass.
                        h, mean = result.hidden, result.z_intrinsic
                    else:
                        h, mean, _ = self.model.encode_intrinsic(result.purified)
                    purified_hidden[ids] = h.cpu().numpy()
                    purified_zc[ids] = mean.cpu().numpy()
                    purified_counts[ids] = result.purified.cpu().numpy()
            hidden = purified_hidden
            zc = purified_zc
            decode_all()
            edge_shift = self._edge_shift(mu)
            edge_weights = self._edge_weights(edge_shift)

        self.cache = ModelCache(hidden, mu, zc, zn, library, eta, edge_weights, edge_shift, purified_counts)
        return self.cache

    def _edge_shift(self, mu: np.ndarray) -> np.ndarray:
        """Expression-guided logit shift per spillover edge (zero when disabled)."""
        e = self.cfg.model.contamination.expression_guided
        shift = np.zeros(self.spill.source.size, dtype=np.float32)
        if not (e.enabled and self.cfg.model.contamination.enabled):
            return shift
        residual_emb = np.empty((self.data.n_cells, e.projection_dim), dtype=np.float32)
        donor_emb = self._l2_normalize(mu @ self.projection)
        batch_size = self.cfg.train.cache_batch_size
        for start in range(0, self.data.n_cells, batch_size):
            ids = np.arange(start, min(start + batch_size, self.data.n_cells))
            x = self._dense_counts(ids)
            observed = x / np.maximum(x.sum(axis=1, keepdims=True), 1.0)
            residual = np.maximum(observed - mu[ids], 0.0)
            residual_emb[ids] = self._l2_normalize(residual @ self.projection)
        score = np.sum(donor_emb[self.spill.source] * residual_emb[self.spill.target], axis=1)
        score = np.clip(score, 0.0, 1.0)
        return np.clip(e.strength * score, 0.0, e.max_logit_shift).astype(np.float32)

    def _edge_weights(self, edge_shift: np.ndarray) -> np.ndarray:
        """Softmax of edge logits over each donor's out-edges (mass conservation)."""
        if self.cfg.model.contamination.kernel == "learned":
            with torch.no_grad():
                logits = self.model.edge_logits(
                    self._tensor(self.edge_features), self._tensor(edge_shift)
                ).cpu().numpy()
        else:
            logits = -self.spill.distance / max(self.distance_scale, 1e-6) + edge_shift
        if logits.size == 0:
            return logits.astype(np.float32)
        # Edges are sorted by donor, so each donor's out-edges form one segment.
        starts = self.spill.out_indptr[:-1][self.has_out]
        sizes = np.diff(self.spill.out_indptr)[self.has_out]
        expo = np.exp(logits - np.repeat(np.maximum.reduceat(logits, starts), sizes))
        return (expo / np.repeat(np.add.reduceat(expo, starts), sizes)).astype(np.float32)

    def _batch_tensors(self, ids: np.ndarray) -> dict[str, Tensor]:
        assert self.cache is not None
        c = self.cfg.model.contamination
        in_edges, local_targets = self.spill.incoming_edges(ids)
        donors, in_donor = np.unique(self.spill.source[in_edges], return_inverse=True)
        if c.kernel == "learned":
            out_edges, out_donor = self.spill.out_edges(donors)
        else:
            out_edges = out_donor = np.empty(0, dtype=np.int64)
        tensors = {
            "x": self._tensor(self._dense_counts(ids)),
            "cell_ids": self._tensor(ids, torch.long),
            "geometry": self._tensor(self.geometry[ids]),
            "own_has_out": self._tensor(self.has_out[ids]),
            "niche_context": self._niche_input(ids, self.cache.hidden, self.cache.zc) if self._niche_needed() else None,
            "neighbor_profile": self._tensor(self._neighbor_profile(ids, self.cache.mu)),
            "local_targets": self._tensor(local_targets, torch.long),
            "in_donor": self._tensor(in_donor.reshape(-1), torch.long),
            "in_features": self._tensor(self.edge_features[in_edges]),
            "in_shift": self._tensor(self.cache.edge_shift[in_edges]),
            "in_weight": self._tensor(self.cache.edge_weights[in_edges]),
            "donor_geometry": self._tensor(self.geometry[donors]),
            "donor_has_out": self._tensor(self.has_out[donors]),
            "donor_mu": self._tensor(self.cache.mu[donors]),
            "donor_library": self._tensor(self.cache.library[donors]),
            "donor_eta": self._tensor(self.cache.eta[donors]),
            "out_donor": self._tensor(out_donor, torch.long),
            "out_features": self._tensor(self.edge_features[out_edges]),
            "out_shift": self._tensor(self.cache.edge_shift[out_edges]),
            "background": self._tensor(self.background[ids]),
            "library_prior_log": self._tensor(self.library_prior_log[ids]),
        }
        if self.eta_override is not None:
            tensors["own_eta"] = self._tensor(self.cache.eta[ids])
        if self.cfg.loss.lambda_niche_zc > 0:
            tensors["neighbor_zc"] = self._tensor(self._neighbor_zc(ids))
        if c.donor_live_profiles:
            tensors["donor_purified"] = self._tensor(self.cache.purified[donors])
            tensors["donor_zn"] = self._tensor(self.cache.zn[donors])
        return tensors

    def _kl_weight(self, phase: str, global_epoch: int, joint_epoch: int) -> float:
        loss = self.cfg.loss
        if loss.kl_warmup_epochs > 0:
            return loss.kl_max * min(1.0, global_epoch / loss.kl_warmup_epochs)
        return self._beta_intrinsic(joint_epoch) if phase == "joint" else loss.beta_intrinsic_start

    def _freeze_spillover(self) -> None:
        """Fix the leakage network and transfer kernel (niche phase)."""
        modules = [self.model.eta_net]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for name in ("log_tau", "alpha_boundary", "alpha_size"):
            if hasattr(self.model, name):
                getattr(self.model, name).requires_grad_(False)

    def validation_loss(self, phase: str) -> float:
        """Deterministic ELBO terms (reconstruction + KL) on the held-out cells."""
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, self.validation_ids.size, self.cfg.train.cache_batch_size):
                ids = self.validation_ids[start : start + self.cfg.train.cache_batch_size]
                result = self.model.forward_batch(
                    **self._batch_tensors(ids), phase=phase, beta_intrinsic=1.0, stochastic=False
                )
                total += (result.metrics["reconstruction"] + result.metrics["kl_intrinsic"]) * len(ids)
                count += len(ids)
        return total / max(count, 1)

    def _beta_intrinsic(self, joint_epoch: int) -> float:
        start = self.cfg.loss.beta_intrinsic_start
        end = self.cfg.loss.beta_intrinsic_end
        total = max(self.cfg.train.joint_epochs - 1, 1)
        return start + (end - start) * min(max(joint_epoch / total, 0.0), 1.0)

    def fit(self) -> "CastleTrainer":
        phases = (
            [("warmup", self.cfg.train.warmup_epochs)]
            + [("contamination", self.cfg.train.contamination_epochs)]
            + [("joint", self.cfg.train.joint_epochs)]
        )
        global_epoch = len(self.history)
        joint_epoch = sum(row.get("phase") == "joint" for row in self.history)
        completed = {
            phase: sum(row.get("phase") == phase for row in self.history)
            for phase, _ in phases
        }
        train = self.cfg.train
        rng = np.random.default_rng(train.seed + 1)
        n_validation = int(round(train.validation_fraction * self.data.n_cells))
        order = rng.permutation(self.data.n_cells)
        self.validation_ids = np.sort(order[:n_validation])
        ids_all = np.sort(order[n_validation:])
        self.refresh_cache()
        batches_per_epoch = (self.data.n_cells + self.cfg.train.batch_size - 1) // self.cfg.train.batch_size
        remaining_epochs = sum(epochs - completed[phase] for phase, epochs in phases)
        total_epochs = sum(epochs for _, epochs in phases)
        progress = tqdm(
            total=batches_per_epoch * remaining_epochs,
            desc="Training",
            unit="batch",
            dynamic_ncols=True,
        )
        for phase, epochs in phases:
            if phase == "joint" and epochs > 0 and self.cfg.model.contamination.freeze_in_joint:
                self._freeze_spillover()
            early = train.early_stop_phase == phase and n_validation > 0
            best_loss, best_state, stale, phase_epoch = np.inf, None, 0, 0
            for _ in range(completed[phase], epochs):
                self.active_phase = phase
                progress.set_description(f"{phase} epoch {global_epoch + 1}/{total_epochs}")
                self.model.train()
                np.random.shuffle(ids_all)
                sums: dict[str, float] = {}
                batches = 0
                beta = self._kl_weight(phase, global_epoch, joint_epoch)
                batch_starts = range(0, ids_all.size, self.cfg.train.batch_size)
                for start in batch_starts:
                    ids = ids_all[start : start + self.cfg.train.batch_size]
                    if ids.size < 2:  # batch norm needs at least two cells
                        continue
                    tensors = self._batch_tensors(ids)
                    self.optimizer.zero_grad(set_to_none=True)
                    result = self.model.forward_batch(
                        **tensors, phase=phase, beta_intrinsic=beta, stochastic=True
                    )
                    result.loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.gradient_clip)
                    self.optimizer.step()
                    for key, value in result.metrics.items():
                        sums[key] = sums.get(key, 0.0) + value
                    batches += 1
                    progress.update(1)
                global_epoch += 1
                if phase == "joint":
                    joint_epoch += 1
                row: dict[str, float | int | str] = {
                    "epoch": global_epoch,
                    "phase": phase,
                    "beta_intrinsic": beta,
                }
                row.update({key: value / max(batches, 1) for key, value in sums.items()})
                self.refresh_cache()
                phase_epoch += 1
                if early:
                    row["validation_loss"] = self.validation_loss(phase)
                    if row["validation_loss"] < best_loss - 1e-4:
                        best_loss, stale = row["validation_loss"], 0
                        best_state = copy.deepcopy(self.model.state_dict())
                    else:
                        stale += 1
                self.history.append(row)
                print(json.dumps(row), flush=True)
                if early and phase_epoch >= train.early_stop_min_epochs and stale >= train.early_stop_patience:
                    break
                if global_epoch % self.cfg.train.checkpoint_every == 0:
                    self.save_checkpoint(Path(self.cfg.data.output_dir) / "checkpoint.pt")
            if early and best_state is not None:
                # Restore the epoch with the best held-out ELBO before moving on.
                self.model.load_state_dict(best_state)
                self.history.append({"epoch": global_epoch, "phase": phase, "restored_best_validation": best_loss})
                self.refresh_cache()
        progress.close()
        return self

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg.to_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> "CastleTrainer":
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        # Checkpoints from before the identity-model options lack these entries;
        # their defaults (all genes visible, unused niche z_c head) are correct.
        optional = {"encoder_visible", "niche_zc_head.weight", "niche_zc_head.bias"}
        missing, unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
        if unexpected or set(missing) - optional:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        if "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError:
                # Older checkpoints have fewer parameters; the optimizer state is
                # only needed to resume training, not to reuse a fitted model.
                pass
        self.history = list(checkpoint.get("history", []))
        self.refresh_cache()
        return self

    def _infer_batch(self, ids: np.ndarray) -> BatchResult:
        # Use the last trained phase, so a run without the joint phase never
        # decodes through an untrained niche branch.
        phase = str(self.history[-1].get("phase")) if self.history else "joint"
        self.active_phase = phase
        tensors = self._batch_tensors(ids)
        return self.model.forward_batch(
            **tensors,
            phase=phase,
            beta_intrinsic=self.cfg.loss.beta_intrinsic_end,
            stochastic=False,
        )

    def export(self) -> Path:
        if self.cache is None:
            self.refresh_cache()
        output = Path(self.cfg.data.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        purified_blocks: list[sparse.csr_matrix] = []
        responsibility_blocks: list[sparse.csr_matrix] = []
        eta = np.empty(self.data.n_cells, dtype=np.float32)
        library = np.empty(self.data.n_cells, dtype=np.float32)
        niche_effect_l1 = np.empty(self.data.n_cells, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, self.data.n_cells, self.cfg.train.cache_batch_size):
                ids = np.arange(start, min(start + self.cfg.train.cache_batch_size, self.data.n_cells))
                result = self._infer_batch(ids)
                purified_blocks.append(sparse.csr_matrix(result.purified.cpu().numpy()))
                # A responsibility is only meaningful where an observed count
                # exists: purified = observed_count * responsibility. Storing
                # dense responsibilities for all zero cell-gene pairs scales as
                # n_cells * n_genes and is prohibitive for 6k-plex data.
                observed = self.data.counts[ids].copy().tocsr()
                observed.data.fill(1.0)
                responsibility_blocks.append(
                    observed.multiply(result.own_responsibility.cpu().numpy()).tocsr()
                )
                eta[ids] = result.eta.cpu().numpy()
                library[ids] = result.library.cpu().numpy()
                niche_effect_l1[ids] = result.niche_effect.abs().mean(1).cpu().numpy()

        purified = sparse.vstack(purified_blocks).tocsr()
        sparse.save_npz(output / "purified_counts_cells_by_genes.npz", purified)
        sparse.save_npz(output / "own_responsibility_cells_by_genes.npz", sparse.vstack(responsibility_blocks).tocsr())
        np.savez_compressed(
            output / "castle_embeddings_and_qc.npz",
            z_intrinsic=self.cache.zc,
            z_niche=self.cache.zn,
            eta=eta,
            library=library,
            has_spillover_edges=self.has_out,
            segmentation_confidence=np.asarray(purified.sum(axis=1)).ravel()
            / np.maximum(np.asarray(self.data.counts.sum(axis=1)).ravel(), 1.0),
            niche_effect_l1=niche_effect_l1,
        )
        np.savetxt(output / "cells.tsv", self.data.cell_ids, fmt="%s", delimiter="\t")
        np.savetxt(output / "genes.tsv", self.data.gene_ids, fmt="%s", delimiter="\t")
        with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(dict.fromkeys(key for row in self.history for key in row)) or ["epoch"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        with (output / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.cfg.to_dict(), handle, indent=2)
        self.save_checkpoint(output / "castle_refree_model.pt")
        return output
