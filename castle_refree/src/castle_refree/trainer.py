from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import Tensor

from .config import CastleConfig
from .data import SpatialCountData
from .graph import SpatialGraph, build_spatial_graph, geometry_features
from .model import BatchResult, CastleModel


@dataclass
class ModelCache:
    hidden: np.ndarray
    mu: np.ndarray
    zc: np.ndarray
    zn: np.ndarray
    library: np.ndarray
    eta: np.ndarray
    edge_weights: np.ndarray


class CastleTrainer:
    """Alternating cached-neighbor trainer suitable for a 16 GB workstation.

    Donor states are refreshed between epochs and treated as stop-gradient values
    inside an epoch. This is the usual scalable approximation for a graph-coupled
    likelihood; target-cell paths remain fully differentiable.
    """

    def __init__(self, data: SpatialCountData, cfg: CastleConfig) -> None:
        self.data = data
        self.cfg = cfg
        self._set_seed(cfg.train.seed)
        if cfg.train.num_threads > 0:
            torch.set_num_threads(cfg.train.num_threads)
        self.device = self._select_device(cfg.train.device)
        self.graph = build_spatial_graph(
            data.coordinates, cfg.graph.n_neighbors, cfg.graph.distance_quantile
        )
        self.geometry = geometry_features(self.graph, data.coordinates, data.areas)
        totals = np.asarray(data.counts.sum(axis=1)).ravel()
        self.median_total = float(np.median(totals[totals > 0]))
        if float(np.std(np.log(np.maximum(data.areas, 1e-6)))) > 1e-3:
            area_scale = self.median_total / max(float(np.median(data.areas)), 1e-6)
            library_prior = data.areas * area_scale
        else:
            # The compact object may not carry cell areas. In that case use a
            # weak observed-total anchor rather than pretending every cell has
            # identical true transcript content.
            library_prior = np.maximum(totals, 1.0)
        self.library_prior_log = np.log(np.maximum(library_prior, 1.0)).astype(np.float32)
        self.model = CastleModel(
            data.n_cells, data.n_genes, self.geometry.shape[1], self.median_total, cfg
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
        # Negative controls are optional. With the processed STexampleData object
        # they are unavailable, so background is deliberately zero instead of a
        # free term that could absorb biology.
        if self.data.negative_control_counts is None:
            return np.zeros((self.data.n_cells, self.data.n_genes), dtype=np.float32)
        # If supplied, this column is interpreted as the already calibrated
        # expected total background per cell. It is spread uniformly because a
        # panel-specific background profile cannot be identified from totals.
        expected_total = np.maximum(
            self.data.negative_control_counts.astype(np.float32), 0.0
        )
        return np.repeat(
            (expected_total / self.data.n_genes)[:, None], self.data.n_genes, axis=1
        ).astype(np.float32)

    def _dense_counts(self, ids: np.ndarray) -> np.ndarray:
        return self.data.counts[ids].toarray().astype(np.float32, copy=False)

    def _niche_context(self, ids: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        neighbors = self.graph.neighbors[ids]
        k = neighbors.shape[1]
        cuts = [max(1, int(round(k * f))) for f in (1 / 3, 2 / 3, 1.0)]
        blocks = [hidden[neighbors[:, :cut]].mean(axis=1) for cut in cuts]
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def _neighbor_profile(self, ids: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return mu[self.graph.neighbors[ids]].mean(axis=1).astype(np.float32)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    def refresh_cache(self) -> ModelCache:
        self.model.eval()
        n, g = self.data.n_cells, self.data.n_genes
        hdim = self.cfg.model.hidden_dim
        zcdim = self.cfg.model.intrinsic_dim
        zndim = self.cfg.model.niche_dim
        hidden = np.empty((n, hdim), dtype=np.float32)
        zc = np.empty((n, zcdim), dtype=np.float32)
        batch_size = self.cfg.train.cache_batch_size
        with torch.no_grad():
            for start in range(0, n, batch_size):
                ids = np.arange(start, min(start + batch_size, n))
                x = torch.from_numpy(self._dense_counts(ids)).to(self.device)
                h, mean, _ = self.model.encode_intrinsic(x)
                hidden[ids] = h.cpu().numpy()
                zc[ids] = mean.cpu().numpy()

        zn = np.empty((n, zndim), dtype=np.float32)
        mu = np.empty((n, g), dtype=np.float32)
        phase = str(self.history[-1].get("phase")) if self.history else "warmup"
        use_niche = phase == "joint"
        with torch.no_grad():
            for start in range(0, n, batch_size):
                ids = np.arange(start, min(start + batch_size, n))
                context = torch.from_numpy(self._niche_context(ids, hidden)).to(self.device)
                _, zn_mean, _ = self.model.encode_niche(context)
                zc_tensor = torch.from_numpy(zc[ids]).to(self.device)
                profile, _ = self.model.decode(zc_tensor, zn_mean, use_niche)
                zn[ids] = zn_mean.cpu().numpy()
                mu[ids] = profile.cpu().numpy()

            all_ids = torch.arange(n, device=self.device)
            library = torch.exp(self.model.log_library(all_ids).squeeze(1)).cpu().numpy()
            geom = torch.from_numpy(self.geometry).to(self.device)
            eta = torch.sigmoid(self.model.eta_net(geom)).squeeze(1)
            eta = (eta * self.cfg.model.contamination.eta_max).cpu().numpy()

        edge_weights = self._edge_weights(mu)
        provisional = ModelCache(hidden, mu, zc, zn, library, eta, edge_weights)
        self.cache = provisional

        # The spatial context must be built from purified, not raw, cell states.
        # Perform a stop-gradient fixed-point refresh after the warm-up phase.
        if phase != "warmup" and self.cfg.model.contamination.enabled:
            purified_hidden = np.empty_like(hidden)
            purified_zc = np.empty_like(zc)
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
                    h, mean, _ = self.model.encode_intrinsic(result.purified)
                    purified_hidden[ids] = h.cpu().numpy()
                    purified_zc[ids] = mean.cpu().numpy()
            hidden = purified_hidden
            zc = purified_zc
            with torch.no_grad():
                for start in range(0, n, batch_size):
                    ids = np.arange(start, min(start + batch_size, n))
                    context = torch.from_numpy(self._niche_context(ids, hidden)).to(self.device)
                    _, zn_mean, _ = self.model.encode_niche(context)
                    zc_tensor = torch.from_numpy(zc[ids]).to(self.device)
                    profile, _ = self.model.decode(zc_tensor, zn_mean, use_niche)
                    zn[ids] = zn_mean.cpu().numpy()
                    mu[ids] = profile.cpu().numpy()
            edge_weights = self._edge_weights(mu)

        self.cache = ModelCache(hidden, mu, zc, zn, library, eta, edge_weights)
        return self.cache

    def _edge_weights(self, mu: np.ndarray) -> np.ndarray:
        distances = self.graph.distances
        scale = self.cfg.graph.distance_scale
        if scale <= 0:
            scale = float(np.median(distances[distances > 0]))
        logits = -distances / max(scale, 1e-6)
        e = self.cfg.model.contamination.expression_guided
        if e.enabled and self.cfg.model.contamination.enabled:
            residual_emb = np.empty((self.data.n_cells, e.projection_dim), dtype=np.float32)
            donor_emb = self._l2_normalize(mu @ self.projection)
            batch_size = self.cfg.train.cache_batch_size
            for start in range(0, self.data.n_cells, batch_size):
                ids = np.arange(start, min(start + batch_size, self.data.n_cells))
                x = self._dense_counts(ids)
                observed = x / np.maximum(x.sum(axis=1, keepdims=True), 1.0)
                residual = np.maximum(observed - mu[ids], 0.0)
                residual_emb[ids] = self._l2_normalize(residual @ self.projection)
            score = np.sum(
                donor_emb[:, None, :] * residual_emb[self.graph.neighbors], axis=2
            )
            score = np.clip(score, 0.0, 1.0)
            shift = np.clip(e.strength * score, 0.0, e.max_logit_shift)
            logits = logits + shift
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        return weights.astype(np.float32).reshape(-1)

    def _batch_tensors(self, ids: np.ndarray) -> dict[str, Tensor]:
        assert self.cache is not None
        edge_ids, local_targets = self.graph.incoming_edges(ids)
        donors = self.graph.source[edge_ids]
        return {
            "x": torch.from_numpy(self._dense_counts(ids)).to(self.device),
            "cell_ids": torch.from_numpy(ids).long().to(self.device),
            "geometry": torch.from_numpy(self.geometry[ids]).to(self.device),
            "niche_context": torch.from_numpy(self._niche_context(ids, self.cache.hidden)).to(self.device),
            "neighbor_profile": torch.from_numpy(self._neighbor_profile(ids, self.cache.mu)).to(self.device),
            "local_targets": torch.from_numpy(local_targets).long().to(self.device),
            "donor_mu": torch.from_numpy(self.cache.mu[donors]).to(self.device),
            "donor_library": torch.from_numpy(self.cache.library[donors]).to(self.device),
            "donor_eta": torch.from_numpy(self.cache.eta[donors]).to(self.device),
            "edge_weight": torch.from_numpy(self.cache.edge_weights[edge_ids]).to(self.device),
            "background": torch.from_numpy(self.background[ids]).to(self.device),
            "library_prior_log": torch.from_numpy(self.library_prior_log[ids]).to(self.device),
        }

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
        ids_all = np.arange(self.data.n_cells)
        self.refresh_cache()
        for phase, epochs in phases:
            for _ in range(completed[phase], epochs):
                self.model.train()
                np.random.shuffle(ids_all)
                sums: dict[str, float] = {}
                batches = 0
                beta = self._beta_intrinsic(joint_epoch) if phase == "joint" else self.cfg.loss.beta_intrinsic_start
                for start in range(0, self.data.n_cells, self.cfg.train.batch_size):
                    ids = ids_all[start : start + self.cfg.train.batch_size]
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
                global_epoch += 1
                if phase == "joint":
                    joint_epoch += 1
                row: dict[str, float | int | str] = {
                    "epoch": global_epoch,
                    "phase": phase,
                    "beta_intrinsic": beta,
                }
                row.update({key: value / max(batches, 1) for key, value in sums.items()})
                self.history.append(row)
                print(json.dumps(row), flush=True)
                self.refresh_cache()
                if global_epoch % self.cfg.train.checkpoint_every == 0:
                    self.save_checkpoint(Path(self.cfg.data.output_dir) / "checkpoint.pt")
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
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.history = list(checkpoint.get("history", []))
        self.refresh_cache()
        return self

    def _infer_batch(self, ids: np.ndarray) -> BatchResult:
        tensors = self._batch_tensors(ids)
        return self.model.forward_batch(
            **tensors,
            phase="joint",
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
        niche_effect_l1 = np.empty(self.data.n_cells, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, self.data.n_cells, self.cfg.train.cache_batch_size):
                ids = np.arange(start, min(start + self.cfg.train.cache_batch_size, self.data.n_cells))
                result = self._infer_batch(ids)
                purified_blocks.append(sparse.csr_matrix(result.purified.cpu().numpy()))
                responsibility_blocks.append(sparse.csr_matrix(result.own_responsibility.cpu().numpy()))
                eta[ids] = result.eta.cpu().numpy()
                niche_effect_l1[ids] = result.niche_effect.abs().mean(1).cpu().numpy()

        sparse.save_npz(output / "purified_counts_cells_by_genes.npz", sparse.vstack(purified_blocks).tocsr())
        sparse.save_npz(output / "own_responsibility_cells_by_genes.npz", sparse.vstack(responsibility_blocks).tocsr())
        np.savez_compressed(
            output / "castle_embeddings_and_qc.npz",
            z_intrinsic=self.cache.zc,
            z_niche=self.cache.zn,
            eta=eta,
            segmentation_confidence=np.asarray(
                sparse.vstack(purified_blocks).sum(axis=1)
            ).ravel() / np.maximum(np.asarray(self.data.counts.sum(axis=1)).ravel(), 1.0),
            niche_effect_l1=niche_effect_l1,
        )
        np.savetxt(output / "cells.tsv", self.data.cell_ids, fmt="%s", delimiter="\t")
        np.savetxt(output / "genes.tsv", self.data.gene_ids, fmt="%s", delimiter="\t")
        with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history[0]) if self.history else ["epoch"])
            writer.writeheader()
            writer.writerows(self.history)
        with (output / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.cfg.to_dict(), handle, indent=2)
        self.save_checkpoint(output / "castle_refree_model.pt")
        return output
