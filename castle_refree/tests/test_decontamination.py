from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from castle_refree.config import CastleConfig
from castle_refree.data import SpatialCountData
from castle_refree.trainer import CastleTrainer


def _grid_data(seed: int = 3) -> SpatialCountData:
    """A 6 x 4 grid of touching cells in two striped types, with raw-bundle extras."""
    rng = np.random.default_rng(seed)
    nx, ny = 6, 4
    xs, ys = np.meshgrid(np.arange(nx) * 10.0, np.arange(ny) * 10.0)
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    n = len(coords)
    labels = (np.arange(n) % nx) // 3
    profiles = np.array([[10, 8, 6, 1, 1, 1, 1, 1], [1, 1, 1, 1, 6, 8, 10, 1]], dtype=np.float32)
    counts = rng.poisson(profiles[labels]).astype(np.float32)
    edges = [(k, k + 1) for k in range(n) if (k % nx) < nx - 1] + [(k, k + nx) for k in range(n - nx)]
    contact = pd.DataFrame(
        {
            "i": [a for a, _ in edges],
            "j": [b for _, b in edges],
            "shared_boundary_um": rng.uniform(5, 15, len(edges)),
            "distance_um": np.full(len(edges), 10.0),
        }
    )
    geometry = pd.DataFrame(
        {
            "area_um2": rng.uniform(80, 200, n),
            "nucleus_ratio": rng.uniform(0.1, 0.5, n),
            "circularity": rng.uniform(0.5, 0.9, n),
            "solidity": rng.uniform(0.8, 1.0, n),
            "contact_degree": np.bincount(contact[["i", "j"]].to_numpy().ravel(), minlength=n),
            "contact_boundary_um": rng.uniform(10, 40, n),
            "contact_boundary_fraction": rng.uniform(0.3, 1.0, n),
            "mean_contact_distance_um": np.full(n, 10.0),
            "min_contact_distance_um": np.full(n, 10.0),
            "distance_to_tissue_edge_um": rng.uniform(0, 30, n),
            "z_extent": rng.uniform(3, 5, n),
            "outer_fraction": rng.uniform(0.0, 0.2, n),
        }
    )
    return SpatialCountData(
        sparse.csr_matrix(counts), coords, geometry["area_um2"].to_numpy(dtype=np.float32),
        np.array([f"c{i}" for i in range(n)]), np.array([f"g{i}" for i in range(8)]),
        contact_edges=contact, geometry=geometry,
        control_counts=rng.poisson(0.2, n).astype(np.float32), n_control_features=10,
    )


def _decontamination_config(tmp_path) -> CastleConfig:
    cfg = CastleConfig()
    c = cfg.model.contamination
    c.spillover_graph = "contact"
    c.kernel = "learned"
    c.geometry_features = "full"
    c.donor_gradient = True
    c.library = "amortized"
    c.background = "controls"
    c.unroll_fix = True
    c.expression_guided.enabled = False
    cfg.loss.lambda_eta = 0.0
    cfg.data.output_dir = str(tmp_path)
    cfg.graph.n_neighbors = 3
    cfg.model.hidden_dim = 16
    cfg.model.intrinsic_dim = 4
    cfg.model.niche_dim = 4
    cfg.train.warmup_epochs = 1
    cfg.train.contamination_epochs = 1
    cfg.train.joint_epochs = 1
    cfg.train.batch_size = 8
    cfg.train.cache_batch_size = 12
    cfg.train.checkpoint_every = 10
    return cfg


def test_full_decontamination_path(tmp_path):
    trainer = CastleTrainer(_grid_data(), _decontamination_config(tmp_path)).fit()
    output = trainer.export()
    result = np.load(output / "castle_embeddings_and_qc.npz")
    assert np.isfinite(result["eta"]).all() and np.isfinite(result["library"]).all()
    purified = sparse.load_npz(output / "purified_counts_cells_by_genes.npz").toarray()
    assert (purified <= trainer.data.counts.toarray() + 1e-5).all()


def test_transfer_weights_conserve_mass(tmp_path):
    for kernel in ("fixed", "learned"):
        cfg = _decontamination_config(tmp_path)
        cfg.model.contamination.kernel = kernel
        trainer = CastleTrainer(_grid_data(), cfg)
        cache = trainer.refresh_cache()
        per_donor = np.bincount(trainer.spill.source, weights=cache.edge_weights, minlength=trainer.data.n_cells)
        assert np.allclose(per_donor[trainer.has_out], 1.0, atol=1e-5)
        # Cells without contacts cannot leak, so nothing is lost from them.
        assert np.all(cache.eta[~trainer.has_out] == 0)


def test_donor_term_trains_leakage_network(tmp_path):
    grads = {}
    for donor_gradient in (False, True):
        cfg = _decontamination_config(tmp_path)
        cfg.model.contamination.donor_gradient = donor_gradient
        trainer = CastleTrainer(_grid_data(), cfg)
        trainer.refresh_cache()
        tensors = trainer._batch_tensors(np.arange(trainer.data.n_cells))
        # Silence the recipients' own retention term: any gradient left must come
        # through the spillover these cells receive from their donors.
        tensors["own_has_out"] = torch.zeros_like(tensors["own_has_out"])
        trainer.model.zero_grad()
        result = trainer.model.forward_batch(
            **tensors, phase="contamination", beta_intrinsic=1.0, stochastic=False
        )
        result.loss.backward()
        grads[donor_gradient] = sum(
            float(p.grad.abs().sum()) for p in trainer.model.eta_net.parameters() if p.grad is not None
        )
    assert grads[False] == 0.0
    assert grads[True] > 0.0


def test_fixed_eta_override_is_used_for_export(tmp_path):
    cfg = _decontamination_config(tmp_path)
    cfg.model.contamination.background = "none"
    trainer = CastleTrainer(_grid_data(), cfg).fit()
    trainer.eta_override = np.zeros(trainer.data.n_cells, dtype=np.float32)
    trainer.refresh_cache()
    output = trainer.export()
    result = np.load(output / "castle_embeddings_and_qc.npz")
    assert np.all(result["eta"] == 0)
    # No leakage and no background: every count is the cell's own.
    purified = sparse.load_npz(output / "purified_counts_cells_by_genes.npz").toarray()
    assert np.allclose(purified, trainer.data.counts.toarray(), atol=1e-4)


def test_spec_stages_run_end_to_end(tmp_path):
    """S4: learned background, transcript geometry, live donors, contact-step niche."""
    cfg = _decontamination_config(tmp_path)
    c = cfg.model.contamination
    c.background = "controls_learned"
    c.geometry_features = "spec"
    c.donor_live_profiles = True
    cfg.model.niche_encoder = "contact_steps"
    cfg.model.niche_steps = 3
    cfg.loss.niche_target = "purified"
    cfg.loss.niche_group_weight = 1.0
    trainer = CastleTrainer(_grid_data(), cfg)
    assert [scale.mask.sum(axis=1).max() for scale in trainer.step_neighbours] == sorted(
        scale.mask.sum(axis=1).max() for scale in trainer.step_neighbours
    )
    trainer.fit()
    output = trainer.export()
    result = np.load(output / "castle_embeddings_and_qc.npz")
    assert np.isfinite(result["z_niche"]).all() and np.isfinite(result["eta"]).all()
    beta = torch.softmax(trainer.model.background_logits.detach(), dim=0)
    assert torch.isclose(beta.sum(), torch.tensor(1.0))


def _identity_config(tmp_path, niche: bool) -> CastleConfig:
    cfg = _decontamination_config(tmp_path)
    cfg.loss.lambda_eta = 3.0
    m = cfg.model
    m.n_layers, m.norm, m.encoder_input, m.prior_components = 2, "batch", "dual", 4
    m.hidden_genes = [0, 5]
    cfg.loss.kl_warmup_epochs, cfg.loss.kl_max = 2, 1.0
    cfg.train.validation_fraction = 0.25
    cfg.train.early_stop_phase = "contamination"
    cfg.train.early_stop_patience, cfg.train.early_stop_min_epochs = 1, 1
    cfg.train.contamination_epochs = 4
    if niche:
        m.niche_encoder, m.niche_steps, m.niche_input = "contact_steps", 2, "zc"
        cfg.loss.identity_recon_weight = 0.5
        cfg.loss.block_delta_grad = True
        cfg.loss.lambda_niche_zc = 1.0
        cfg.loss.niche_target = "purified"
        cfg.model.contamination.freeze_in_joint = True
    else:
        cfg.train.joint_epochs = 0
    return cfg


def test_identity_model_without_niche(tmp_path):
    trainer = CastleTrainer(_grid_data(), _identity_config(tmp_path, niche=False)).fit()
    assert trainer.validation_ids.size == 6
    assert not np.isin(trainer.validation_ids, np.arange(24)[~np.isin(np.arange(24), trainer.validation_ids)]).any()
    assert all(row.get("phase") != "joint" for row in trainer.history)
    output = trainer.export()
    assert np.isfinite(np.load(output / "castle_embeddings_and_qc.npz")["z_intrinsic"]).all()
    # Hidden genes never reach the encoder.
    assert float(trainer.model.encoder_visible[0]) == 0.0 and float(trainer.model.encoder_visible[5]) == 0.0


def test_identity_model_with_niche_freezes_spillover(tmp_path):
    trainer = CastleTrainer(_grid_data(), _identity_config(tmp_path, niche=True))
    eta_before = [p.detach().clone() for p in trainer.model.eta_net.parameters()]
    tau_before = trainer.model.log_tau.detach().clone()
    trainer.fit()
    assert any(row.get("phase") == "joint" for row in trainer.history)
    # Spillover was trained before the niche phase and then frozen in it.
    assert not any(p.requires_grad for p in trainer.model.eta_net.parameters())
    assert not trainer.model.log_tau.requires_grad
    assert any(not torch.equal(a, b) for a, b in zip(eta_before, trainer.model.eta_net.parameters())) or not torch.equal(tau_before, trainer.model.log_tau)
    result = np.load(trainer.export() / "castle_embeddings_and_qc.npz")
    assert np.isfinite(result["z_niche"]).all()
