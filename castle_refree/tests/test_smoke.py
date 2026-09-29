from __future__ import annotations

import numpy as np
from scipy import sparse

from castle_refree.config import CastleConfig
from castle_refree.data import SpatialCountData
from castle_refree.trainer import CastleTrainer


def test_tiny_end_to_end(tmp_path):
    rng = np.random.default_rng(4)
    n_cells, n_genes = 24, 8
    labels = np.repeat([0, 1], n_cells // 2)
    profiles = np.array(
        [[10, 8, 6, 1, 1, 1, 1, 1], [1, 1, 1, 1, 6, 8, 10, 1]],
        dtype=np.float32,
    )
    counts = rng.poisson(profiles[labels]).astype(np.float32)
    coords = np.column_stack([np.arange(n_cells), np.zeros(n_cells)]).astype(np.float32)
    data = SpatialCountData(
        sparse.csr_matrix(counts), coords, np.ones(n_cells, dtype=np.float32),
        np.array([f"c{i}" for i in range(n_cells)]),
        np.array([f"g{i}" for i in range(n_genes)]),
    )
    cfg = CastleConfig()
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
    trainer = CastleTrainer(data, cfg).fit()
    output = trainer.export()
    assert (output / "purified_counts_cells_by_genes.npz").exists()
    result = np.load(output / "castle_embeddings_and_qc.npz")
    assert result["z_intrinsic"].shape == (n_cells, 4)
    assert np.isfinite(result["segmentation_confidence"]).all()

