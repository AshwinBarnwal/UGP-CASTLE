"""Run the pinned UCE CLI with AnnData nullable-string writing enabled."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import anndata as ad


ROOT = Path(__file__).resolve().parents[1]
UCE = ROOT / "external" / "UCE"
ad.settings.allow_write_nullable_strings = True
sys.path.insert(0, str(UCE))
runpy.run_path(str(UCE / "eval_single_anndata.py"), run_name="__main__")

