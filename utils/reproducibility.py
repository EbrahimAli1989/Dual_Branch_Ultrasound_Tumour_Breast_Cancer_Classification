"""
Reproducibility utilities: seed control and split persistence.
"""

import os
import random
import logging
import json
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set all random seeds for full reproducibility.

    Covers Python's random, NumPy, PyTorch (CPU + CUDA), and
    enables deterministic CuDNN mode (may slightly reduce throughput).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic backend — disable for speed if acceptable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch ≥ 1.11: fully deterministic mode
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass

    logger.debug("Global seed set to %d", seed)


def generate_seeds(n: int, base_seed: int = 0) -> List[int]:
    """Generate n reproducible seeds from a base seed."""
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, 100_000, size=n).tolist()


def save_splits(
    splits: List[Dict[str, np.ndarray]],
    save_dir: str,
    filename: str = "splits.json",
):
    """
    Persist split indices to a JSON file for exact reproducibility.

    Each run's indices are serialised as lists of ints.
    """
    os.makedirs(save_dir, exist_ok=True)
    serialisable = [
        {phase: idx.tolist() for phase, idx in run.items()}
        for run in splits
    ]
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Splits saved to %s", path)


def load_splits(
    save_dir: str,
    filename: str = "splits.json",
) -> List[Dict[str, np.ndarray]]:
    """
    Load previously saved split indices.

    Returns:
        list of dicts mapping phase → np.ndarray of indices
    """
    path = os.path.join(save_dir, filename)
    with open(path) as f:
        raw = json.load(f)
    return [
        {phase: np.array(idx) for phase, idx in run.items()}
        for run in raw
    ]


def create_stratified_splits(
    labels: np.ndarray,
    n_runs: int = 1,
    seeds: Optional[List[int]] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> List[Dict[str, np.ndarray]]:
    """
    Create `n_runs` stratified train/val/test splits.

    This is a thin wrapper that defers to the dataset module but lives
    here so callers can import from utils without touching datasets.
    """
    from datasets.ultrasound_dataset import create_repeated_splits
    return create_repeated_splits(
        labels=labels,
        n_runs=n_runs,
        seeds=seeds,
        test_size=test_size,
        val_size=val_size,
    )
