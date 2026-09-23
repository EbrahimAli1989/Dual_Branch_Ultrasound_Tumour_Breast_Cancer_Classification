"""
Default configuration for the Dual-Branch US Tumor Classification Framework.
All values here mirror the argparse defaults in train.py and serve as
a single source of truth for documentation and programmatic access.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import os


@dataclass
class ModelConfig:
    us_encoder: str = "cnn"          # "cnn" | "transformer"
    mask_encoder: str = "cnn"        # "cnn" | "transformer" | "snn"
    attention_type: str = "self"     # "self" | "cross"
    classifier_type: str = "mlp"     # "mlp" | "kan"
    feature_dim: int = 256           # output dim of each encoder branch
    dropout: float = 0.3

    # CNN encoder settings
    cnn_base_channels: int = 32

    # Transformer encoder settings
    vit_patch_size: int = 16
    vit_num_layers: int = 4
    vit_num_heads: int = 8
    vit_mlp_ratio: float = 4.0

    # SNN encoder settings
    snn_timesteps: int = 4
    snn_threshold: float = 1.0
    snn_decay: float = 0.5

    # Attention settings
    attn_num_heads: int = 8

    # MLP classifier settings
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    # KAN classifier settings
    kan_grid_size: int = 5
    kan_spline_order: int = 3
    kan_hidden_dims: List[int] = field(default_factory=lambda: [64])


@dataclass
class DataConfig:
    images_path: str = "data/images.npy"
    masks_path: str = "data/masks.npy"
    labels_path: str = "data/labels.npy"
    test_size: float = 0.15
    val_size: float = 0.15
    augment: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"          # "adamw" | "adam" | "sgd"
    scheduler: str = "cosine"         # "cosine" | "plateau" | "none"
    patience: int = 15                # early stopping patience
    grad_clip: float = 1.0
    amp: bool = True                  # automatic mixed precision
    label_smoothing: float = 0.0


@dataclass
class ExperimentConfig:
    seed: int = 42
    n_runs: int = 1                   # repeated random splits
    seeds: Optional[List[int]] = None # per-run seeds; auto-generated if None
    output_dir: str = "outputs"
    experiment_name: str = "exp"
    gpu: int = 0
    save_splits: bool = True


@dataclass
class FullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FullConfig":
        with open(path) as f:
            d = json.load(f)
        cfg = cls()
        cfg.model = ModelConfig(**d.get("model", {}))
        cfg.data = DataConfig(**d.get("data", {}))
        cfg.train = TrainConfig(**d.get("train", {}))
        cfg.experiment = ExperimentConfig(**d.get("experiment", {}))
        return cfg


def get_default_config() -> FullConfig:
    return FullConfig()
