"""
Dataset loading and preprocessing pipeline for US tumor classification.

Assumes .npy files with shapes:
  images.npy : (N, H, W)  or  (N, H, W, 1)  — grayscale US images
  masks.npy  : (N, H, W)  or  (N, H, W, 1)  — binary tumour segmentation masks
  labels.npy : (N,)                           — binary labels {0, 1}

All images are resized/normalised to (1, 224, 224) float32 in [0, 1].
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms.functional as TF
import random
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Augmentation helpers (applied to both image and mask simultaneously)
# ---------------------------------------------------------------------------

class SyncAugment:
    """Synchronised geometric augmentation applied identically to image and mask."""

    def __init__(
        self,
        hflip_p: float = 0.5,
        vflip_p: float = 0.3,
        rotate_deg: float = 15.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
    ):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotate_deg = rotate_deg
        self.brightness = brightness
        self.contrast = contrast

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Horizontal flip
        if random.random() < self.hflip_p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Vertical flip
        if random.random() < self.vflip_p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation (mask uses nearest to keep binary)
        if self.rotate_deg > 0:
            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Intensity jitter — only on the image, not the mask
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = torch.clamp(image * factor, 0.0, 1.0)

        if self.contrast > 0:
            mean = image.mean()
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = torch.clamp((image - mean) * factor + mean, 0.0, 1.0)

        return image, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UltrasoundDataset(Dataset):
    """
    PyTorch dataset for paired ultrasound images and tumour masks.

    Args:
        images : float32 numpy array of shape (N, 1, 224, 224)
        masks  : float32 numpy array of shape (N, 1, 224, 224)
        labels : int64  numpy array of shape (N,)
        augment: whether to apply SyncAugment during __getitem__
    """

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        assert len(images) == len(masks) == len(labels), \
            "images, masks, and labels must have the same first dimension."

        self.images = images   # already (N,1,224,224) float32 in [0,1]
        self.masks = masks
        self.labels = labels
        self.augment = augment
        self._sync_aug = SyncAugment() if augment else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[idx])   # (1, 224, 224)
        mask = torch.from_numpy(self.masks[idx])     # (1, 224, 224)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.augment and self._sync_aug is not None:
            image, mask = self._sync_aug(image, mask)

        return image, mask, label


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_array(arr: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Normalise and reshape a raw numpy array to (N, 1, H, W) float32 in [0, 1].

    Handles input shapes:
      (N, H, W)        — single-channel images stored without the channel dim
      (N, H, W, 1)     — channel-last
      (N, 1, H, W)     — channel-first (pass-through)
    """
    arr = arr.astype(np.float32)

    # Add channel dimension if missing
    if arr.ndim == 3:
        arr = arr[:, np.newaxis, :, :]          # (N, 1, H, W)
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.transpose(0, 3, 1, 2)         # (N, H, W, 1) → (N, 1, H, W)
    elif arr.ndim == 4 and arr.shape[1] == 1:
        pass 
    elif arr.ndim == 4 and arr.shape[1] == 3:  
        arr = arr[:,0,:,:]                                 # already (N, 1, H, W)
    else:
        raise ValueError(f"Unexpected array shape: {arr.shape}")

    # Normalise to [0, 1] per-image using min-max
    n = arr.shape[0]
    arr_flat = arr.reshape(n, -1)
    mins = arr_flat.min(axis=1, keepdims=True).reshape(n, 1, 1, 1)
    maxs = arr_flat.max(axis=1, keepdims=True).reshape(n, 1, 1, 1)
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
    arr = (arr - mins) / denom

    # Resize if necessary using bilinear interpolation via torch
    h, w = arr.shape[2], arr.shape[3]
    if h != target_size or w != target_size:
        tensor = torch.from_numpy(arr)  # (N, 1, H, W)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
        )
        arr = tensor.numpy()

    return arr  # (N, 1, 224, 224) float32


def load_npy_data(
    images_path: str,
    masks_path: str,
    labels_path: str,
    target_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the three .npy files."""
    for p in (images_path, masks_path, labels_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Data file not found: {p}")

    images = np.load(images_path)
    masks = np.load(masks_path)
    labels = np.load(labels_path).astype(np.int64)

    images = _preprocess_array(images, target_size)
    masks = _preprocess_array(masks, target_size)

    # Binary-threshold masks: any value > 0.5 → 1
    masks = (masks > 0.5).astype(np.float32)

    return images, masks, labels


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------

def create_single_split(
    labels: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Stratified split into train / val / test index arrays.

    val_size is expressed as a fraction of the full dataset
    (not of the train set).
    """
    n = len(labels)
    indices = np.arange(n)

    # First, carve out test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(indices, labels))

    # Then split trainval into train / val
    relative_val = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_val, random_state=seed + 1
    )
    train_idx_local, val_idx_local = next(
        sss_val.split(trainval_idx, labels[trainval_idx])
    )

    train_idx = trainval_idx[train_idx_local]
    val_idx = trainval_idx[val_idx_local]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def create_repeated_splits(
    labels: np.ndarray,
    n_runs: int = 10,
    seeds: Optional[List[int]] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> List[Dict[str, np.ndarray]]:
    """Create n_runs independent stratified splits."""
    if seeds is None:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, 100_000, size=n_runs).tolist()
    assert len(seeds) >= n_runs, "Not enough seeds provided."

    return [
        create_single_split(labels, test_size, val_size, seed=seeds[i])
        for i in range(n_runs)
    ]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    images: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    split: Dict[str, np.ndarray],
    batch_size: int = 32,
    augment: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from pre-loaded arrays and split indices.

    Returns a dict with keys "train", "val", "test".
    """
    loaders = {}
    for phase in ("train", "val", "test"):
        is_train = phase == "train"
        ds = UltrasoundDataset(
            images=images,
            masks=masks,
            labels=labels,
            augment=(augment and is_train),
        )
        subset = Subset(ds, split[phase])
        loaders[phase] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,       # avoid stray size-1 batches with BN
            persistent_workers=(num_workers > 0),
        )

    return loaders


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

def dataset_summary(images: np.ndarray, masks: np.ndarray, labels: np.ndarray) -> str:
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    return (
        f"Dataset: {n} samples | "
        f"Positive (1): {n_pos} ({100*n_pos/n:.1f}%) | "
        f"Negative (0): {n_neg} ({100*n_neg/n:.1f}%) | "
        f"Image shape: {images.shape[1:]} | "
        f"Mask shape: {masks.shape[1:]}"
    )
