"""
Standalone evaluation entry-point for the Dual-Branch US Tumour Classification Framework.

Loads a trained checkpoint, runs inference on the test set, reports the same performance
metrics as train.py (accuracy, precision, recall/sensitivity, specificity, F1, AUC, loss,
confusion matrix, ROC curve), saves all logs and CSV results, and generates Grad-CAM
visualisations on a randomly selected subset of test images.

Two evaluation modes
--------------------
split   (default)
    Loads the same full dataset used during training and evaluates on the held-out
    test partition only.  The partition is resolved in priority order:
      1. --split_file  →  exact indices saved by train.py (reproducible)
      2. no split_file →  a fresh stratified split is created from --seed

external
    Loads a completely separate, dedicated test dataset.  Provide
    --test_images_path / --test_masks_path / --test_labels_path.
    Every sample in those files is used for evaluation; no splitting is performed.

Usage examples
--------------
# [split] Minimal — carve test set from the training dataset
python test.py \\
    --images_path data/images.npy \\
    --masks_path  data/masks.npy  \\
    --labels_path data/labels.npy \\
    --checkpoint  outputs/exp/run_0/best_model.pth

# [split] Reuse the exact test split saved during training
python test.py \\
    --images_path  data/images.npy \\
    --masks_path   data/masks.npy  \\
    --labels_path  data/labels.npy \\
    --checkpoint   outputs/exp/run_0/best_model.pth \\
    --split_file   outputs/exp/splits.json \\
    --run_id       0 \\
    --gradcam_n_samples 20 \\
    --output_dir   outputs/exp \\
    --experiment_name  test_run_0 \\
    --gpu          0

# [external] Evaluate on a fully independent test dataset
python test.py \\
    --eval_mode         external \\
    --test_images_path  data/ext_test_images.npy \\
    --test_masks_path   data/ext_test_masks.npy  \\
    --test_labels_path  data/ext_test_labels.npy \\
    --checkpoint        outputs/exp/run_0/best_model.pth \\
    --gradcam_n_samples 20 \\
    --output_dir        outputs/exp \\
    --experiment_name   test_external \\
    --gpu               0

# [external] Transformer + SNN on an external dataset
python test.py \\
    --eval_mode         external \\
    --test_images_path  data/ext_test_images.npy \\
    --test_masks_path   data/ext_test_masks.npy  \\
    --test_labels_path  data/ext_test_labels.npy \\
    --checkpoint        outputs/exp/run_0/best_model.pth \\
    --us_encoder        transformer \\
    --mask_encoder      snn \\
    --attention_type    cross \\
    --classifier_type   kan \\
    --gpu 0
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt

# Local imports
from utils.logger import setup_logger
from utils.reproducibility import set_seed, load_splits
from datasets.ultrasound_dataset import (
    load_npy_data,
    UltrasoundDataset,
    create_single_split,
    dataset_summary,
)
from models.fusion_model import build_model, DualBranchClassifier
from metrics.evaluation import (
    compute_metrics,
    save_confusion_matrix,
    save_roc_curve,
    save_metrics_csv,
    print_metrics_table,
)


# ---------------------------------------------------------------------------
# Argument parser  (mirrors train.py exactly; adds test-specific arguments)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dual-Branch US Tumour Classification — Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Evaluation mode ----
    mo = p.add_argument_group("Evaluation Mode")
    mo.add_argument("--eval_mode", type=str, default="split",
                    choices=["split", "external"],
                    help="'split'   : carve the test set from the full training dataset; "
                         "'external': evaluate on a separate, dedicated test dataset")

    # ---- Dataset paths (split mode) ----
    dg = p.add_argument_group("Dataset — split mode")
    dg.add_argument("--images_path", type=str, default="data/images.npy",
                    help="[split] Full dataset images .npy, shape (N, H, W) or (N, H, W, 1)")
    dg.add_argument("--masks_path",  type=str, default="data/masks.npy",
                    help="[split] Full dataset masks  .npy file")
    dg.add_argument("--labels_path", type=str, default="data/labels.npy",
                    help="[split] Full dataset labels .npy file, shape (N,)")
    dg.add_argument("--test_size",   type=float, default=0.2,
                    help="[split] Test set fraction (used only if --split_file is not given)")
    dg.add_argument("--val_size",    type=float, default=0.1,
                    help="[split] Validation set fraction (used only if --split_file is not given)")
    dg.add_argument("--num_workers", type=int,   default=4,
                    help="DataLoader worker count")

    # ---- Dataset paths (external mode) ----
    xg = p.add_argument_group("Dataset — external mode")
    xg.add_argument("--test_images_path", type=str, default=None,
                    help="[external] Path to external test images .npy file")
    xg.add_argument("--test_masks_path",  type=str, default=None,
                    help="[external] Path to external test masks  .npy file")
    xg.add_argument("--test_labels_path", type=str, default=None,
                    help="[external] Path to external test labels .npy file, shape (N,)")

    # ---- Model  (identical to train.py so the architecture can be reconstructed) ----
    mg = p.add_argument_group("Model")
    mg.add_argument("--us_encoder",      type=str, default="cnn",
                    choices=["cnn", "transformer"],
                    help="US image encoder architecture")
    mg.add_argument("--mask_encoder",    type=str, default="cnn",
                    choices=["cnn", "transformer", "snn"],
                    help="Tumour mask encoder architecture")
    mg.add_argument("--attention_type",  type=str, default="self",
                    choices=["self", "cross"],
                    help="Feature fusion attention mechanism")
    mg.add_argument("--classifier_type", type=str, default="mlp",
                    choices=["mlp", "kan"],
                    help="Classification head type")
    mg.add_argument("--feature_dim",     type=int, default=256,
                    help="Per-branch encoder output dimension")
    mg.add_argument("--dropout",         type=float, default=0.3,
                    help="Dropout rate throughout the model")
    mg.add_argument("--num_classes",     type=int, default=2)

    # CNN encoder options
    mg.add_argument("--base_channels",   type=int, default=32,
                    help="Base channel count for CNN encoders")

    # Transformer encoder options
    mg.add_argument("--vit_patch_size",  type=int, default=16)
    mg.add_argument("--vit_num_layers",  type=int, default=4)
    mg.add_argument("--vit_num_heads",   type=int, default=8)
    mg.add_argument("--vit_mlp_ratio",   type=float, default=4.0)

    # SNN options
    mg.add_argument("--snn_timesteps",   type=int, default=4)
    mg.add_argument("--snn_threshold",   type=float, default=1.0)
    mg.add_argument("--snn_decay",       type=float, default=0.5)

    # Attention options
    mg.add_argument("--attn_num_heads",  type=int, default=8)

    # MLP classifier
    mg.add_argument("--mlp_hidden_dims", type=int, nargs="+", default=[256, 128],
                    help="Hidden layer widths for MLP classifier")

    # KAN classifier
    mg.add_argument("--kan_hidden_dims", type=int, nargs="+", default=[64],
                    help="Hidden layer widths for KAN classifier")
    mg.add_argument("--kan_grid_size",   type=int, default=5)
    mg.add_argument("--kan_spline_order",type=int, default=3)

    # ---- Evaluation ----
    ev = p.add_argument_group("Evaluation")
    ev.add_argument("--checkpoint",      type=str, required=True,
                    help="Path to the trained model checkpoint (.pth)")
    ev.add_argument("--batch_size",      type=int,   default=16)
    ev.add_argument("--amp",             action="store_true", default=True,
                    help="Enable automatic mixed precision for inference (GPU only)")
    ev.add_argument("--no_amp",          action="store_false", dest="amp",
                    help="Disable AMP")
    ev.add_argument("--split_file",      type=str, default=None,
                    help="Path to splits.json saved during training "
                         "(enables exact-reproducibility of the test partition)")
    ev.add_argument("--run_id",          type=int, default=0,
                    help="Which run's split to load from split_file")
    ev.add_argument("--seed",            type=int, default=42,
                    help="Random seed (controls split creation and Grad-CAM sample selection)")

    # ---- Grad-CAM ----
    gg = p.add_argument_group("Grad-CAM")
    gg.add_argument("--gradcam_n_samples",     type=int, default=20,
                    help="Number of randomly selected test images to visualise with Grad-CAM")
    gg.add_argument("--gradcam_target_class",  type=int, default=None,
                    help="Class index to target in Grad-CAM "
                         "(None → use the model's predicted class per image)")

    # ---- Experiment / output ----
    og = p.add_argument_group("Experiment")
    og.add_argument("--output_dir",      type=str, default="outputs",
                    help="Root directory for all outputs")
    og.add_argument("--experiment_name", type=str, default="test",
                    help="Sub-folder name inside output_dir")
    og.add_argument("--gpu",             type=int, default=0,
                    help="CUDA device id (-1 for CPU)")

    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_config(args: argparse.Namespace, path: str):
    """Serialise argparse namespace to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)


def get_device(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and args.gpu >= 0:
        dev = torch.device(f"cuda:{args.gpu}")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(args.gpu))
    else:
        dev = torch.device("cpu")
        logger.info("Using CPU")
    return dev


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    """Load model weights from a training checkpoint; return the raw state dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    epoch = ckpt.get("epoch", "unknown")
    logger.info("Loaded checkpoint '%s'  (saved at epoch %s)", path, epoch)
    return ckpt


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for the dual-branch model.

    Registers forward and backward hooks on a target convolutional layer to capture
    activation maps and their gradients, then computes spatially-resolved saliency maps.

    Args:
        model        : DualBranchClassifier whose weights are already loaded
        target_layer : nn.Module with a 4-D output (B, C, H, W); must lie on the
                       forward path from the input tensors to the final logits
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _save_activation(self, module, input, output):
        self._activations = output.detach().clone()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach().clone()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a *single* image-mask pair.

        Processes one sample at a time to ensure per-sample gradient correctness.

        Args:
            image     : (1, 1, H, W) tensor on the model's device
            mask      : (1, 1, H, W) tensor on the model's device
            class_idx : target class index; None → the model's predicted class

        Returns:
            heatmap : (H, W) float32 numpy array with values in [0, 1]
        """
        self.model.eval()
        self._activations = None
        self._gradients   = None

        # Forward — must NOT be inside torch.no_grad()
        logits = self.model(image, mask)            # (1, num_classes)

        target = int(logits.argmax(dim=1).item()) if class_idx is None else class_idx
        score  = logits[0, target]                 # scalar

        self.model.zero_grad()
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Verify that the target layer lies on the forward path."
            )

        # Global-average-pool gradients → channel importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1)          # (1, H', W')
        cam     = F.relu(cam)

        # Upsample to the original input spatial size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).squeeze(0)                                     # (H, W)

        # Normalise to [0, 1]
        cam_np = cam.cpu().numpy().astype(np.float32)
        mn, mx = cam_np.min(), cam_np.max()
        if mx - mn > 1e-8:
            cam_np = (cam_np - mn) / (mx - mn)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self):
        """Deregister all hooks to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_gradcam_target_layer(model: DualBranchClassifier) -> Optional[nn.Module]:
    """
    Identify the most suitable target layer for Grad-CAM in the US image encoder.

    CNN encoder        → last ResBlock of the final conv stage  (7 × 7 spatial map)
    Transformer enc.   → patch-embedding Conv2d projection      (14 × 14 spatial map)

    Returns None and logs a warning for unknown encoder types.
    """
    enc  = model.us_encoder
    name = type(enc).__name__

    if name == "CNNUSEncoder":
        # stages[-1] is a Sequential: [ConvBNReLU, ResBlock, MaxPool]
        # index [1] is the ResBlock whose output is (B, 16C, 7, 7)
        return enc.stages[-1][1]

    elif name == "TransformerUSEncoder":
        # patch_embed.proj is Conv2d(1 → embed_dim, k=patch_size, s=patch_size)
        # its output is (B, embed_dim, H/P, W/P) — a spatial feature map
        return enc.patch_embed.proj

    else:
        logger.warning(
            "Unknown US encoder type '%s'. Grad-CAM target layer not identified.", name
        )
        return None


# ---------------------------------------------------------------------------
# Grad-CAM visualisation
# ---------------------------------------------------------------------------

_CLASS_NAMES: Dict[int, str] = {0: "Benign", 1: "Malignant"}


def save_gradcam_figure(
    image_np:       np.ndarray,
    heatmap_np:     np.ndarray,
    save_path:      str,
    true_label:     int,
    pred_label:     int,
    prob_malignant: float,
    sample_idx:     int,
):
    """
    Save a three-panel Grad-CAM figure: original image | heatmap | overlay.

    Args:
        image_np        : (H, W) float32 US image in [0, 1]
        heatmap_np      : (H, W) float32 Grad-CAM map in [0, 1]
        save_path       : output file path (.png)
        true_label      : ground-truth class index
        pred_label      : predicted class index
        prob_malignant  : predicted probability for the malignant class
        sample_idx      : global dataset index of this sample (used in title)
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1 — original image
    axes[0].imshow(image_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original US Image", fontsize=11)
    axes[0].axis("off")

    # Panel 2 — Grad-CAM heatmap
    im = axes[1].imshow(heatmap_np, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — overlay
    axes[2].imshow(image_np, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(heatmap_np, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM Overlay", fontsize=11)
    axes[2].axis("off")

    correct_marker = "✓" if true_label == pred_label else "✗"
    suptitle = (
        f"Sample #{sample_idx}  |  "
        f"True: {_CLASS_NAMES.get(true_label, str(true_label))}  |  "
        f"Pred: {_CLASS_NAMES.get(pred_label, str(pred_label))} {correct_marker}  |  "
        f"P(Malignant) = {prob_malignant:.3f}"
    )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_figure(
    y_true: List[int],
    y_pred: List[int],
    save_path: str,
):
    """Save a colour-coded confusion matrix as a .pdf figure."""
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(y_true, y_pred, labels=[0, 1])
    class_names = ["Benign", "Malignant"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
            )

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix figure saved to %s", save_path)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    use_amp: bool,
) -> Tuple[List[int], List[int], List[float], float]:
    """
    Run a full evaluation pass over `loader`.

    Args:
        model   : model in eval mode, on `device`
        loader  : DataLoader for the test set
        device  : target device
        use_amp : whether to use automatic mixed precision

    Returns:
        y_true   : list of ground-truth labels
        y_pred   : list of predicted labels
        y_prob   : list of positive-class (malignant) probabilities
        avg_loss : mean cross-entropy loss over the test set
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for images, masks, labels in tqdm(loader, desc="[test]", leave=False):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits = model(images, masks)
                loss   = criterion(logits, labels)

            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            bs          = labels.size(0)
            total_loss += loss.item() * bs
            total      += bs

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    return y_true, y_pred, y_prob, avg_loss


# ---------------------------------------------------------------------------
# Grad-CAM generation loop
# ---------------------------------------------------------------------------

def run_gradcam(
    model:              DualBranchClassifier,
    images_np:          np.ndarray,
    masks_np:           np.ndarray,
    labels_np:          np.ndarray,
    test_indices:       np.ndarray,
    sample_local_idxs:  np.ndarray,
    y_pred:             List[int],
    y_prob:             List[float],
    device:             torch.device,
    output_dir:         str,
    target_class:       Optional[int],
):
    """
    Compute and save Grad-CAM figures for a randomly selected subset of test images.

    Args:
        model              : DualBranchClassifier with loaded weights
        images_np          : full dataset images array (N, 1, H, W)
        masks_np           : full dataset masks array  (N, 1, H, W)
        labels_np          : full dataset labels array (N,)
        test_indices       : global indices for the test set (maps local→global)
        sample_local_idxs  : positions *within* the test set to visualise
        y_pred             : predicted labels aligned with test_indices
        y_prob             : malignant probabilities aligned with test_indices
        device             : compute device
        output_dir         : directory to write .png files
        target_class       : Grad-CAM class target (None → predicted class)
    """
    target_layer = get_gradcam_target_layer(model)
    if target_layer is None:
        logger.warning("Grad-CAM skipped: could not identify a target layer.")
        return

    gradcam = GradCAM(model, target_layer)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Generating Grad-CAM for %d images …", len(sample_local_idxs))

    for rank, local_idx in enumerate(sample_local_idxs):
        global_idx = int(test_indices[local_idx])

        img_t  = torch.from_numpy(images_np[global_idx]).unsqueeze(0).to(device)  # (1,1,H,W)
        mask_t = torch.from_numpy(masks_np[global_idx]).unsqueeze(0).to(device)   # (1,1,H,W)

        try:
            heatmap = gradcam.compute(img_t, mask_t, class_idx=target_class)
        except Exception as exc:
            logger.warning(
                "  [%3d/%d] Grad-CAM failed for global idx %d: %s",
                rank + 1, len(sample_local_idxs), global_idx, exc,
            )
            continue

        true_lbl = int(labels_np[global_idx])
        pred_lbl = int(y_pred[local_idx])
        prob_mal = float(y_prob[local_idx])

        fname = (
            f"gradcam_{rank:03d}"
            f"_idx{global_idx:05d}"
            f"_true{true_lbl}"
            f"_pred{pred_lbl}.png"
        )
        save_path = os.path.join(output_dir, fname)

        save_gradcam_figure(
            image_np=images_np[global_idx, 0],   # (H, W)
            heatmap_np=heatmap,
            save_path=save_path,
            true_label=true_lbl,
            pred_label=pred_lbl,
            prob_malignant=prob_mal,
            sample_idx=global_idx,
        )
        logger.info(
            "  [%3d/%d] saved %s",
            rank + 1, len(sample_local_idxs), os.path.basename(save_path),
        )

    gradcam.remove_hooks()
    logger.info("Grad-CAM visualisations saved to: %s", output_dir)


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------

def run_test(
    args:    argparse.Namespace,
    images:  np.ndarray,
    masks:   np.ndarray,
    labels:  np.ndarray,
    split:   Dict[str, np.ndarray],
    device:  torch.device,
    exp_dir: str,
) -> Dict:
    """
    Full evaluation pipeline: build model → load checkpoint → test → metrics → Grad-CAM.

    Args:
        args    : parsed argparse namespace
        images  : (N, 1, H, W) float32 preprocessed images
        masks   : (N, 1, H, W) float32 preprocessed masks
        labels  : (N,) int64 labels
        split   : dict with keys "train", "val", "test" mapping to index arrays
        device  : compute device
        exp_dir : output directory for this run

    Returns:
        dict of all evaluation metrics
    """
    os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "test.log")
    setup_logger("us_clf", log_file=log_path)

    logger.info("=" * 60)
    logger.info("Evaluation run  [mode: %s]", getattr(args, "eval_mode", "split"))
    logger.info("Checkpoint : %s", args.checkpoint)
    logger.info("Output dir : %s", exp_dir)

    # ---- Build model and load weights ----
    set_seed(args.seed)
    model = build_model(args).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # ---- Test DataLoader (no augmentation) ----
    test_dataset = UltrasoundDataset(
        images=images, masks=masks, labels=labels, augment=False
    )
    test_subset = Subset(test_dataset, split["test"])
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    if len(split.get("train", [])) > 0 or len(split.get("val", [])) > 0:
        logger.info(
            "Split sizes — train: %d  val: %d  test: %d",
            len(split["train"]), len(split["val"]), len(split["test"]),
        )
    else:
        logger.info("Test set size: %d samples (external dataset — no split)", len(split["test"]))

    use_amp = getattr(args, "amp", True) and device.type == "cuda"

    # ---- Evaluation pass ----
    t0 = time.time()
    y_true, y_pred, y_prob, avg_loss = evaluate(model, test_loader, device, use_amp)
    test_time = time.time() - t0

    # ---- Compute and display metrics ----
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = round(avg_loss, 6)
    print_metrics_table(metrics)

    # ---- Save standard outputs ----
    save_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(exp_dir, "confusion_matrix.txt"),
    )
    save_confusion_matrix_figure(
        y_true, y_pred,
        save_path=os.path.join(exp_dir, "confusion_matrix.pdf"),
    )
    save_roc_curve(
        y_true, y_prob,
        save_path=os.path.join(exp_dir, "roc_curve.pdf"),
        title="ROC Curve — Test Set",
    )

    # ---- Persist metrics ----
    full_metrics = {
        "run": "test",
        **metrics,
        "test_time_s": round(test_time, 4),
        "us_encoder":   args.us_encoder,
        "mask_encoder": args.mask_encoder,
        "attention":    args.attention_type,
        "classifier":   args.classifier_type,
        "checkpoint":   args.checkpoint,
    }

    # CSV (compatible with train.py's metrics_all_runs.csv schema)
    save_metrics_csv(
        [full_metrics],
        save_path=os.path.join(exp_dir, "test_metrics.csv"),
        append=False,
    )

    # Full JSON (includes fields not in the fixed CSV schema)
    json_path = os.path.join(exp_dir, "test_metrics_full.json")
    with open(json_path, "w") as f:
        json.dump(full_metrics, f, indent=2, default=str)
    logger.info("Full metrics JSON saved to %s", json_path)

    logger.info(
        "Test completed in %.2fs | loss=%.4f | acc=%.4f | F1=%.4f | AUC=%.4f",
        test_time, avg_loss, metrics["accuracy"], metrics["f1"], metrics["auc"],
    )

    # ---- Grad-CAM on randomly selected test images ----
    n_samples = min(args.gradcam_n_samples, len(split["test"]))
    if n_samples > 0:
        rng = np.random.default_rng(args.seed)
        sample_local_idxs = np.sort(
            rng.choice(len(split["test"]), size=n_samples, replace=False)
        )

        run_gradcam(
            model=model,
            images_np=images,
            masks_np=masks,
            labels_np=labels,
            test_indices=split["test"],
            sample_local_idxs=sample_local_idxs,
            y_pred=y_pred,
            y_prob=y_prob,
            device=device,
            output_dir=os.path.join(exp_dir, "gradcam"),
            target_class=getattr(args, "gradcam_target_class", None),
        )

    return full_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global logger

    parser = build_parser()
    args = parser.parse_args()

    # Root experiment directory
    exp_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Logger (console + root file)
    root_log = os.path.join(exp_dir, "experiment.log")
    logger = setup_logger("us_clf", log_file=root_log)

    logger.info("Dual-Branch US Tumour Classification — Evaluation")
    logger.info("Mode      : %s", args.eval_mode)
    logger.info("Arguments : %s", vars(args))

    # Save config snapshot
    save_config(args, os.path.join(exp_dir, "test_config.json"))

    # Device
    device = get_device(args)

    # =========================================================================
    # Mode A — split: carve test set from the full training dataset
    # =========================================================================
    if args.eval_mode == "split":
        logger.info("Loading full dataset …")
        images, masks, labels = load_npy_data(
            args.images_path, args.masks_path, args.labels_path
        )
        logger.info(dataset_summary(images, masks, labels))

        if args.split_file is not None:
            logger.info(
                "Loading split from '%s'  (run_id=%d)", args.split_file, args.run_id
            )
            splits = load_splits(
                save_dir=os.path.dirname(os.path.abspath(args.split_file)),
                filename=os.path.basename(args.split_file),
            )
            if args.run_id >= len(splits):
                raise ValueError(
                    f"run_id={args.run_id} is out of range — "
                    f"split_file contains only {len(splits)} run(s)."
                )
            split = splits[args.run_id]
        else:
            logger.info(
                "No split_file provided — creating a new stratified split (seed=%d)",
                args.seed,
            )
            split = create_single_split(
                labels=labels,
                test_size=args.test_size,
                val_size=args.val_size,
                seed=args.seed,
            )

    # =========================================================================
    # Mode B — external: evaluate on a fully independent test dataset
    # =========================================================================
    elif args.eval_mode == "external":
        missing = [
            flag for flag, val in (
                ("--test_images_path", args.test_images_path),
                ("--test_masks_path",  args.test_masks_path),
                ("--test_labels_path", args.test_labels_path),
            )
            if val is None
        ]
        if missing:
            parser.error(
                f"eval_mode=external requires: {', '.join(missing)}"
            )

        logger.info("Loading external test dataset …")
        images, masks, labels = load_npy_data(
            args.test_images_path,
            args.test_masks_path,
            args.test_labels_path,
        )
        logger.info(dataset_summary(images, masks, labels))

        # All samples are test samples; train/val partitions are empty
        split = {
            "train": np.array([], dtype=np.int64),
            "val":   np.array([], dtype=np.int64),
            "test":  np.arange(len(labels), dtype=np.int64),
        }

    else:
        raise ValueError(f"Unknown eval_mode: '{args.eval_mode}'")

    # ---- Evaluate ----
    run_test(
        args=args,
        images=images,
        masks=masks,
        labels=labels,
        split=split,
        device=device,
        exp_dir=exp_dir,
    )

    logger.info("All done. Results in: %s", exp_dir)


if __name__ == "__main__":
    logger: logging.Logger  # forward declaration for type checker
    main()
