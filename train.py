"""
Main entry point for the Dual-Branch US Tumour Classification Framework.

Usage examples
--------------
# Single run, defaults (CNN + CNN, self-attention, MLP)
python train.py --images_path data/images.npy --masks_path data/masks.npy --labels_path data/labels.npy

# Transformer US encoder + SNN mask encoder, cross-attention, KAN classifier, 3 repeated runs
python train.py \\
    --images_path data/images.npy \\
    --masks_path  data/masks.npy  \\
    --labels_path data/labels.npy \\
    --us_encoder transformer \\
    --mask_encoder snn \\
    --attention_type cross \\
    --classifier_type kan \\
    --n_runs 3 \\
    --epochs 50 \\
    --batch_size 16 \\
    --gpu 0
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from utils.logger import setup_logger
from utils.reproducibility import set_seed, generate_seeds, save_splits
from utils.complexity import count_parameters, compute_flops
from datasets.ultrasound_dataset import (
    load_npy_data,
    build_dataloaders,
    create_repeated_splits,
    dataset_summary,
    UltrasoundDataset,
)
from models.fusion_model import build_model
from trainers.trainer import Trainer
from metrics.evaluation import (
    compute_metrics,
    save_confusion_matrix,
    save_roc_curve,
    save_metrics_csv,
    print_metrics_table,
    aggregate_runs,
    print_aggregate_table,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dual-Branch US Tumour Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Dataset paths ----
    dg = p.add_argument_group("Dataset")
    dg.add_argument("--images_path", type=str, default="data/images.npy",
                    help="Path to images .npy file, shape (N, H, W) or (N, H, W, 1)")
    dg.add_argument("--masks_path",  type=str, default="data/masks.npy",
                    help="Path to masks  .npy file")
    dg.add_argument("--labels_path", type=str, default="data/labels.npy",
                    help="Path to labels .npy file, shape (N,)")
    dg.add_argument("--test_size",  type=float, default=0.2, help="Test set fraction")
    dg.add_argument("--val_size",   type=float, default=0.1, help="Validation set fraction")
    dg.add_argument("--augment",    action="store_true",      help="Enable training augmentation")
    dg.add_argument("--num_workers",type=int,   default=4,    help="DataLoader worker count")

    # ---- Model ----
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

    # ---- Training ----
    tg = p.add_argument_group("Training")
    tg.add_argument("--batch_size",       type=int,   default=16)
    tg.add_argument("--epochs",           type=int,   default=100)
    tg.add_argument("--lr",               type=float, default=1e-4)
    tg.add_argument("--weight_decay",     type=float, default=1e-4)
    tg.add_argument("--optimizer",        type=str,   default="adamw",
                    choices=["adamw", "adam", "sgd"])
    tg.add_argument("--scheduler",        type=str,   default="cosine",
                    choices=["cosine", "plateau", "none"])
    tg.add_argument("--patience",         type=int,   default=15,
                    help="Early stopping patience (epochs)")
    tg.add_argument("--grad_clip",        type=float, default=1.0)
    tg.add_argument("--amp",              action="store_true", default=True,
                    help="Enable automatic mixed precision (GPU only)")
    tg.add_argument("--no_amp",           action="store_false", dest="amp",
                    help="Disable AMP")
    tg.add_argument("--label_smoothing",  type=float, default=0.0)

    # ---- Experiment ----
    eg = p.add_argument_group("Experiment")
    eg.add_argument("--seed",             type=int,  default=42,
                    help="Base random seed")
    eg.add_argument("--n_runs",           type=int,  default=10,
                    help="Number of repeated random-split experiments")
    eg.add_argument("--output_dir",       type=str,  default="outputs",
                    help="Root directory for all outputs")
    eg.add_argument("--experiment_name",  type=str,  default="exp",
                    help="Sub-folder name inside output_dir")
    eg.add_argument("--gpu",              type=int,  default=0,
                    help="CUDA device id (-1 for CPU)")
    eg.add_argument("--resume",           type=str,  default=None,
                    help="Path to checkpoint to resume training from")
    eg.add_argument("--save_splits",      action="store_true", default=True,
                    help="Save split indices for reproducibility")

    # ---- External Test Dataset (optional) ----
    xt = p.add_argument_group("External Test Dataset")
    xt.add_argument("--ext_images_path", type=str, default=None,
                    help="Path to external test images .npy; if provided the best model "
                         "is also evaluated on this dataset after each run's split-based test")
    xt.add_argument("--ext_masks_path",  type=str, default=None,
                    help="Path to external test masks  .npy file")
    xt.add_argument("--ext_labels_path", type=str, default=None,
                    help="Path to external test labels .npy file, shape (N,)")

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


# ---------------------------------------------------------------------------
# External dataset evaluation
# ---------------------------------------------------------------------------

def run_external_test(
    model: torch.nn.Module,
    args: argparse.Namespace,
    run_dir: str,
    device: torch.device,
    run_id: int,
) -> Dict:
    """
    Evaluate the best-checkpoint model on a separate external test dataset.

    Called after the split-based test inside each run. At this point the model
    already carries the best-checkpoint weights (loaded by Trainer.test()).

    Saves ext_confusion_matrix.txt and ext_roc_curve.pdf into run_dir and
    returns a metrics dict with the same CSV schema as run_single (flops /
    parameters are omitted since they belong to the model, not the dataset).
    """
    logger.info("Loading external test dataset …")
    ext_images, ext_masks, ext_labels = load_npy_data(
        args.ext_images_path, args.ext_masks_path, args.ext_labels_path
    )
    logger.info("[External] %s", dataset_summary(ext_images, ext_masks, ext_labels))

    ext_dataset = UltrasoundDataset(ext_images, ext_masks, ext_labels, augment=False)
    ext_loader = DataLoader(
        ext_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    use_amp = getattr(args, "amp", True) and device.type == "cuda"
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    t0 = time.time()
    with torch.no_grad():
        for images, masks, labels in tqdm(ext_loader, desc="[ext-test]", leave=False):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, masks)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
    ext_test_time = time.time() - t0

    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info("[External] Results:")
    print_metrics_table(metrics)

    save_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(run_dir, "ext_confusion_matrix.txt"),
    )
    save_roc_curve(
        y_true, y_prob,
        save_path=os.path.join(run_dir, "ext_roc_curve.pdf"),
        title=f"ROC Curve — External Test Set (Run {run_id})",
    )

    return {
        "run":          run_id,
        **metrics,
        "test_time_s":  round(ext_test_time, 4),
        "us_encoder":   args.us_encoder,
        "mask_encoder": args.mask_encoder,
        "attention":    args.attention_type,
        "classifier":   args.classifier_type,
    }


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

def run_single(
    args: argparse.Namespace,
    images: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    split: Dict[str, np.ndarray],
    run_id: int,
    device: torch.device,
    exp_dir: str,
) -> Tuple[Dict, Optional[Dict]]:
    """
    Execute one complete train → validate → test pipeline.

    Returns:
        full_metrics : split-based test metrics dict
        ext_metrics  : external test metrics dict, or None if not requested
    """
    run_dir = os.path.join(exp_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "training_log.txt")
    setup_logger("us_clf", log_file=log_path)   # add file handler for this run

    logger.info("=" * 60)
    logger.info("Run %d / %d", run_id + 1, args.n_runs)
    logger.info("Output directory: %s", run_dir)

    # ---- DataLoaders ----
    loaders = build_dataloaders(
        images=images,
        masks=masks,
        labels=labels,
        split=split,
        batch_size=args.batch_size,
        augment=args.augment,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(split["train"]), len(split["val"]), len(split["test"]),
    )

    # ---- Model ----
    set_seed(args.seed + run_id)
    model = build_model(args)
    n_params = count_parameters(model)
    logger.info("Model built. Trainable parameters: %d", n_params)

    # FLOPs (profile once on CPU to avoid GPU memory during setup)
    flops = compute_flops(model, device=torch.device("cpu"))
    logger.info("FLOPs (MACs): %d", flops)

    # ---- Train ----
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        args=args,
        output_dir=run_dir,
        device=device,
    )

    if args.resume and os.path.isfile(args.resume):
        trainer.load_checkpoint(args.resume)

    train_start = time.time()
    trainer.train()
    train_time = trainer.total_train_time

    # ---- Test ----
    y_true, y_pred, y_prob = trainer.test()
    test_time = getattr(trainer, "test_time", 0.0)

    # ---- Metrics ----
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics_table(metrics)

    # ---- Save outputs ----
    save_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(run_dir, "confusion_matrix.txt"),
    )
    save_roc_curve(
        y_true, y_prob,
        save_path=os.path.join(run_dir, "roc_curve.pdf"),
        title=f"ROC Curve — Run {run_id}",
    )

    full_metrics = {
        "run": run_id,
        **metrics,
        "train_time_s": round(train_time, 2),
        "test_time_s": round(test_time, 4),
        "flops": flops,
        "parameters": n_params,
        "us_encoder": args.us_encoder,
        "mask_encoder": args.mask_encoder,
        "attention": args.attention_type,
        "classifier": args.classifier_type,
    }

    # ---- External Test (optional) ----
    ext_metrics = None
    if getattr(args, "ext_images_path", None) is not None:
        logger.info("-" * 60)
        logger.info("External dataset evaluation for run %d …", run_id + 1)
        ext_metrics = run_external_test(trainer.model, args, run_dir, device, run_id)

    return full_metrics, ext_metrics


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

    logger.info("Dual-Branch US Tumour Classification")
    logger.info("Arguments: %s", vars(args))

    # Save config
    save_config(args, os.path.join(exp_dir, "config.json"))

    # Device
    device = get_device(args)

    # ---- Load data ----
    logger.info("Loading data …")
    images, masks, labels = load_npy_data(
        args.images_path, args.masks_path, args.labels_path
    )
    logger.info(dataset_summary(images, masks, labels))

    # ---- Create splits ----
    seeds = generate_seeds(args.n_runs, base_seed=args.seed)
    splits = create_repeated_splits(
        labels=labels,
        n_runs=args.n_runs,
        seeds=seeds,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    if args.save_splits:
        save_splits(splits, save_dir=exp_dir)

    # ---- CSV paths for all runs ----
    csv_path     = os.path.join(exp_dir, "metrics_all_runs.csv")
    ext_csv_path = os.path.join(exp_dir, "ext_metrics_all_runs.csv")

    # ---- Repeated experiments ----
    all_metrics:     List[Dict] = []
    all_ext_metrics: List[Dict] = []

    for run_id, split in enumerate(splits):
        run_metrics, ext_metrics = run_single(
            args=args,
            images=images,
            masks=masks,
            labels=labels,
            split=split,
            run_id=run_id,
            device=device,
            exp_dir=exp_dir,
        )
        all_metrics.append(run_metrics)
        save_metrics_csv([run_metrics], csv_path, append=True)
        if ext_metrics is not None:
            all_ext_metrics.append(ext_metrics)
            save_metrics_csv([ext_metrics], ext_csv_path, append=True)

    # ---- Aggregate over runs ----
    if args.n_runs > 1:
        agg = aggregate_runs(all_metrics)
        print_aggregate_table(agg)

        # Append summary row
        summary_row = {
            "run": "mean±std",
            **{k: f"{v['mean']:.4f}±{v['std']:.4f}" for k, v in agg.items()
               if k not in ("flops", "parameters")},
            "flops": all_metrics[0].get("flops", ""),
            "parameters": all_metrics[0].get("parameters", ""),
            "us_encoder": args.us_encoder,
            "mask_encoder": args.mask_encoder,
            "attention": args.attention_type,
            "classifier": args.classifier_type,
        }
        save_metrics_csv([summary_row], csv_path, append=True)

        # ---- Aggregate external test metrics (if any) ----
        if all_ext_metrics:
            logger.info("External Test — Aggregated Results:")
            ext_agg = aggregate_runs(all_ext_metrics)
            print_aggregate_table(ext_agg)
            ext_summary_row = {
                "run": "mean±std",
                **{k: f"{v['mean']:.4f}±{v['std']:.4f}" for k, v in ext_agg.items()},
                "us_encoder":   args.us_encoder,
                "mask_encoder": args.mask_encoder,
                "attention":    args.attention_type,
                "classifier":   args.classifier_type,
            }
            save_metrics_csv([ext_summary_row], ext_csv_path, append=True)

    logger.info("All done. Results in: %s", exp_dir)


if __name__ == "__main__":
    logger: logging.Logger  # forward declaration for type checker
    main()
