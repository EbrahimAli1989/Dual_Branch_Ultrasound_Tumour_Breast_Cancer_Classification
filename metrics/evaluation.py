"""
Performance evaluation utilities.

Outputs:
  - Confusion matrix saved as .txt
  - ROC curve saved as .pdf
  - Metrics summary saved as a row in a .csv file

Metrics computed:
  accuracy, sensitivity (recall), specificity, precision, F1, AUC,
  training time, testing time, FLOPs, parameter count
"""

import os
import csv
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true : ground-truth labels (0/1)
        y_pred : predicted labels (0/1)
        y_prob : predicted probabilities for the positive class (label=1)

    Returns:
        dict with keys: accuracy, sensitivity, specificity, precision,
                        f1, auc
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)   # sensitivity = recall
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, cm[1, 1])
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "auc": float(auc),
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    save_path: str,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute and save the confusion matrix to a .txt file.

    Args:
        y_true      : ground-truth labels
        y_pred      : predicted labels
        save_path   : full path to the output .txt file
        class_names : optional list of class name strings

    Returns:
        cm: numpy confusion matrix array
    """
    if class_names is None:
        class_names = ["Benign (0)", "Malignant (1)"]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lines = []
    lines.append("Confusion Matrix")
    lines.append("=" * 50)

    # Header
    header = " " * 18 + "  ".join(f"Pred:{c:>12}" for c in class_names)
    lines.append(header)
    lines.append("-" * 50)

    for i, row_name in enumerate(class_names):
        row_str = f"True:{row_name:>12}  " + "  ".join(f"{cm[i, j]:>16}" for j in range(len(class_names)))
        lines.append(row_str)

    lines.append("=" * 50)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, cm[1, 1])
    lines.append(f"\nTP={tp}  TN={tn}  FP={fp}  FN={fn}")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    lines.append(f"Accuracy from CM: {acc:.4f}")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Confusion matrix saved to %s", save_path)
    return cm


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def save_roc_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    save_path: str,
    title: str = "ROC Curve",
) -> float:
    """
    Plot and save the ROC curve as a .pdf file.

    Args:
        y_true    : ground-truth binary labels
        y_prob    : positive-class predicted probabilities
        save_path : full path to the output .pdf file
        title     : plot title

    Returns:
        auc: float
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    try:
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except ValueError as e:
        logger.warning("Could not compute ROC curve: %s", e)
        return float("nan")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    logger.info("ROC curve (AUC=%.4f) saved to %s", auc, save_path)
    return auc


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "run",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "auc",
    "train_time_s",
    "test_time_s",
    "flops",
    "parameters",
    "us_encoder",
    "mask_encoder",
    "attention",
    "classifier",
]


def save_metrics_csv(
    rows: List[Dict],
    save_path: str,
    append: bool = True,
):
    """
    Save (or append) per-run metric rows to a CSV file.

    Args:
        rows      : list of dicts, each containing at least the _CSV_FIELDS keys
        save_path : path to the output CSV file
        append    : if True and the file exists, append rows without re-writing header
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_exists = os.path.isfile(save_path)
    mode = "a" if (append and file_exists) else "w"

    with open(save_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if not (append and file_exists):
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _CSV_FIELDS})

    logger.info("Metrics CSV saved/updated at %s", save_path)


def print_metrics_table(metrics: Dict[str, float]):
    """Pretty-print a metrics dict to stdout."""
    print("\n" + "=" * 45)
    print("  Evaluation Results")
    print("=" * 45)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<18}: {v:.4f}")
        else:
            print(f"  {k:<18}: {v}")
    print("=" * 45 + "\n")


# ---------------------------------------------------------------------------
# Aggregate multi-run statistics
# ---------------------------------------------------------------------------

def aggregate_runs(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean ± std over repeated runs.

    Returns:
        dict mapping metric_name → {"mean": ..., "std": ...}
    """
    keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
    results = {}
    for k in keys:
        vals = np.array([m[k] for m in all_metrics if not np.isnan(m.get(k, float("nan")))])
        results[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return results


def print_aggregate_table(agg: Dict[str, Dict[str, float]]):
    """Pretty-print aggregated mean ± std."""
    print("\n" + "=" * 55)
    print("  Aggregated Results (mean ± std over runs)")
    print("=" * 55)
    for k, v in agg.items():
        print(f"  {k:<20}: {v['mean']:.4f} ± {v['std']:.4f}")
    print("=" * 55 + "\n")
