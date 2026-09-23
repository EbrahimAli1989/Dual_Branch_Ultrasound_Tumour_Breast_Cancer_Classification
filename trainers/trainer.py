"""
Training, validation, and testing framework with:
  - Mixed-precision training (AMP)
  - Gradient clipping
  - Cosine-annealing / ReduceLROnPlateau scheduler
  - Early stopping
  - Best-model and latest-checkpoint saving
  - tqdm progress bars
  - Per-epoch logging
"""

import os
import time
import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training if the monitored metric does not improve for `patience` epochs.

    Args:
        patience  : epochs with no improvement before stopping
        min_delta : minimum change to qualify as improvement
        mode      : "min" (lower is better) or "max"
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-5, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            (score < self.best_score - self.min_delta) if self.mode == "min"
            else (score > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training lifecycle for DualBranchClassifier.

    Args:
        model       : the DualBranchClassifier
        train_loader: training DataLoader
        val_loader  : validation DataLoader
        test_loader : test DataLoader
        args        : parsed argparse namespace with training hyperparameters
        output_dir  : directory to save checkpoints and logs
        device      : torch.device (inferred from args.gpu if not provided)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        args,
        output_dir: str,
        device: Optional[torch.device] = None,
    ):
        self.args = args
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Device
        if device is None:
            if torch.cuda.is_available() and getattr(args, "gpu", 0) >= 0:
                device = torch.device(f"cuda:{args.gpu}")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # ---- Criterion ----
        label_smoothing = getattr(args, "label_smoothing", 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # ---- Optimizer ----
        self.optimizer = self._build_optimizer()

        # ---- Scheduler ----
        self.scheduler = self._build_scheduler()

        # ---- AMP ----
        self.use_amp = getattr(args, "amp", True) and device.type == "cuda"
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        # ---- Gradient clipping ----
        self.grad_clip = getattr(args, "grad_clip", 1.0)

        # ---- Early stopping ----
        patience = getattr(args, "patience", 15)
        self.early_stop = EarlyStopping(patience=patience, mode="min")

        # ---- History ----
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

        self.best_val_loss = math.inf
        self.start_epoch = 0
        self.total_train_time = 0.0

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr = getattr(self.args, "lr", 1e-4)
        wd = getattr(self.args, "weight_decay", 1e-4)
        opt_name = getattr(self.args, "optimizer", "adamw").lower()

        # Separate parameters for weight decay (skip BN/bias)
        decay_params, no_decay_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if opt_name == "adamw":
            return torch.optim.AdamW(param_groups, lr=lr)
        elif opt_name == "adam":
            return torch.optim.Adam(param_groups, lr=lr)
        elif opt_name == "sgd":
            return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self):
        sched_name = getattr(self.args, "scheduler", "cosine").lower()
        epochs = getattr(self.args, "epochs", 100)

        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-7
            )
        elif sched_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
            )
        elif sched_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

    # ------------------------------------------------------------------
    # Core training / validation / test loops
    # ------------------------------------------------------------------

    def _run_epoch(
        self, loader: DataLoader, phase: str
    ) -> Tuple[float, float]:
        """
        Run one epoch of training or evaluation.

        Returns:
            avg_loss : float
            accuracy : float (0-1)
        """
        is_train = phase == "train"
        self.model.train(is_train)

        total_loss, correct, total = 0.0, 0, 0
        desc = f"[{phase}]"

        with tqdm(loader, desc=desc, leave=False) as pbar:
            for images, masks, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images, masks)
                    loss = self.criterion(logits, labels)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                bs = labels.size(0)
                total_loss += loss.item() * bs
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += bs

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(total, 1), correct / max(total, 1)

    def train(self) -> dict:
        """
        Full training loop with early stopping and checkpointing.

        Returns:
            history dict with per-epoch loss/accuracy lists
        """
        epochs = getattr(self.args, "epochs", 100)
        t0 = time.time()

        for epoch in range(self.start_epoch, epochs):
            t_ep = time.time()

            train_loss, train_acc = self._run_epoch(self.train_loader, "train")
            val_loss, val_acc = self._run_epoch(self.val_loader, "val")

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log
            lr_now = self.optimizer.param_groups[0]["lr"]
            ep_time = time.time() - t_ep
            logger.info(
                "Epoch %3d/%d | "
                "train_loss=%.4f train_acc=%.4f | "
                "val_loss=%.4f val_acc=%.4f | "
                "lr=%.2e | %.1fs",
                epoch + 1, epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                lr_now, ep_time,
            )

            # History
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Checkpoint: always save last, conditionally save best
            self.save_checkpoint(epoch, is_best=(val_loss < self.best_val_loss))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            # Early stopping
            if self.early_stop(val_loss):
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

        self.total_train_time = time.time() - t0
        logger.info("Training complete. Total time: %.1fs", self.total_train_time)
        return self.history

    def test(self) -> Tuple[list, list, list]:
        """
        Evaluate the best saved model on the test set.

        Returns:
            y_true : list of true labels
            y_pred : list of predicted labels
            y_prob : list of positive-class probabilities
        """
        best_ckpt = os.path.join(self.output_dir, "best_model.pth")
        if os.path.isfile(best_ckpt):
            self.load_checkpoint(best_ckpt)
            logger.info("Loaded best model from %s", best_ckpt)

        self.model.eval()
        y_true, y_pred, y_prob = [], [], []

        t0 = time.time()
        with torch.no_grad():
            for images, masks, labels in tqdm(self.test_loader, desc="[test]", leave=False):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images, masks)

                probs = F.softmax(logits, dim=1)[:, 1]   # positive class prob
                preds = logits.argmax(dim=1)

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_prob.extend(probs.cpu().tolist())

        self.test_time = time.time() - t0
        return y_true, y_pred, y_prob

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model + optimiser + scheduler state."""
        state = {
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        if self.scheduler is not None:
            state["sched_state"] = self.scheduler.state_dict()

        last_path = os.path.join(self.output_dir, "last_model.pth")
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(state, best_path)

    def load_checkpoint(self, path: str):
        """Restore model weights (and optionally optimiser state) from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optim_state"])
        if "sched_state" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["sched_state"])
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.start_epoch = ckpt.get("epoch", 0)
        self.best_val_loss = ckpt.get("best_val_loss", math.inf)
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info("Resumed from checkpoint: %s (epoch %d)", path, self.start_epoch)
