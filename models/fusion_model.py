"""
DualBranchClassifier: top-level model that assembles encoders, attention,
and classifier into one end-to-end trainable module.

Forward pass:
    image  (B, 1, 224, 224)  →  US encoder  →  f_img  (B, 256)
    mask   (B, 1, 224, 224)  →  mask encoder →  f_mask (B, 256)
    attention_fusion(f_img, f_mask) → fused  (B, 512)
    classifier(fused)                → logits (B, 2)
"""

import torch
import torch.nn as nn
from typing import Optional

from models.us_encoders import get_us_encoder
from models.mask_encoders import get_mask_encoder
from models.attention import get_attention
from models.classifiers import get_classifier


class DualBranchClassifier(nn.Module):
    """
    Dual-branch deep learning model for binary tumour classification.

    Args:
        us_encoder   : nn.Module with forward(image) → (B, feature_dim)
        mask_encoder : nn.Module with forward(mask)  → (B, feature_dim)
        attention    : nn.Module with forward(f1, f2) → (B, 2*feature_dim)
        classifier   : nn.Module with forward(fused) → (B, num_classes)
        feature_dim  : per-branch output dimension (default 256)
        dropout      : dropout applied to each branch output before fusion
    """

    def __init__(
        self,
        us_encoder: nn.Module,
        mask_encoder: nn.Module,
        attention: nn.Module,
        classifier: nn.Module,
        feature_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.us_encoder = us_encoder
        self.mask_encoder = mask_encoder
        self.attention = attention
        self.classifier = classifier
        self.branch_drop = nn.Dropout(dropout)
        self.feature_dim = feature_dim

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image : (B, 1, 224, 224)  — normalised US image
            mask  : (B, 1, 224, 224)  — normalised tumour mask
        Returns:
            logits: (B, num_classes)
        """
        f_img = self.branch_drop(self.us_encoder(image))    # (B, feature_dim)
        f_mask = self.branch_drop(self.mask_encoder(mask))  # (B, feature_dim)
        fused = self.attention(f_img, f_mask)               # (B, 2*feature_dim)
        return self.classifier(fused)                       # (B, num_classes)

    def get_features(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> dict:
        """Return intermediate feature tensors for analysis/visualisation."""
        with torch.no_grad():
            f_img = self.us_encoder(image)
            f_mask = self.mask_encoder(mask)
            fused = self.attention(f_img, f_mask)
        return {"f_img": f_img, "f_mask": f_mask, "fused": fused}


# ---------------------------------------------------------------------------
# Model builder — called from train.py
# ---------------------------------------------------------------------------

def build_model(args) -> DualBranchClassifier:
    """
    Construct the full DualBranchClassifier from parsed argparse arguments.

    Relevant args attributes:
        us_encoder, mask_encoder, attention_type, classifier_type
        feature_dim, dropout, base_channels, vit_patch_size, vit_num_layers,
        vit_num_heads, vit_mlp_ratio, snn_timesteps, snn_threshold, snn_decay,
        attn_num_heads, mlp_hidden_dims, kan_hidden_dims, kan_grid_size,
        kan_spline_order, num_classes
    """
    feature_dim: int = getattr(args, "feature_dim", 256)
    dropout: float = getattr(args, "dropout", 0.1)

    # ---- US image encoder ----
    us_enc = get_us_encoder(
        encoder_type=args.us_encoder,
        feature_dim=feature_dim,
        base_channels=getattr(args, "base_channels", 32),
        patch_size=getattr(args, "vit_patch_size", 16),
        num_layers=getattr(args, "vit_num_layers", 4),
        num_heads=getattr(args, "vit_num_heads", 8),
        mlp_ratio=getattr(args, "vit_mlp_ratio", 4.0),
        dropout=dropout,
    )

    # ---- Mask encoder ----
    mask_enc = get_mask_encoder(
        encoder_type=args.mask_encoder,
        feature_dim=feature_dim,
        base_channels=getattr(args, "base_channels", 32),
        patch_size=getattr(args, "vit_patch_size", 16),
        num_layers=getattr(args, "vit_num_layers", 4),
        num_heads=getattr(args, "vit_num_heads", 8),
        mlp_ratio=getattr(args, "vit_mlp_ratio", 4.0),
        dropout=dropout,
        snn_timesteps=getattr(args, "snn_timesteps", 4),
        snn_threshold=getattr(args, "snn_threshold", 1.0),
        snn_decay=getattr(args, "snn_decay", 0.5),
    )

    # ---- Attention / fusion ----
    attn = get_attention(
        attention_type=args.attention_type,
        feature_dim=feature_dim,
        num_heads=getattr(args, "attn_num_heads", 8),
        dropout=dropout,
    )

    # ---- Classifier ----
    fused_dim = 2 * feature_dim  # 512
    clf = get_classifier(
        classifier_type=args.classifier_type,
        in_features=fused_dim,
        num_classes=getattr(args, "num_classes", 2),
        dropout=getattr(args, "dropout", 0.3),
        mlp_hidden_dims=getattr(args, "mlp_hidden_dims", [256, 128]),
        kan_hidden_dims=getattr(args, "kan_hidden_dims", [64]),
        kan_grid_size=getattr(args, "kan_grid_size", 5),
        kan_spline_order=getattr(args, "kan_spline_order", 3),
    )

    return DualBranchClassifier(
        us_encoder=us_enc,
        mask_encoder=mask_enc,
        attention=attn,
        classifier=clf,
        feature_dim=feature_dim,
        dropout=dropout,
    )
