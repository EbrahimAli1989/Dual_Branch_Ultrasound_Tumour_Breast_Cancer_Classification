"""
Ultrasound image encoders: CNN and Vision Transformer (ViT).

Both accept input of shape (B, 1, 224, 224) and produce a feature
vector of shape (B, feature_dim) where feature_dim defaults to 256.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """Two-layer residual block with optional projection shortcut."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + x)


# ---------------------------------------------------------------------------
# CNN Encoder
# ---------------------------------------------------------------------------

class CNNUSEncoder(nn.Module):
    """
    Convolutional encoder for grayscale US images (1 × 224 × 224).

    Architecture:
        Stage 1 : ConvBNReLU(1  → C)   + ResBlock + MaxPool  → C  × 112×112
        Stage 2 : ConvBNReLU(C  → 2C)  + ResBlock + MaxPool  → 2C ×  56× 56
        Stage 3 : ConvBNReLU(2C → 4C)  + ResBlock + MaxPool  → 4C ×  28× 28
        Stage 4 : ConvBNReLU(4C → 8C)  + ResBlock + MaxPool  → 8C ×  14× 14
        Stage 5 : ConvBNReLU(8C → 16C) + ResBlock + MaxPool  → 16C×   7×  7
        GlobalAvgPool → Flatten → Linear(16C, feature_dim)

    With base_channels=32 → 512 → 256 projection.
    """

    def __init__(self, feature_dim: int = 256, base_channels: int = 32):
        super().__init__()
        C = base_channels
        self.stages = nn.ModuleList([
            self._make_stage(1,    C,    pool=True),
            self._make_stage(C,    2*C,  pool=True),
            self._make_stage(2*C,  4*C,  pool=True),
            self._make_stage(4*C,  8*C,  pool=True),
            self._make_stage(8*C,  16*C, pool=True),
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * C, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, pool: bool = True) -> nn.Sequential:
        layers: List[nn.Module] = [ConvBNReLU(in_ch, out_ch), ResBlock(out_ch)]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 224, 224)
        Returns:
            features: (B, feature_dim)
        """
        for stage in self.stages:
            x = stage(x)
        x = self.gap(x)          # (B, 16C, 1, 1)
        return self.proj(x)      # (B, feature_dim)


# ---------------------------------------------------------------------------
# Vision Transformer (ViT) Encoder
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and linearly embed each.

    Args:
        img_size    : spatial size of the square input
        patch_size  : size of each square patch
        in_channels : number of input channels
        embed_dim   : embedding dimension per patch
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 256,
    ):
        super().__init__()
        assert img_size % patch_size == 0, \
            "Image size must be divisible by patch size."
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        # Single conv performs the patch split + linear embedding simultaneously
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        x = self.proj(x)                      # (B, embed_dim, H//P, W//P)
        x = x.flatten(2).transpose(1, 2)      # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Standard Pre-LN Transformer encoder block with multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerUSEncoder(nn.Module):
    """
    Vision Transformer encoder for grayscale US images (1 × 224 × 224).

    Architecture:
        PatchEmbed(patch_size=16)  → 196 tokens of dim embed_dim
        [CLS] token prepended      → 197 tokens
        Learnable positional embed
        num_layers × TransformerBlock
        CLS token extraction
        LayerNorm → Linear(embed_dim, feature_dim)

    With patch_size=16 on 224×224: num_patches = 196.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, feature_dim)
        self.feature_dim = feature_dim

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 224, 224)
        Returns:
            features: (B, feature_dim)
        """
        B = x.shape[0]
        x = self.patch_embed(x)                         # (B, num_patches, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)                  # (B, num_patches+1, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]                               # (B, embed_dim)
        return self.proj(cls_out)                       # (B, feature_dim)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_us_encoder(
    encoder_type: str,
    feature_dim: int = 256,
    base_channels: int = 32,
    patch_size: int = 16,
    num_layers: int = 4,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Return an encoder for ultrasound images.

    Args:
        encoder_type: "cnn" or "transformer"
        feature_dim : output feature vector size (default 256)
    """
    encoder_type = encoder_type.lower()
    if encoder_type == "cnn":
        return CNNUSEncoder(feature_dim=feature_dim, base_channels=base_channels)
    elif encoder_type in ("transformer", "vit"):
        return TransformerUSEncoder(
            feature_dim=feature_dim,
            embed_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            patch_size=patch_size,
        )
    else:
        raise ValueError(
            f"Unknown US encoder type '{encoder_type}'. Choose 'cnn' or 'transformer'."
        )
