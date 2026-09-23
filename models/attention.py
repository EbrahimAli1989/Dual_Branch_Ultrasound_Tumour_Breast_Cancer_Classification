"""
Attention-based feature fusion modules.

Both modules accept the two 256-dimensional branch feature vectors,
concatenate them into a 512-dimensional vector, apply an attention
mechanism, and return a 512-dimensional fused representation.

Self-attention  : treats the 512-dim concat as 2 tokens of dim 256,
                  applies multi-head self-attention, then flattens back.
Cross-attention : f_image attends to f_mask (and vice versa) in parallel,
                  then the two 256-dim attended vectors are concatenated.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Self-Attention Fusion
# ---------------------------------------------------------------------------

class SelfAttentionFusion(nn.Module):
    """
    Fuse two 256-dim feature vectors via multi-head self-attention.

    The two vectors [f1, f2] are treated as a sequence of 2 tokens
    (each of dimension `feature_dim`).  After self-attention + residual
    + layer-norm, the tokens are flattened back to 2*feature_dim.

    Architecture:
        concat([f1, f2]) → reshape to (B, 2, D)
        Pre-LN Multi-Head Self-Attention (MHSA) with skip connection
        LayerNorm
        Position-wise FFN (2-layer MLP, expansion=2) with skip connection
        LayerNorm
        Flatten → (B, 2*D)

    Args:
        feature_dim : dimension of each encoder's output (D)
        num_heads   : number of attention heads (D must be divisible by num_heads)
        dropout     : attention and FFN dropout probability
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert feature_dim % num_heads == 0, \
            f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"

        self.feature_dim = feature_dim
        self.out_dim = 2 * feature_dim  # 512

        # Positional embedding for the 2-token sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(feature_dim)
        hidden_dim = feature_dim * 2
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f1 : (B, feature_dim)  — US image branch features
            f2 : (B, feature_dim)  — mask branch features
        Returns:
            fused: (B, 2 * feature_dim)  — 512-dim fused representation
        """
        # Stack into sequence of 2 tokens: (B, 2, D)
        x = torch.stack([f1, f2], dim=1) + self.pos_embed   # (B, 2, D)

        # Pre-LN self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)

        # Pre-LN FFN with residual
        x = x + self.ffn(self.norm2(x))

        # Flatten the 2 tokens into one 512-dim vector
        return x.flatten(1)    # (B, 2*D)


# ---------------------------------------------------------------------------
# Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Fuse two 256-dim feature vectors via bidirectional cross-attention.

    Step 1 (image → mask): f1 as query, f2 as key/value
            → attended image features f1' (256-dim)
    Step 2 (mask → image): f2 as query, f1 as key/value
            → attended mask features  f2' (256-dim)

    Both steps share weights to halve parameter count.
    Final output: concat([f1', f2'])  →  512-dim

    Architecture for each cross-attention block:
        Query = x_q (1 token of dim D)
        Key/Value = x_kv (1 token of dim D)
        Pre-LN Multi-Head Cross-Attention + skip
        Pre-LN FFN + skip
        LayerNorm

    Args:
        feature_dim : dimension of each branch's feature vector (D)
        num_heads   : attention heads
        dropout     : dropout probability
        share_weights: if True, both directions share the same CA block
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        share_weights: bool = True,
    ):
        super().__init__()
        assert feature_dim % num_heads == 0

        self.feature_dim = feature_dim
        self.out_dim = 2 * feature_dim    # 512
        self.share_weights = share_weights

        self.cross_attn_1 = _CrossAttnBlock(feature_dim, num_heads, dropout)
        if share_weights:
            self.cross_attn_2 = self.cross_attn_1
        else:
            self.cross_attn_2 = _CrossAttnBlock(feature_dim, num_heads, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f1 : (B, feature_dim)  — US image branch
            f2 : (B, feature_dim)  — mask branch
        Returns:
            fused: (B, 2 * feature_dim)
        """
        # Unsqueeze to add sequence length of 1: (B, 1, D)
        q1 = f1.unsqueeze(1)   # image as query
        kv2 = f2.unsqueeze(1)  # mask as key/value
        q2 = f2.unsqueeze(1)
        kv1 = f1.unsqueeze(1)

        # Cross-attend: image features enriched by mask context
        f1_prime = self.cross_attn_1(q1, kv2).squeeze(1)   # (B, D)
        # Cross-attend: mask features enriched by image context
        f2_prime = self.cross_attn_2(q2, kv1).squeeze(1)   # (B, D)

        return torch.cat([f1_prime, f2_prime], dim=1)       # (B, 2*D)


class _CrossAttnBlock(nn.Module):
    """
    Single cross-attention block: query attends to key/value.
    Applies Pre-LN cross-attention + residual, then Pre-LN FFN + residual.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query : (B, Lq, D)
            kv    : (B, Lkv, D)
        Returns:
            out: (B, Lq, D)
        """
        q = self.norm_q(query)
        k = self.norm_kv(kv)
        attn_out, _ = self.attn(q, k, k)
        query = query + self.drop(attn_out)
        query = query + self.ffn(self.norm2(query))
        return query


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_attention(
    attention_type: str,
    feature_dim: int = 256,
    num_heads: int = 8,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Return a fusion attention module.

    Args:
        attention_type: "self" | "cross"
        feature_dim   : each encoder's output dimension
        num_heads     : number of attention heads
    Returns:
        module whose forward(f1, f2) → (B, 2*feature_dim)
    """
    attention_type = attention_type.lower()
    if attention_type == "self":
        return SelfAttentionFusion(
            feature_dim=feature_dim, num_heads=num_heads, dropout=dropout
        )
    elif attention_type == "cross":
        return CrossAttentionFusion(
            feature_dim=feature_dim, num_heads=num_heads, dropout=dropout
        )
    else:
        raise ValueError(
            f"Unknown attention type '{attention_type}'. Choose 'self' or 'cross'."
        )
