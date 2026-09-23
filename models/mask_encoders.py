"""
Tumour mask encoders: CNN, Vision Transformer, and Spiking Neural Network (SNN).

All accept input of shape (B, 1, 224, 224) and produce (B, feature_dim).

SNN implementation uses custom Leaky Integrate-and-Fire (LIF) neurons
with a fast-sigmoid surrogate gradient so the model is end-to-end
trainable via standard back-propagation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use shared building blocks from the US encoder module
from models.us_encoders import (
    ConvBNReLU,
    ResBlock,
    TransformerUSEncoder,
    get_us_encoder,
)


# ===========================================================================
# CNN mask encoder (same topology as CNN US encoder, but independent weights)
# ===========================================================================

class CNNMaskEncoder(nn.Module):
    """
    Identical architecture to CNNUSEncoder but with its own weight set.
    Delegates to the factory in us_encoders for code-reuse.
    """

    def __init__(self, feature_dim: int = 256, base_channels: int = 32):
        super().__init__()
        # Import here to avoid circular import at module level
        from models.us_encoders import CNNUSEncoder
        self._enc = CNNUSEncoder(feature_dim=feature_dim, base_channels=base_channels)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._enc(x)


# ===========================================================================
# Transformer mask encoder
# ===========================================================================

class TransformerMaskEncoder(nn.Module):
    """
    Vision Transformer encoder for binary masks (shares ViT topology
    but has independent weights from the US image encoder).
    """

    def __init__(
        self,
        feature_dim: int = 256,
        patch_size: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._enc = TransformerUSEncoder(
            feature_dim=feature_dim,
            embed_dim=feature_dim,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._enc(x)


# ===========================================================================
# Spiking Neural Network (SNN) encoder
# ===========================================================================

# ---------------------------------------------------------------------------
# Surrogate gradient for the Heaviside spike function.
# During the forward pass we emit binary spikes; during the backward pass
# we substitute the derivative of a fast sigmoid centred at the threshold.
# ---------------------------------------------------------------------------

class _SpikeFn(torch.autograd.Function):
    """
    Spike generation with fast-sigmoid surrogate gradient.

    Forward : s = 1 if V >= threshold else 0
    Backward: ds/dV ≈ beta * sigmoid(beta*(V - threshold)) * (1 - sigmoid(...))
    """

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float = 1.0, beta: float = 5.0):
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.beta = beta
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        beta = ctx.beta
        # Fast-sigmoid surrogate
        sg = torch.sigmoid(beta * (membrane - threshold))
        surrogate = beta * sg * (1.0 - sg)
        return grad_output * surrogate, None, None


def spike(membrane: torch.Tensor, threshold: float = 1.0, beta: float = 5.0) -> torch.Tensor:
    """Functional wrapper for the differentiable spike operation."""
    return _SpikeFn.apply(membrane, threshold, beta)


# ---------------------------------------------------------------------------
# LIF neuron state container and layer
# ---------------------------------------------------------------------------

class LIFNeuronLayer(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer (stateless in the module sense;
    state is maintained externally and passed step-by-step).

    Membrane dynamics (discrete-time):
        V[t] = decay * V[t-1] * (1 - s[t-1]) + I[t]
        s[t] = spike(V[t], threshold)

    Hard reset: V is zeroed for spiking neurons after each step.
    """

    def __init__(self, threshold: float = 1.0, decay: float = 0.5, beta: float = 5.0):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.beta = beta

    def forward(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor,
        prev_spike: torch.Tensor,
    ):
        """
        Args:
            current    : synaptic input, any shape
            membrane   : membrane potential at t-1, same shape as current
            prev_spike : binary spikes at t-1, same shape

        Returns:
            new_spike   : spike tensor at t
            new_membrane: updated membrane potential at t
        """
        # Decay + hard reset + integrate
        new_membrane = self.decay * membrane * (1.0 - prev_spike) + current
        new_spike = spike(new_membrane, self.threshold, self.beta)
        return new_spike, new_membrane


# ---------------------------------------------------------------------------
# SNN convolutional block: Conv2d → BN → LIF
# ---------------------------------------------------------------------------

class SNNConvBlock(nn.Module):
    """Single convolutional layer followed by a LIF neuron layer."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        threshold: float = 1.0,
        decay: float = 0.5,
    ):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = LIFNeuronLayer(threshold=threshold, decay=decay)

    def forward(
        self,
        x: torch.Tensor,
        membrane: torch.Tensor,
        prev_spike: torch.Tensor,
    ):
        current = self.bn(self.conv(x))
        return self.lif(current, membrane, prev_spike)


# ---------------------------------------------------------------------------
# Full SNN Mask Encoder
# ---------------------------------------------------------------------------

class SNNMaskEncoder(nn.Module):
    """
    Spiking Neural Network encoder for binary tumour masks.

    Pipeline:
        1. Rate-encode the mask: at each of T timesteps, Bernoulli-sample
           spikes from the pixel-intensity probabilities (already in [0,1]).
        2. Feed spike trains through stacked SNN-Conv blocks.
        3. Average spike rates over all T timesteps.
        4. GlobalAvgPool → flatten → linear projection to feature_dim.

    Architecture (base_ch = C):
        SNNConvBlock(1  → C,   pool=True)   → C  × 112×112
        SNNConvBlock(C  → 2C,  pool=True)   → 2C ×  56× 56
        SNNConvBlock(2C → 4C,  pool=True)   → 4C ×  28× 28
        SNNConvBlock(4C → 8C,  pool=True)   → 8C ×  14× 14
        GlobalAvgPool → Flatten → Linear(8C, feature_dim)

    Args:
        feature_dim  : output vector dimension
        base_channels: base number of channels (C)
        timesteps    : number of SNN simulation steps (T)
        threshold    : LIF spike threshold
        decay        : LIF membrane potential decay factor
    """

    def __init__(
        self,
        feature_dim: int = 256,
        base_channels: int = 32,
        timesteps: int = 4,
        threshold: float = 1.0,
        decay: float = 0.5,
    ):
        super().__init__()
        C = base_channels
        self.T = timesteps
        self.threshold = threshold
        self.decay = decay

        # Channel progression per stage
        channels = [(1, C), (C, 2*C), (2*C, 4*C), (4*C, 8*C)]

        self.conv_blocks = nn.ModuleList([
            SNNConvBlock(inc, outc, threshold=threshold, decay=decay)
            for inc, outc in channels
        ])
        # Max pool after each stage
        self.pools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in channels])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * C, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

        # Shape book-keeping for membrane init
        self._stage_out_channels = [c[1] for c in channels]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _rate_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rate-coding: Bernoulli-sample T binary frames from pixel probabilities.

        Args:
            x: (B, 1, H, W) float in [0, 1]
        Returns:
            spikes: (T, B, 1, H, W) binary float
        """
        x = x.clamp(0.0, 1.0)
        # Expand along new time axis, then sample
        x_t = x.unsqueeze(0).expand(self.T, -1, -1, -1, -1)   # (T, B, 1, H, W)
        return torch.bernoulli(x_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 224, 224)
        Returns:
            features: (B, feature_dim)
        """
        B, C_in, H, W = x.shape
        spikes_in = self._rate_encode(x)   # (T, B, 1, H, W)

        # Compute spatial size after each pool for membrane init
        h, w = H, W
        spatial_sizes = []
        for _ in self.conv_blocks:
            spatial_sizes.append((h, w))
            h, w = h // 2, w // 2

        # Initialise membrane potentials and previous spikes (zeros)
        membranes = [
            torch.zeros(B, out_ch, *sz, device=x.device, dtype=x.dtype)
            for out_ch, sz in zip(self._stage_out_channels, spatial_sizes)
        ]
        prev_spikes = [torch.zeros_like(m) for m in membranes]

        # Accumulate output spikes for rate readout
        out_acc = torch.zeros(B, self._stage_out_channels[-1], h, w,
                              device=x.device, dtype=x.dtype)

        for t in range(self.T):
            feat = spikes_in[t]          # (B, 1, H, W)

            for i, (block, pool) in enumerate(zip(self.conv_blocks, self.pools)):
                new_spike, new_mem = block(feat, membranes[i], prev_spikes[i])
                membranes[i] = new_mem.detach()   # detach to avoid unbounded graph growth
                prev_spikes[i] = new_spike
                feat = pool(new_spike)

            # feat is now (B, 8C, 14, 14) — the output of the last pooled stage
            out_acc = out_acc + feat

        # Average firing rate
        out_rate = out_acc / self.T          # (B, 8C, 14, 14)
        out_rate = self.gap(out_rate)        # (B, 8C, 1, 1)
        return self.proj(out_rate)           # (B, feature_dim)


# ===========================================================================
# Factory
# ===========================================================================

def get_mask_encoder(
    encoder_type: str,
    feature_dim: int = 256,
    base_channels: int = 32,
    patch_size: int = 16,
    num_layers: int = 4,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    snn_timesteps: int = 4,
    snn_threshold: float = 1.0,
    snn_decay: float = 0.5,
) -> nn.Module:
    """
    Return a mask encoder.

    Args:
        encoder_type: "cnn" | "transformer" | "snn"
        feature_dim : output vector dimension
    """
    encoder_type = encoder_type.lower()
    if encoder_type == "cnn":
        return CNNMaskEncoder(feature_dim=feature_dim, base_channels=base_channels)
    elif encoder_type in ("transformer", "vit"):
        return TransformerMaskEncoder(
            feature_dim=feature_dim,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    elif encoder_type == "snn":
        return SNNMaskEncoder(
            feature_dim=feature_dim,
            base_channels=base_channels,
            timesteps=snn_timesteps,
            threshold=snn_threshold,
            decay=snn_decay,
        )
    else:
        raise ValueError(
            f"Unknown mask encoder type '{encoder_type}'. "
            "Choose 'cnn', 'transformer', or 'snn'."
        )
