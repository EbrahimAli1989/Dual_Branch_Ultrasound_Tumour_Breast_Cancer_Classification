"""
Classification heads: MLP and KAN (Kolmogorov–Arnold Network).

Both accept (B, in_features) and produce (B, num_classes) logits.

KAN Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024).
Implementation: efficient B-spline KAN layer (no external library required).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ===========================================================================
# MLP Classifier
# ===========================================================================

class MLPClassifier(nn.Module):
    """
    Multi-layer Perceptron classifier with BatchNorm, GELU, and Dropout.

    Architecture:
        in_features → hidden[0] → ... → hidden[-1] → num_classes
        Each hidden layer: Linear → BN1d → GELU → Dropout

    Args:
        in_features  : input dimension (typically 512)
        hidden_dims  : list of hidden layer widths
        num_classes  : number of output classes (2 for binary)
        dropout      : dropout probability
    """

    def __init__(
        self,
        in_features: int = 512,
        hidden_dims: List[int] = (256, 128),
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = in_features

        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_features)
        Returns:
            logits: (B, num_classes)
        """
        return self.net(x)


# ===========================================================================
# KAN — Kolmogorov-Arnold Network Classifier
# ===========================================================================

class KANLinear(nn.Module):
    """
    A single KAN layer.

    Each output unit j computes:
        y_j = sum_i  phi_{i,j}(x_i)

    where:
        phi_{i,j}(x) = w_base_{i,j} * b(x)  +  w_spline_{i,j} * spline(x)
        b(x)         = SiLU(x) = x * sigmoid(x)   (residual basis)
        spline(x)    = sum_k  c_{i,j,k} * B_k^p(x)  (B-spline of order p)

    The B-spline basis is defined over a uniform grid on [-1, 1] extended by
    `spline_order` knots on each side.

    Args:
        in_features    : number of input features
        out_features   : number of output features
        grid_size      : number of interior grid intervals
        spline_order   : order of the B-spline (3 = cubic)
        scale_noise    : noise scale for weight initialisation
        scale_base     : std of base weight initialisation
        scale_spline   : scale applied to spline coefficients at init
        enable_standalone_scale_spline: learn per-output scaling factor
        base_activation: activation for the residual path (default SiLU)
        grid_eps       : mixing factor for adaptive grid update
        grid_range     : domain of the B-spline grid
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: nn.Module = nn.SiLU,
        grid_range: List[float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        # Grid: uniform knots extended by spline_order on each side
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        # Shape: (grid_size + 2*spline_order + 1,)
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.empty(out_features, in_features)
            )
        else:
            self.spline_scaler = None

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self._curve2coeff(
                    self.grid[self.spline_order : -self.spline_order],   # interior points
                    noise,
                )
            )
        if self.enable_standalone_scale_spline:
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis values.

        Args:
            x: (B, in_features)
        Returns:
            bases: (B, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        x = x.unsqueeze(-1)         # (B, in_features, 1)
        grid = self.grid             # (n_knots,)

        # Zeroth-order basis: indicator for each interval
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()  # (B, in, n_intervals)

        # Cox–de Boor recursion to build higher-order splines
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[: -(k + 1)]
            left_den = grid[k:-1] - grid[: -(k + 1)]
            left = left_num / (left_den + 1e-8) * bases[..., :-1]

            right_num = grid[k + 1:] - x
            right_den = grid[k + 1:] - grid[1:(-k)]
            right = right_num / (right_den + 1e-8) * bases[..., 1:]

            bases = left + right

        # bases: (B, in_features, grid_size + spline_order)
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fit spline coefficients given evaluation points x and values y.

        Args:
            x : (n_points,)
            y : (n_points, in_features, out_features)
        Returns:
            coeffs: (out_features, in_features, n_points)
        """
        # Evaluate basis at x points: (n_points, in_features, n_coeffs)
        A = self.b_splines(x.unsqueeze(1).expand(-1, self.in_features))
        # A: (n_points, in_features, n_coeffs)
        # y: (n_points, in_features, out_features)
        # Solve least-squares: A^T A c = A^T y
        A_T = A.permute(1, 2, 0)   # (in, n_coeffs, n_pts)
        y_T = y.permute(1, 2, 0)   # (in, out, n_pts) → need (in, n_pts, out)
        y_T = y.permute(1, 0, 2)   # (in, n_pts, out)

        solution = torch.linalg.lstsq(
            A.permute(1, 0, 2),    # (in, n_pts, n_coeffs)
            y_T,                   # (in, n_pts, out)
        ).solution                 # (in, n_coeffs, out)

        return solution.permute(2, 0, 1)  # (out, in, n_coeffs)

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_features)
        Returns:
            out: (B, out_features)
        """
        # Clamp to grid domain for numerical stability
        x_clamped = x.clamp(float(self.grid[0]), float(self.grid[-1]))

        # Residual (base) path: w_base * SiLU(x)
        base_out = F.linear(self.base_activation(x), self.base_weight)   # (B, out)

        # Spline path: sum_i w_spline_{i,j} * B_i(x_j)
        spline_basis = self.b_splines(x_clamped)                         # (B, in, n_coeff)
        # (B, in, n_coeff) × (out, in, n_coeff)^T → manual einsum
        spline_out = torch.einsum(
            "bik,oik->bo", spline_basis, self.scaled_spline_weight
        )                                                                  # (B, out)

        return base_out + spline_out

    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Adaptive grid update: refit the grid based on input distribution.
        Call after a warmup phase for better coverage.
        """
        with torch.no_grad():
            x_sorted, _ = x.sort(dim=0)
            n = x.size(0)
            # Quantile-based knot placement
            q = torch.linspace(0, 1, self.grid_size + 1, device=x.device)
            idx = (q * (n - 1)).long().clamp(0, n - 1)
            new_grid_inner = x_sorted[idx, :].mean(dim=1)  # rough quantile

            # Extend with order-many ghost points on each side
            step = (new_grid_inner[-1] - new_grid_inner[0]) / self.grid_size
            left = new_grid_inner[0] - step * self.spline_order
            right = new_grid_inner[-1] + step * self.spline_order
            left_ext = torch.linspace(
                left.item(), new_grid_inner[0].item(), self.spline_order + 1, device=x.device
            )[:-1]
            right_ext = torch.linspace(
                new_grid_inner[-1].item(), right.item(), self.spline_order + 1, device=x.device
            )[1:]
            new_grid = torch.cat([left_ext, new_grid_inner, right_ext])
            self.grid.data.copy_(new_grid)


class KANClassifier(nn.Module):
    """
    KAN-based classification head.

    Stacks KANLinear layers with optional hidden widths, followed by
    a standard linear output layer (logits, no activation).

    Architecture:
        KANLinear(in_features → hidden[0])
        KANLinear(hidden[0] → hidden[1]) ...
        nn.Linear(hidden[-1] → num_classes)  [standard linear head]

    Args:
        in_features  : input dimension
        hidden_dims  : list of KAN hidden layer widths
        num_classes  : output dimension
        grid_size    : B-spline grid size per KAN layer
        spline_order : B-spline order per KAN layer
    """

    def __init__(
        self,
        in_features: int = 512,
        hidden_dims: List[int] = (64,),
        num_classes: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = in_features

        for h in hidden_dims:
            layers.append(
                KANLinear(prev_dim, h, grid_size=grid_size, spline_order=spline_order)
            )
            prev_dim = h

        self.kan_layers = nn.ModuleList(layers)
        self.head = nn.Linear(prev_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_features)
        Returns:
            logits: (B, num_classes)
        """
        for layer in self.kan_layers:
            x = layer(x)
        return self.head(x)

    def update_grids(self, x: torch.Tensor):
        """Update all KAN layer grids from a representative batch."""
        with torch.no_grad():
            for layer in self.kan_layers:
                if isinstance(layer, KANLinear):
                    layer.update_grid(x)
                    # Pass through for next layer
                    x = layer(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_classifier(
    classifier_type: str,
    in_features: int = 512,
    num_classes: int = 2,
    dropout: float = 0.3,
    mlp_hidden_dims: List[int] = (256, 128),
    kan_hidden_dims: List[int] = (64,),
    kan_grid_size: int = 5,
    kan_spline_order: int = 3,
) -> nn.Module:
    """
    Return a classification head.

    Args:
        classifier_type: "mlp" | "kan"
        in_features    : input dimension (typically 512)
        num_classes    : number of output classes
    """
    classifier_type = classifier_type.lower()
    if classifier_type == "mlp":
        return MLPClassifier(
            in_features=in_features,
            hidden_dims=mlp_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif classifier_type == "kan":
        return KANClassifier(
            in_features=in_features,
            hidden_dims=kan_hidden_dims,
            num_classes=num_classes,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )
    else:
        raise ValueError(
            f"Unknown classifier type '{classifier_type}'. Choose 'mlp' or 'kan'."
        )
