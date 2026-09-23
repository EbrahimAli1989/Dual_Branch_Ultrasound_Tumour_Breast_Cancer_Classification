"""
Computational complexity analysis: FLOPs and parameter count.

Uses `thop` (pip install thop) when available, with a pure-PyTorch
fallback based on torch.profiler for environments where thop is absent.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of *trainable* parameters in the model.

    Returns:
        n_params: int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_summary(model: nn.Module) -> str:
    """Return a human-readable parameter count string."""
    n = count_parameters(model)
    if n >= 1e9:
        return f"{n / 1e9:.2f}B parameters"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M parameters"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K parameters"
    return f"{n} parameters"


# ---------------------------------------------------------------------------
# FLOPs computation
# ---------------------------------------------------------------------------

def compute_flops(
    model: nn.Module,
    image_size: Tuple[int, int, int] = (1, 224, 224),
    mask_size: Tuple[int, int, int] = (1, 224, 224),
    device: Optional[torch.device] = None,
) -> int:
    """
    Estimate multiply-add operations (FLOPs / MACs) for one forward pass.

    Tries `thop` first; falls back to a torch.profiler-based estimate.

    Args:
        model      : DualBranchClassifier or any nn.Module
        image_size : (C, H, W) of the image branch input
        mask_size  : (C, H, W) of the mask branch input
        device     : target device (defaults to CPU for profiling)

    Returns:
        flops: integer MACs (multiply-accumulate ops)
    """
    if device is None:
        device = torch.device("cpu")

    model_eval = model.eval()
    dummy_image = torch.zeros(1, *image_size, device=device)
    dummy_mask = torch.zeros(1, *mask_size, device=device)

    # ---- thop (preferred) ----
    try:
        from thop import profile as thop_profile, clever_format

        macs, params = thop_profile(
            model_eval,
            inputs=(dummy_image, dummy_mask),
            verbose=False,
        )
        flops = int(macs)
        logger.info(
            "FLOPs: %s | Params: %s",
            *clever_format([flops, params], "%.2f"),
        )
        return flops

    except ImportError:
        logger.warning(
            "thop not installed. Falling back to torch.profiler FLOPs estimate. "
            "Install with: pip install thop"
        )

    # ---- torch.profiler fallback ----
    try:
        from torch.profiler import profile, ProfilerActivity, record_function

        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with record_function("model_inference"):
                with torch.no_grad():
                    _ = model_eval(dummy_image, dummy_mask)

        total_flops = sum(
            e.flops for e in prof.key_averages() if hasattr(e, "flops") and e.flops > 0
        )
        logger.info("FLOPs (profiler estimate): %d", total_flops)
        return int(total_flops)

    except Exception as e:
        logger.warning("FLOPs computation failed: %s. Returning 0.", e)
        return 0


def print_complexity(model: nn.Module, device: Optional[torch.device] = None):
    """Convenience function: log and print model complexity."""
    n_params = count_parameters(model)
    flops = compute_flops(model, device=device)
    print(f"\nModel complexity:")
    print(f"  Trainable parameters : {parameter_summary(model)}")
    print(f"  FLOPs (one forward)  : {flops / 1e9:.3f} GFLOPs\n")
    return flops, n_params
