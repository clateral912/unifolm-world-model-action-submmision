"""Enable fused GEGLU kernel on GEGLU modules.

Replaces the naive chunk→GELU→mul with a single Triton kernel that
fuses bias-add + chunk + GELU + gate-multiply, reducing elementwise
memory traffic by ~2/3.

Usage:
    from unifolm_wma.ops.wrappers.fused_geglu import enable_fused_geglu
    n = enable_fused_geglu(model.model.diffusion_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_fused_forward(original_module: nn.Module):
    """Create a fused forward method for a GEGLU module.

    The original GEGLU.forward does:
        x, gate = self.proj(x).chunk(2, dim=-1)   # Linear + chunk
        return x * F.gelu(gate)                     # gelu + mul

    The fused version:
        matmul_out = F.linear(x, self.proj.weight)  # matmul only (no bias)
        return geglu_fused(matmul_out, self.proj.bias)  # bias+chunk+gelu+mul
    """
    from unifolm_wma.ops.kernels.geglu import geglu_fused

    proj = original_module.proj

    def fused_forward(x):
        # cuBLAS matmul (no bias — bias is fused into the Triton kernel)
        matmul_out = F.linear(x, proj.weight)
        return geglu_fused(matmul_out, proj.bias)

    return fused_forward


def enable_fused_geglu(model: nn.Module) -> int:
    """Replace GEGLU.forward with fused Triton kernel on all GEGLU instances.

    Args:
        model: nn.Module tree to scan for GEGLU modules.

    Returns:
        Number of GEGLU modules patched.
    """
    from unifolm_wma.modules.attention import GEGLU

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, GEGLU):
            module.forward = _make_fused_forward(module)
            count += 1
    return count
