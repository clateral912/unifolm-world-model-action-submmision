"""Fused GroupNorm wrappers (Apex NHWC CUDA + Triton fallback).

Covers all GroupNorm usage in the codebase:
  - GroupNorm + SiLU  → Apex NHWC CUDA (channels_last) / Triton (NCHW)
  - GroupNorm (no act) → Apex NHWC CUDA (channels_last) / Triton (NCHW)
  - GroupNorm + Mish   → Triton only (no Apex Mish support)

Usage:
    from unifolm_wma.ops.wrappers.fused_groupnorm import (
        enable_fused_groupnorm,
        FusedGroupNormSiLU,
        FusedGroupNormMish,
        FusedGroupNorm,
    )
    # Auto-replace all GroupNorm in a model:
    enable_fused_groupnorm(model)
"""
import torch
import torch.nn as nn

from unifolm_wma.ops.kernels.groupnorm import (
    GroupNorm_persistent_mish,
    GroupNorm_reduce,
    GroupNorm_reduce_stats,
    GroupNorm_reduce_norm,
)

# Apex NHWC GroupNorm CUDA kernel (enabled by default, graceful fallback to Triton)
# Set APEX_NHWC_GN=0 to force-disable Apex
import os as _os
_HAS_APEX_GN = False
if _os.environ.get("APEX_NHWC_GN", "1") != "0":
    try:
        import group_norm_cuda as _apex_gn_cuda
        _HAS_APEX_GN = True
    except ImportError:
        pass

# Channels/groups supported by Apex one-pass or two-pass kernels
_APEX_SUPPORTED_CHANNELS = frozenset([
    128, 256, 320, 384, 448, 512, 640, 768, 896, 960, 1024, 1280, 1344,
    1536, 1792, 1920, 2048, 2240, 2560, 2688, 3072, 3136, 3584, 4096,
])
_APEX_SUPPORTED_GROUPS = frozenset([16, 32])


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def _can_use_apex(x, num_groups, num_channels):
    """Check if Apex NHWC kernel is usable for this input."""
    return (
        _HAS_APEX_GN
        and x.ndim == 4
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.is_contiguous(memory_format=torch.channels_last)
        and num_channels in _APEX_SUPPORTED_CHANNELS
        and num_groups in _APEX_SUPPORTED_GROUPS
    )


def _apex_group_norm(x, weight, bias, num_groups, eps, with_silu):
    """Call Apex NHWC GroupNorm CUDA kernel directly (bypasses torch.library).

    Uses passes=1 (cooperative one-pass kernel) for deterministic results.
    The two-pass algorithm (passes=2) uses atomicAdd for cross-block reduction,
    which causes non-determinism due to floating-point addition order.
    """
    y, _sums = _apex_gn_cuda.forward(x, num_groups, weight, bias, eps, 1, with_silu)
    return y


def triton_group_norm(x, weight, bias, num_groups, eps=1e-5):
    """GroupNorm without fused activation."""
    if _can_use_apex(x, num_groups, weight.numel()):
        return _apex_group_norm(x, weight, bias, num_groups, eps, False)
    return _launch_reduce(x, weight, bias, num_groups, eps, fused_silu=False)


def triton_group_norm_silu(x, weight, bias, num_groups, eps=1e-5):
    """GroupNorm + SiLU fused."""
    if _can_use_apex(x, num_groups, weight.numel()):
        return _apex_group_norm(x, weight, bias, num_groups, eps, True)
    return _launch_reduce(x, weight, bias, num_groups, eps, fused_silu=True)


def triton_group_norm_mish(x, weight, bias, num_groups, eps=1e-5):
    """GroupNorm + Mish fused (persistent single-pass). Triton only — no Apex Mish."""
    return _launch_persistent_mish(x, weight, bias, num_groups, eps)


# ---------------------------------------------------------------------------
# Internal launchers
# ---------------------------------------------------------------------------

def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _detect_channels_last(x):
    """Detect channels_last (4D) or channels_last_3d (5D) memory format."""
    if x.ndim == 4 and x.is_contiguous(memory_format=torch.channels_last):
        return True
    if x.ndim == 5 and x.is_contiguous(memory_format=torch.channels_last_3d):
        return True
    return False


def _launch_reduce(x, weight, bias, num_groups, eps, fused_silu):
    orig_shape = x.shape
    orig_dtype = x.dtype
    is_cl = _detect_channels_last(x)
    N = x.shape[0]
    C = x.shape[1]

    # 5D cl3d → 4D CL → Apex fast path (zero-copy views, no layout conversion)
    # Apex NHWC kernel only supports fp16/bf16 inputs
    if (is_cl and x.ndim == 5 and _HAS_APEX_GN
            and x.dtype in (torch.float16, torch.bfloat16)
            and C in _APEX_SUPPORTED_CHANNELS
            and num_groups in _APEX_SUPPORTED_GROUPS):
        T, H, W = orig_shape[2], orig_shape[3], orig_shape[4]
        # Zero-copy view: (N,C,T,H,W) cl3d → (N,C,T*H,W) cl
        x_4d = x.as_strided(
            (N, C, T * H, W),
            (x.stride(0), x.stride(1), x.stride(3), x.stride(4))
        )
        y_4d = _apex_group_norm(x_4d, weight, bias, num_groups, eps, fused_silu)
        # Zero-copy view: (N,C,T*H,W) cl → (N,C,T,H,W) cl3d
        return y_4d.as_strided(
            orig_shape,
            (y_4d.stride(0), y_4d.stride(1),
             H * y_4d.stride(2), y_4d.stride(2), y_4d.stride(3))
        )

    K = C // num_groups
    S = 1
    for d in orig_shape[2:]:
        S *= d

    # NCHW kernel (stride_s=1) is 3.5x faster than NHWC (stride_s=C) due to
    # cache line utilization. For channels_last input, use copy_() to fuse
    # dtype + layout conversion into a single CUDA kernel (2 copies total
    # instead of 4).
    if is_cl:
        x_flat = torch.empty(N, C, S, dtype=torch.float32, device=x.device)
        x_flat.view(orig_shape).copy_(x)
    else:
        x_flat = x.float().reshape(N, C, S).contiguous()
    out = torch.empty_like(x_flat)

    K_VAL = _next_pow2(K)
    MAX_TILE = 8192
    RBLOCK = max(min(_next_pow2(S), MAX_TILE // K_VAL), 32)
    num_splits = (S + RBLOCK - 1) // RBLOCK

    if num_splits > 1 and num_groups * N < 128:
        _launch_reduce_split(
            x_flat, out, weight, bias,
            N, C, S, num_groups, K, eps,
            K_VAL, RBLOCK, num_splits, fused_silu,
        )
    else:
        grid = (num_groups, N)
        GroupNorm_reduce[grid](
            x_flat, out, weight, bias,
            N, C, S, 1,
            num_groups, K,
            eps,
            HxW=S,
            K_VAL=K_VAL,
            RBLOCK=RBLOCK,
            FUSED_SILU=fused_silu,
        )

    if is_cl:
        fmt = torch.channels_last if x.ndim == 4 else torch.channels_last_3d
        result = torch.empty(orig_shape, dtype=orig_dtype, device=x.device,
                             memory_format=fmt)
        result.copy_(out.view(orig_shape))
    else:
        result = out.reshape(orig_shape).to(orig_dtype)
    return result


def _launch_reduce_split(x_flat, out, weight, bias,
                          N, C, S, num_groups, K, eps,
                          K_VAL, RBLOCK, num_splits, fused_silu):
    """Two-phase split-spatial: stats → normalize."""
    NUM_SPLITS = _next_pow2(num_splits)

    partial_sum = torch.empty(N * num_groups * num_splits,
                              device=x_flat.device, dtype=torch.float32)
    partial_sum_q = torch.empty_like(partial_sum)

    grid = (num_groups, N, num_splits)

    GroupNorm_reduce_stats[grid](
        x_flat, partial_sum, partial_sum_q,
        N, C, S, 1,
        num_groups, K,
        num_splits,
        K_VAL=K_VAL,
        RBLOCK=RBLOCK,
    )

    GroupNorm_reduce_norm[grid](
        x_flat, out, weight, bias,
        partial_sum, partial_sum_q,
        N, C, S, 1,
        num_groups, K,
        eps,
        num_splits,
        K_VAL=K_VAL,
        RBLOCK=RBLOCK,
        NUM_SPLITS=NUM_SPLITS,
        FUSED_SILU=fused_silu,
    )


def _launch_persistent_mish(x, weight, bias, num_groups, eps):
    orig_shape = x.shape
    orig_dtype = x.dtype
    is_cl = _detect_channels_last(x)
    N = x.shape[0]
    C = x.shape[1]
    K = C // num_groups
    S = 1
    for d in orig_shape[2:]:
        S *= d

    if is_cl:
        x_flat = torch.empty(N, C, S, dtype=torch.float32, device=x.device)
        x_flat.view(orig_shape).copy_(x)
    else:
        x_flat = x.float().reshape(N, C, S).contiguous()
    out = torch.empty_like(x_flat)

    S_BLOCK = _next_pow2(S)
    K_VAL = _next_pow2(K)

    grid = (num_groups, N)
    GroupNorm_persistent_mish[grid](
        x_flat, out, weight, bias,
        N, C, S, 1,              # H=S, W=1
        num_groups, K,
        eps,
        HxW=S_BLOCK,
        K_VAL=K_VAL,
    )

    if is_cl:
        fmt = torch.channels_last if x.ndim == 4 else torch.channels_last_3d
        result = torch.empty(orig_shape, dtype=orig_dtype, device=x.device,
                             memory_format=fmt)
        result.copy_(out.view(orig_shape))
    else:
        result = out.reshape(orig_shape).to(orig_dtype)
    return result


# ---------------------------------------------------------------------------
# nn.Module wrappers  (drop-in replacements)
# ---------------------------------------------------------------------------

class FusedGroupNorm(nn.GroupNorm):
    """Drop-in replacement for nn.GroupNorm — Triton reduce kernel."""

    def forward(self, x):
        return triton_group_norm(x, self.weight, self.bias,
                                 self.num_groups, self.eps)


class FusedGroupNormSiLU(nn.Module):
    """Replaces nn.Sequential(GroupNorm, SiLU) with a single fused kernel."""

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return triton_group_norm_silu(x, self.weight, self.bias,
                                      self.num_groups, self.eps)


class FusedGroupNormMish(nn.Module):
    """Replaces nn.Sequential(GroupNorm, Mish) with a single fused kernel."""

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return triton_group_norm_mish(x, self.weight, self.bias,
                                      self.num_groups, self.eps)


# ---------------------------------------------------------------------------
# Model conversion utility
# ---------------------------------------------------------------------------

def _copy_gn_params(src, dst):
    """Copy weight/bias from an nn.GroupNorm to a fused module."""
    dst.weight = src.weight
    dst.bias = src.bias


def _replace_sequential_gn(seq):
    """Scan a Sequential and fuse GroupNorm+activation pairs in-place.

    Patterns detected:
      [GroupNorm, SiLU, ...]     → [FusedGroupNormSiLU, ...]
      [GroupNorm, Mish, ...]     → [FusedGroupNormMish, ...]
      [Conv*, GroupNorm, Mish]   → [Conv*, FusedGroupNormMish]   (Conv1dBlock)
    """
    modules = list(seq.children())
    new_modules = []
    i = 0
    changed = False
    while i < len(modules):
        m = modules[i]
        nxt = modules[i + 1] if i + 1 < len(modules) else None

        if isinstance(m, nn.GroupNorm) and isinstance(nxt, nn.SiLU):
            fused = FusedGroupNormSiLU(m.num_groups, m.num_channels, m.eps)
            _copy_gn_params(m, fused)
            new_modules.append(fused)
            i += 2
            changed = True
        elif isinstance(m, nn.GroupNorm) and isinstance(nxt, nn.Mish):
            fused = FusedGroupNormMish(m.num_groups, m.num_channels, m.eps)
            _copy_gn_params(m, fused)
            new_modules.append(fused)
            i += 2
            changed = True
        else:
            new_modules.append(m)
            i += 1

    if changed:
        return nn.Sequential(*new_modules)
    return seq


def enable_fused_groupnorm(model):
    """Walk *model* and replace GroupNorm (± activation) with Triton fused kernels.

    Handles:
      1. Sequential(GroupNorm, SiLU, ...) → Sequential(FusedGroupNormSiLU, ...)
      2. Sequential(GroupNorm, Mish, ...) → Sequential(FusedGroupNormMish, ...)
      3. Standalone nn.GroupNorm           → FusedGroupNorm

    Returns the number of modules replaced.
    """
    count = 0

    # Pass 1: fuse GroupNorm+activation inside Sequential containers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Sequential):
            new_seq = _replace_sequential_gn(module)
            if new_seq is not module:
                # Set on parent
                parts = name.rsplit('.', 1)
                parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
                attr = parts[-1]
                setattr(parent, attr, new_seq)
                count += 1

    # Pass 2: standalone GroupNorm (not inside a Sequential that was already handled)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.GroupNorm) and not isinstance(module, FusedGroupNorm):
            fused = FusedGroupNorm(module.num_groups, module.num_channels, module.eps)
            _copy_gn_params(module, fused)
            parts = name.rsplit('.', 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            attr = parts[-1]
            setattr(parent, attr, fused)
            count += 1

    return count


def enable_channels_last(model):
    """Convert Conv2d (4D) and Conv3d (5D) weights to channels_last memory format.

    - 4D parameters (Conv2d weights) → channels_last
    - 5D parameters (Conv3d weights) → channels_last_3d
    - Other dimensions (Conv1d, Linear, bias) → unchanged

    Returns the number of parameters converted.
    """
    count = 0
    for name, param in model.named_parameters():
        if param.ndim == 4:
            param.data = param.data.contiguous(memory_format=torch.channels_last)
            count += 1
        elif param.ndim == 5:
            param.data = param.data.contiguous(memory_format=torch.channels_last_3d)
            count += 1
    return count
