"""Fused GEGLU Triton kernel.

GEGLU: out = gate * GELU(proj)  where [gate, proj] = chunk(linear_out, 2, dim=-1)

The naive PyTorch path does 3 global memory round-trips:
  1. linear_out → memory  (from nn.Linear)
  2. read linear_out, compute GELU(proj) → memory
  3. read gate + gelu_proj, multiply → memory

This kernel fuses steps 2+3 into a single read + single write,
cutting elementwise memory traffic by ~2/3.

The Linear matmul itself (step 1) stays in cuBLAS and is NOT fused here.
We fuse the bias-add into this kernel to also eliminate the bias round-trip.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _geglu_fused_kernel(
    # Pointers
    matmul_out_ptr,   # [N, 2*D] — raw matmul output (no bias)
    bias_ptr,         # [2*D]    — Linear bias
    out_ptr,          # [N, D]   — fused output
    # Dimensions
    half_dim: tl.constexpr,  # D (= dim_out)
    full_dim: tl.constexpr,  # 2*D
    # Meta
    BLOCK_SIZE: tl.constexpr,
):
    """Each program processes BLOCK_SIZE output elements.

    Output layout: [N, D] flattened.  For output index i:
      row = i // D,  col = i % D
      gate = matmul_out[row, col]       + bias[col]
      proj = matmul_out[row, col + D]   + bias[col + D]
      out[i] = gate * GELU(proj)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Row and column in the output matrix [N, D]
    col = offsets % half_dim
    row = offsets // half_dim

    # Load gate (first half) and proj (second half) from matmul output
    gate_idx = row * full_dim + col
    proj_idx = gate_idx + half_dim

    gate = tl.load(matmul_out_ptr + gate_idx)
    proj = tl.load(matmul_out_ptr + proj_idx)

    # Fuse bias addition
    bias_gate = tl.load(bias_ptr + col)
    bias_proj = tl.load(bias_ptr + col + half_dim)
    # Promote to fp32 for GELU computation (matches PyTorch F.gelu behavior)
    gate_f = (gate + bias_gate).to(tl.float32)
    proj_f = (proj + bias_proj).to(tl.float32)

    # GELU(proj) = 0.5 * proj * (1 + erf(proj / sqrt(2)))
    gelu_proj = 0.5 * proj_f * (1.0 + tl.math.erf(proj_f * 0.7071067811865476))

    result = gate_f * gelu_proj
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty))


def geglu_fused(matmul_out: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused GEGLU: bias-add + chunk + GELU + gate multiply in one kernel.

    Args:
        matmul_out: [*, 2*D] raw matmul output (weight @ x, NO bias added yet)
        bias: [2*D] Linear bias vector

    Returns:
        [*, D] fused GEGLU output
    """
    orig_shape = matmul_out.shape
    half_dim = orig_shape[-1] // 2
    full_dim = orig_shape[-1]

    # Flatten to 2D: [N, 2*D]
    flat = matmul_out.reshape(-1, full_dim)
    N = flat.shape[0]
    n_elements = N * half_dim

    out = torch.empty(N, half_dim, device=matmul_out.device, dtype=matmul_out.dtype)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _geglu_fused_kernel[grid](
        flat, bias, out,
        half_dim, full_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Restore leading dimensions
    out_shape = orig_shape[:-1] + (half_dim,)
    return out.view(out_shape)
