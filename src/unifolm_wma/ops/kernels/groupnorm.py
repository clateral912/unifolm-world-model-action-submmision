import triton
import triton.language as tl

@triton.jit
def welford_helper(
    mean_1,
    m2_1,
    n_1,
    mean_2,
    m2_2,
    n_2
):
    delta = mean_2 - mean_1
    new_n = n_1 + n_2
    new_mean = mean_1 + delta * (n_2 / new_n)
    new_m2 = m2_1 + m2_2 + delta * delta * (n_1 * n_2 / new_n)
    return new_mean, new_m2, new_n

@triton.jit
def GroupNorm_kernel(
    in_ptr,
    out_mean,
    out_m2,
    N,
    H,
    W,
    C,
    G,  # how many groups in a image
    K: tl.constexpr,  # how many channels in a group
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr
):
    # Process XBLOCK groups each program
    xoffset = tl.program_id(axis=0) * XBLOCK
    pid_batch = tl.program_id(axis=1)

    # Base pointer for this batch element (NHWC layout: stride between images = H*W*C)
    base = in_ptr + pid_batch * H * W * C

    # Group indices this program handles: (XBLOCK,)
    g_idx = xoffset + tl.arange(0, XBLOCK)  # (XBLOCK,)

    mean = tl.zeros([XBLOCK], dtype=tl.float32)
    m2 = tl.zeros([XBLOCK], dtype=tl.float32)
    cnt = tl.zeros([XBLOCK], dtype=tl.float32)

    # Flatten K into the reduction dim: iterate over H*W*K (pixel, channel) pairs.
    # 2D tiles (RBLOCK, XBLOCK) avoid 3D block_ptr layout bugs in Triton.
    total = H * W * K
    for i in range(0, total, RBLOCK):
        r_idx = i + tl.arange(0, RBLOCK)             # (RBLOCK,)
        pixel = r_idx // K                            # which pixel
        chan  = r_idx % K                             # which channel within group

        # Pointer: base + pixel * C + group * K + chan — NHWC layout
        ptr = base + pixel[:, None] * C + g_idx[None, :] * K + chan[:, None]
        valid = (r_idx[:, None] < total) & (g_idx[None, :] < G)

        val = tl.load(ptr, mask=valid, other=0.0).to(tl.float32)  # (RBLOCK, XBLOCK)

        block_sum   = tl.sum(val, axis=0)             # (XBLOCK,)
        block_sum_q = tl.sum(val * val, axis=0)       # (XBLOCK,)

        # Valid element count for this block (scalar)
        n_valid = tl.where(i + RBLOCK <= total, RBLOCK, total - i)
        block_count = n_valid

        block_mean = block_sum / block_count
        block_m2 = block_sum_q - (block_sum * block_sum / block_count)

        mean, m2, cnt = welford_helper(mean, m2, cnt, block_mean, block_m2, block_count)

    xindex = xoffset + tl.arange(0, XBLOCK)
    store_mask = xindex < G
    store_offset = pid_batch * G + xindex
    tl.store(out_mean + store_offset, mean, mask=store_mask)
    tl.store(out_m2 + store_offset, m2, mask=store_mask)

@triton.jit
def GroupNorm_persistent_mish(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    N,C,H,W,
    G, K,  # G: how many groups in a pixel. K: how many channels in a group
    eps,
    # C = G * K
    HxW: tl.constexpr,
    K_VAL: tl.constexpr,
    IS_NHWC: tl.constexpr = False,
):
    # C MUST be divisible with K
    rindex = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)

    if IS_NHWC:
        # (N, S, C) contiguous: group's K channels are contiguous at each spatial pos
        in_base = in_ptr + batch * H * W * C + rindex * K
        out_base = out_ptr + batch * H * W * C + rindex * K
        stride_s = C
        stride_k = 1
    else:
        # (N, C, S) contiguous: each channel's spatial elems are contiguous
        in_base = in_ptr + batch * C * H * W + rindex * K * H * W
        out_base = out_ptr + batch * C * H * W + rindex * K * H * W
        stride_s = 1
        stride_k = H * W

    x_ptr = tl.make_block_ptr(
        base=in_base,
        shape=(H * W, K),
        strides=(stride_s, stride_k),
        offsets=(0, 0),
        block_shape=(HxW, K_VAL),
        order=(0, 1)
    )

    out_block_ptr = tl.make_block_ptr(
        base=out_base,
        shape=(H * W, K),
        strides=(stride_s, stride_k),
        offsets=(0, 0),
        block_shape=(HxW, K_VAL),
        order=(0, 1)
    )

    coffset = rindex * K + tl.arange(0, K_VAL)
    w = tl.load(weight_ptr + coffset, mask=tl.arange(0, K_VAL) < K, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + coffset, mask=tl.arange(0, K_VAL) < K, other=0.0).to(tl.float32)

    val = tl.load(x_ptr, boundary_check=(0, 1)).to(tl.float32)

    sum = tl.sum(val)
    sum_q = tl.sum(val * val)
    cnt = H * W * K
    mean = sum / cnt
    m2 = sum_q - (sum * sum / cnt)
    variance = m2 / cnt
    rstd = 1.0 / tl.sqrt(variance + eps)

    val = ((val - mean) * rstd) * w[None, :] + b[None, :]

    softplus = tl.where(val > 20.0, val, tl.log(1.0 + tl.exp(val)))
    result = val * (2.0 * tl.sigmoid(2.0 * softplus) - 1.0)

    tl.store(out_block_ptr, result, boundary_check=(0, 1))

@triton.jit
def GroupNorm_reduce(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    N, C, H, W,
    G, K,  # G: how many groups in a pixel. K: how many channels in a group
    eps,
    # C = G * K
    HxW: tl.constexpr,
    K_VAL: tl.constexpr,
    RBLOCK: tl.constexpr,
    FUSED_SILU: tl.constexpr,
    IS_NHWC: tl.constexpr = False,
):
    # C MUST be divisible with K
    rindex = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)

    if IS_NHWC:
        in_base = in_ptr + batch * H * W * C + rindex * K
        out_base = out_ptr + batch * H * W * C + rindex * K
        stride_s = C
        stride_k = 1
    else:
        in_base = in_ptr + batch * C * H * W + rindex * K * H * W
        out_base = out_ptr + batch * C * H * W + rindex * K * H * W
        stride_s = 1
        stride_k = H * W

    x_ptr = tl.make_block_ptr(
        base=in_base,
        shape=(H * W, K),
        strides=(stride_s, stride_k),
        offsets=(0, 0),
        block_shape=(RBLOCK, K_VAL),
        order=(0, 1)
    )

    coffset = rindex * K + tl.arange(0, K_VAL)
    w = tl.load(weight_ptr + coffset, mask=tl.arange(0, K_VAL) < K, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + coffset, mask=tl.arange(0, K_VAL) < K, other=0.0).to(tl.float32)

    acc_sum = 0.0
    acc_sum_q = 0.0

    for i in range(0, H * W, RBLOCK):
        val = tl.load(x_ptr, boundary_check=(0, 1)).to(tl.float32)
        acc_sum += tl.sum(val)
        acc_sum_q += tl.sum(val * val)

        x_ptr = tl.advance(x_ptr, [RBLOCK, 0])

    num_elements = H * W * K
    mean = acc_sum / num_elements
    variance = (acc_sum_q / num_elements) - (mean * mean)
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Reset pointer
    x_ptr = tl.make_block_ptr(
        base=in_base,
        shape=(H * W, K), strides=(stride_s, stride_k),
        offsets=(0, 0), block_shape=(RBLOCK, K_VAL), order=(0, 1)
    )

    out_ptr_block = tl.make_block_ptr(
        base=out_base,
        shape=(H * W, K), strides=(stride_s, stride_k),
        offsets=(0, 0), block_shape=(RBLOCK, K_VAL), order=(0, 1)
    )

    # Hopefully we will hit L2 Cache when calling tl.load
    for i in range(0, H * W, RBLOCK):
        val = tl.load(x_ptr, boundary_check=(0, 1)).to(tl.float32)

        val = ((val - mean) * rstd) * w[None, :] + b[None, :]

        if FUSED_SILU:
            val = val * tl.sigmoid(val)
        tl.store(out_ptr_block, val, boundary_check=(0, 1))
        x_ptr = tl.advance(x_ptr, [RBLOCK, 0])
        out_ptr_block = tl.advance(out_ptr_block, [RBLOCK, 0])


@triton.jit
def GroupNorm_reduce_stats(
    in_ptr,
    partial_sum_ptr,
    partial_sum_q_ptr,
    N, C, H, W,
    G, K,
    num_splits,
    K_VAL: tl.constexpr,
    RBLOCK: tl.constexpr,
    IS_NHWC: tl.constexpr = False,
):
    rindex = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)
    split = tl.program_id(axis=2)

    if IS_NHWC:
        in_base = in_ptr + batch * H * W * C + rindex * K
        stride_s = C
        stride_k = 1
    else:
        in_base = in_ptr + batch * C * H * W + rindex * K * H * W
        stride_s = 1
        stride_k = H * W

    x_ptr = tl.make_block_ptr(
        base=in_base,
        shape=(H * W, K),
        strides=(stride_s, stride_k),
        offsets=(split * RBLOCK, 0),
        block_shape=(RBLOCK, K_VAL),
        order=(0, 1)
    )

    val = tl.load(x_ptr, boundary_check=(0, 1)).to(tl.float32)

    acc_sum = tl.sum(val)
    acc_sum_q = tl.sum(val * val)

    idx = batch * G * num_splits + rindex * num_splits + split
    tl.store(partial_sum_ptr + idx, acc_sum)
    tl.store(partial_sum_q_ptr + idx, acc_sum_q)


@triton.jit
def GroupNorm_reduce_norm(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    partial_sum_ptr,
    partial_sum_q_ptr,
    N, C, H, W,
    G, K,
    eps,
    num_splits,
    K_VAL: tl.constexpr,
    RBLOCK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    FUSED_SILU: tl.constexpr,
    IS_NHWC: tl.constexpr = False,
):
    rindex = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)
    split = tl.program_id(axis=2)

    # Aggregate all partials for this (batch, group)
    base_idx = batch * G * num_splits + rindex * num_splits
    offsets = base_idx + tl.arange(0, NUM_SPLITS)
    mask = tl.arange(0, NUM_SPLITS) < num_splits

    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)
    sum_qs = tl.load(partial_sum_q_ptr + offsets, mask=mask, other=0.0)

    acc_sum = tl.sum(sums)
    acc_sum_q = tl.sum(sum_qs)

    num_elements = H * W * K
    mean = acc_sum / num_elements
    variance = (acc_sum_q / num_elements) - (mean * mean)
    rstd = 1.0 / tl.sqrt(variance + eps)

    coffset = rindex * K + tl.arange(0, K_VAL)
    w = tl.load(weight_ptr + coffset, mask=tl.arange(0, K_VAL) < K, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + coffset, mask=tl.arange(0, K_VAL) < K, other=0.0).to(tl.float32)

    if IS_NHWC:
        in_base = in_ptr + batch * H * W * C + rindex * K
        out_base = out_ptr + batch * H * W * C + rindex * K
        stride_s = C
        stride_k = 1
    else:
        in_base = in_ptr + batch * C * H * W + rindex * K * H * W
        out_base = out_ptr + batch * C * H * W + rindex * K * H * W
        stride_s = 1
        stride_k = H * W

    x_ptr = tl.make_block_ptr(
        base=in_base,
        shape=(H * W, K),
        strides=(stride_s, stride_k),
        offsets=(split * RBLOCK, 0),
        block_shape=(RBLOCK, K_VAL),
        order=(0, 1)
    )

    out_ptr_block = tl.make_block_ptr(
        base=out_base,
        shape=(H * W, K),
        strides=(stride_s, stride_k),
        offsets=(split * RBLOCK, 0),
        block_shape=(RBLOCK, K_VAL),
        order=(0, 1)
    )

    val = tl.load(x_ptr, boundary_check=(0, 1)).to(tl.float32)

    val = ((val - mean) * rstd) * w[None, :] + b[None, :]

    if FUSED_SILU:
        val = val * tl.sigmoid(val)

    tl.store(out_ptr_block, val, boundary_check=(0, 1))
