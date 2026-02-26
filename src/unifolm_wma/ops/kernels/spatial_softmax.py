import math
import triton
import triton.language as tl


def get_rblock(args):
    n_pixels = args['HxW'] if 'HxW' in args else (args['H'] * args['W'])
    if n_pixels >= 4096:
        return 2048
    elif n_pixels >= 1024:
        return 1024
    else:
        return max(128, int(2 ** math.ceil(math.log2(n_pixels))))


@triton.jit
def SpatialSoftmax_NHWC_collect(
    in_ptr,
    in_bias,
    in_temperature,
    out_max,
    out_sum,
    N,
    HxW,
    C,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr
):
    # 2D grid: axis=0 -> channel blocks within one batch, axis=1 -> batch index
    ch_offset = tl.program_id(axis=0) * XBLOCK
    batch_idx = tl.program_id(axis=1)

    ch_index = ch_offset + tl.arange(0, XBLOCK)[None, :]  # (1, XBLOCK)
    ch_mask = ch_index < C

    bias = tl.load(in_bias + ch_index, mask=ch_mask, other=0.0)
    temperature = tl.load(in_temperature)

    x_start = tl.make_block_ptr(
        base=in_ptr + batch_idx * HxW * C,
        shape=(HxW, C),
        strides=(C, 1),
        offsets=(0, ch_offset),
        block_shape=(RBLOCK, XBLOCK),
        order=(1, 0)
    )

    # -- Pass 1: running max --
    rbase = tl.arange(0, RBLOCK)[:, None]      # (RBLOCK, 1)
    running_max = tl.full([1, XBLOCK], float("-inf"), tl.float32)
    x_local = x_start
    for i in range(0, HxW, RBLOCK):
        x = tl.load(x_local, boundary_check=(0, 1))       # (RBLOCK, XBLOCK)
        val = (x + bias) / temperature
        pixel_ok = (i + rbase) < HxW                       # (RBLOCK, 1)
        val = tl.where(pixel_ok, val, float("-inf"))
        local_maximum = tl.max(val, axis=0)                # (XBLOCK,)
        running_max = tl.maximum(running_max, local_maximum[None, :])
        x_local = tl.advance(x_local, (RBLOCK, 0))

    # -- Pass 2: running sum of exp --
    running_sum = tl.full([1, XBLOCK], 0.0, tl.float32)
    x_local = x_start
    for i in range(0, HxW, RBLOCK):
        x = tl.load(x_local, boundary_check=(0, 1))
        val = (x + bias) / temperature
        pixel_ok = (i + rbase) < HxW
        exp_val = tl.where(pixel_ok, tl.exp(val - running_max), 0.0)
        local_sum_exp = tl.sum(exp_val, axis=0)            # (XBLOCK,)
        running_sum += local_sum_exp[None, :]
        x_local = tl.advance(x_local, (RBLOCK, 0))

    out_idx = batch_idx * C + ch_index
    tl.store(out_max + out_idx, running_max, mask=ch_mask)
    tl.store(out_sum + out_idx, running_sum, mask=ch_mask)

@triton.jit
def SpatialSoftmax_NHWC_collect_onlinesoftmax(
    in_ptr,
    in_bias,
    in_temperature,
    out_max,
    out_sum,
    N,
    HxW,
    C,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr
):
    # 2D grid: axis=0 -> channel blocks within one batch, axis=1 -> batch index
    ch_offset = tl.program_id(axis=0) * XBLOCK
    batch_idx = tl.program_id(axis=1)

    ch_index = ch_offset + tl.arange(0, XBLOCK)[None, :]  # (1, XBLOCK)
    ch_mask = ch_index < C

    bias = tl.load(in_bias + ch_index, mask=ch_mask, other=0.0)
    temperature = tl.load(in_temperature)

    x_start = tl.make_block_ptr(
        base=in_ptr + batch_idx * HxW * C,
        shape=(HxW, C),
        strides=(C, 1),
        offsets=(0, ch_offset),
        block_shape=(RBLOCK, XBLOCK),
        order=(1, 0)
    )

    rbase = tl.arange(0, RBLOCK)[:, None]      # (RBLOCK, 1)
    running_max = tl.full([1, XBLOCK], float("-inf"), tl.float32)
    running_sum = tl.full([1, XBLOCK], 0.0, tl.float32)
    temp = tl.full([1, XBLOCK], 0.0, tl.float32)
    x_local = x_start
    for i in range(0, HxW, RBLOCK):
        x = tl.load(x_local, boundary_check=(0, 1))       # (RBLOCK, XBLOCK)
        val = (x + bias) / temperature
        pixel_ok = (i + rbase) < HxW                       # (RBLOCK, 1)
        val = tl.where(pixel_ok, val, float("-inf"))
        block_max = tl.max(val, axis=0)[None, :]

        new_max = tl.maximum(running_max, block_max)
        factor = tl.exp(running_max - new_max)

        current_exp = tl.exp(val - new_max)
        current_exp = tl.where(pixel_ok, current_exp, 0.0)
        current_sum = tl.sum(current_exp, axis=0)[None, :]

        running_sum = running_sum * factor + current_sum
        running_max = new_max

        x_local = tl.advance(x_local, (RBLOCK, 0))

    out_idx = batch_idx * C + ch_index
    tl.store(out_max + out_idx, running_max, mask=ch_mask)
    tl.store(out_sum + out_idx, running_sum, mask=ch_mask)


@triton.jit
def SpatialSoftmax_NHWC_normalize_expectation(
    in_ptr,
    in_bias,
    in_temperature,
    in_max,
    in_sum,
    out_expected_x,
    out_expected_y,
    N,
    H,
    W,
    C,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr
):
    # 2D grid: axis=0 -> channel blocks, axis=1 -> batch index
    ch_offset = tl.program_id(axis=0) * XBLOCK
    batch_idx = tl.program_id(axis=1)

    ch_index = ch_offset + tl.arange(0, XBLOCK)[None, :]
    ch_mask = ch_index < C

    bias = tl.load(in_bias + ch_index, mask=ch_mask, other=0.0)
    temperature = tl.load(in_temperature)

    x_start = tl.make_block_ptr(
        base=in_ptr + batch_idx * H * W * C,
        shape=(H * W, C),
        strides=(C, 1),
        offsets=(0, ch_offset),
        block_shape=(RBLOCK, XBLOCK),
        order=(1, 0)
    )

    out_idx = batch_idx * C + ch_index
    max_val = tl.load(in_max + out_idx, mask=ch_mask, other=0.0)
    sum_val = tl.load(in_sum + out_idx, mask=ch_mask, other=1.0)

    running_expected_x = tl.full([1, XBLOCK], 0.0, tl.float32)
    running_expected_y = tl.full([1, XBLOCK], 0.0, tl.float32)
    rbase = tl.arange(0, RBLOCK)[:, None]
    x_local = x_start
    for offset in range(0, H * W, RBLOCK):
        x = tl.load(x_local, boundary_check=(0, 1))
        prob = tl.exp((x + bias) / temperature - max_val) / sum_val
        pixel_idx = offset + rbase                          # (RBLOCK, 1)
        mask = pixel_idx < (H * W)
        prob = tl.where(mask, prob, 0.0)
        coord_x = pixel_idx % W
        coord_y = pixel_idx // W
        norm_x = -1.0 + 2.0 * coord_x.to(tl.float32) / (W - 1)
        norm_y = -1.0 + 2.0 * coord_y.to(tl.float32) / (H - 1)
        local_x_expected = norm_x * prob
        local_y_expected = norm_y * prob
        running_expected_x += tl.sum(local_x_expected, axis=0)[None, :]
        running_expected_y += tl.sum(local_y_expected, axis=0)[None, :]
        x_local = tl.advance(x_local, (RBLOCK, 0))

    tl.store(out_expected_x + out_idx, running_expected_x, mask=ch_mask)
    tl.store(out_expected_y + out_idx, running_expected_y, mask=ch_mask)
