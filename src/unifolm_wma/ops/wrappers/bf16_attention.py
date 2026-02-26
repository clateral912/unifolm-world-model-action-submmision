"""Utility to enable bf16 attention on specific UNet levels.

Usage:
    from unifolm_wma.ops.wrappers.bf16_attention import enable_bf16_attention
    enable_bf16_attention(wma_model, target_levels=[0, 1, 2])
"""
import torch
from unifolm_wma.modules.attention import SpatialTransformer, TemporalTransformer


def enable_bf16_attention(model, target_levels=None, dtype=torch.bfloat16):
    """Enable bf16 (or other dtype) attention for specific UNet levels.

    Sets `compute_dtype` on SpatialTransformer and TemporalTransformer modules
    within the WMAModel backbone.  When `compute_dtype` is set, their forward
    methods wrap proj_in -> attention -> proj_out with torch.amp.autocast,
    while GroupNorm stays in fp32.

    Args:
        model: WMAModel (or any nn.Module containing input_blocks / middle_block
               / output_blocks).
        target_levels: list[int] | 'all' | None.
            - None / 'all': enable on every SpatialTransformer / TemporalTransformer.
            - list[int]: UNet level indices to enable.
              Level 0 = highest resolution (320ch), Level 1 = 640ch, etc.
        dtype: torch dtype for autocast (default torch.bfloat16).

    Returns:
        int: number of modules modified.
    """
    if target_levels is None:
        target_levels = 'all'

    count = 0

    if target_levels == 'all':
        for module in model.modules():
            if isinstance(module, (SpatialTransformer, TemporalTransformer)):
                module.compute_dtype = dtype
                count += 1
        return count

    # Build level -> ds mapping from config:
    # channel_mult = [1, 2, 4, 4], each level except last adds a Downsample.
    # ds doubles at each downsample: level0=ds1, level1=ds2, level2=ds4, level3=ds8.
    # attention_resolutions = [4, 2, 1] means attention at ds in {1, 2, 4}.
    # Encoder input_blocks layout (with num_res_blocks=2):
    #   block 0: Conv2d (no attention)
    #   level 0: blocks 1,2 (ResBlock + SpatialTF + TemporalTF each)
    #            block 3: Downsample
    #   level 1: blocks 4,5
    #            block 6: Downsample
    #   level 2: blocks 7,8
    #            block 9: Downsample
    #   level 3: blocks 10,11 (no attention)
    # Middle block: always has attention (level = 'middle')
    # Decoder output_blocks: reverse order.

    # Strategy: walk input_blocks / middle_block / output_blocks,
    # determine the UNet level of each block, and set compute_dtype
    # on matching SpatialTransformer / TemporalTransformer submodules.

    target_set = set(target_levels)

    def _set_dtype_on_transformers(module):
        nonlocal count
        for sub in module.modules():
            if isinstance(sub, (SpatialTransformer, TemporalTransformer)):
                sub.compute_dtype = dtype
                count += 1

    # Determine num_res_blocks and channel_mult from model structure
    if not hasattr(model, 'input_blocks'):
        # Fallback: just set on all matching modules
        for module in model.modules():
            if isinstance(module, (SpatialTransformer, TemporalTransformer)):
                module.compute_dtype = dtype
                count += 1
        return count

    channel_mult = model.channel_mult
    num_res_blocks = model.num_res_blocks

    # ---- Encoder ----
    block_idx = 1  # skip block 0 (initial conv)
    for level, mult in enumerate(channel_mult):
        for _ in range(num_res_blocks):
            if level in target_set:
                _set_dtype_on_transformers(model.input_blocks[block_idx])
            block_idx += 1
        if level != len(channel_mult) - 1:
            block_idx += 1  # skip Downsample block

    # ---- Middle block (same level as deepest attention level) ----
    # Middle block is at ds = 2^(len(channel_mult)-1).
    # We include it if the deepest attention level is in target_set.
    deepest_attn_level = len(channel_mult) - 2  # level with smallest resolution that has attn
    if deepest_attn_level in target_set:
        _set_dtype_on_transformers(model.middle_block)

    # ---- Decoder ----
    out_idx = 0
    for level, mult in list(enumerate(channel_mult))[::-1]:
        for i in range(num_res_blocks + 1):
            if level in target_set:
                _set_dtype_on_transformers(model.output_blocks[out_idx])
            out_idx += 1

    return count
