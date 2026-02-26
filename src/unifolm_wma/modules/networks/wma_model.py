import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch import Tensor
from functools import partial
from abc import abstractmethod
from einops import rearrange
from omegaconf import OmegaConf
from typing import Optional, Sequence, Any, Tuple, Union, List, Dict
from collections.abc import Mapping, Iterable, Callable

from unifolm_wma.utils.diffusion import timestep_embedding
from unifolm_wma.utils.common import checkpoint
from unifolm_wma.utils.basics import (zero_module, conv_nd, linear,
                                      avg_pool_nd, normalization)
from unifolm_wma.modules.attention import SpatialTransformer, TemporalTransformer
from unifolm_wma.utils.utils import instantiate_from_config



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, batch_size=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, batch_size=batch_size)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, TemporalTransformer):
                '''
                为了利用 GPU 并行计算 2D 卷积，输入 x 在大部分时候被 reshape 成了 ((B*F), C, H, W),
                把时间维 F (Frames) 和 Batch B 混合在了一起。2D 卷积层根本不知道哪张图是第1帧,哪张是第2帧。

                TemporalTransformer 必须知道时间顺序。
                所以这里显式地调用 rearrange 将维度还原回 5D (B, C, F, H, W)，让层能看到 F 维度。
                '''
                x = rearrange(x, '(b f) c h w -> b c f h w', b=batch_size)
                x = layer(x, context)
                x = rearrange(x, 'b c f h w -> (b f) c h w')
                x = x.contiguous(memory_format=torch.channels_last)
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        # 跨步采样
        stride = 2 if dims != 3 else (1, 2, 2)
        # 要么用卷积, 要么用池化
        if use_conv:
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,
                              stride=stride,
                              padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims,
                                self.channels,
                                self.out_channels,
                                3,
                                padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            # 最近邻插值（Nearest Neighbor）
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                            mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            # 平滑插值带来的伪影。由于最近邻插值会导致信号在高频段产生强烈的谐波，
            # 这里的卷积层充当了一个低通滤波器，学习如何填充这些新增像素的语义信息
            x = self.conv(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param use_temporal_conv: if True, use the temporal convolution.
    :param use_image_dataset: if True, the temporal parameters will not be optimized.
    """

    def __init__(self,
                 channels,
                 emb_channels,
                 dropout,
                 out_channels=None,
                 use_scale_shift_norm=False,
                 dims=2,
                 use_checkpoint=False,
                 use_conv=False,
                 up=False,
                 down=False,
                 use_temporal_conv=False,
                 tempspatial_aware=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        # h (Hidden / Main Path): 主路径, 经过线性代数变换
        # x (Identity / Skip Path): 原始输入
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims,
                                           channels,
                                           self.out_channels,
                                           3,
                                           padding=1)
        else:
            # 1x1 卷积本质上就是对每一个像素位置的通道向量做了一次全连接层
            self.skip_connection = conv_nd(dims, channels, self.out_channels,
                                           1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                spatial_aware=tempspatial_aware)

        # [WMA-HPC] Pruned dispatch: for the common case in this model config
        # (no updown, no scale_shift_norm), bypass checkpoint/partial/branching.
        if not self.updown and not self.use_scale_shift_norm:
            self.forward = self._pruned_forward

    def _pruned_forward(self, x, emb, batch_size=None):
        """Pruned ResBlock path: no updown, no scale_shift_norm.

        Eliminates checkpoint wrapper, partial allocation, branch checks,
        while loop for emb_out dims, and NVTX overhead.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = self.out_layers(h + emb_out[..., None, None])
        h = self.skip_connection(x) + h
        if self.use_temporal_conv and batch_size:
            h = rearrange(h, '(b t) c h w -> b c t h w', b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, 'b c t h w -> (b t) c h w')
            h = h.contiguous(memory_format=torch.channels_last)
        return h

    def forward(self, x, emb, batch_size=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        input_tuple = (x, emb)
        if batch_size:
            forward_batchsize = partial(self._forward, batch_size=batch_size)
            return checkpoint(forward_batchsize, input_tuple,
                              self.parameters(), self.use_checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(),
                          self.use_checkpoint)

    def _forward(self, x, emb, batch_size=None):
        # 注意! ResBlock的输入是4D的! batch size和视频帧t被压缩为了一个维度: (b t) c h w
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # scale_shift 模式多了一个乘法步骤和一倍的 emb 维度，但它赋予了模型极强的控制力。
        # 它允许时间步 $t$ 直接关闭某些特征通道（scale = -1）或放大某些特征通道，这在生成精细纹理时至关重要
        if self.use_scale_shift_norm:
            # out_norm 取出第一个元素，通常是 GroupNorm 层
            # out_rest 取出剩下的层（SiLU 激活、Dropout、Conv 卷积）。
            # 我们需要在归一化之后、进入后续卷积之前，强行插入"缩放和偏移"操作。
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            # 输入 emb_out：形状为 $(B, 2 \times C)$ 的 2D 张量。torch.chunk(..., 2, dim=1)：
            # 在内存中将这个连续的向量从中间"劈开"，分成两段。scale :
            # 形状 (B, C)。shift : 形状 (B, C)。
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            # 对 h执行 GroupNorm。输出 h_norm 形状仍为 (B * T, C, H, W)。
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            # 直接将时间信息加到特征图上。这只改变了特征的"偏置"，而不能改变特征的"增益（Gain）"
            h = h + emb_out
            h = self.out_layers(h)

        h = self.skip_connection(x) + h

        if self.use_temporal_conv and batch_size:
            h = rearrange(h, '(b t) c h w -> b c t h w', b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, 'b c t h w -> (b t) c h w')
            h = h.contiguous(memory_format=torch.channels_last)
        return h

def _global_temporal_conv_impl(x, conv1, conv2, conv3, conv4):
    # 捕获 identity 用于残差
    identity = x

    # 纯净的卷积链路
    x = conv1(x)
    x = conv2(x)
    x = conv3(x)
    x = conv4(x)

    # [HPC] 将残差加法也包含在编译范围内
    # 这样 Inductor 可以将 Element-wise Add 融合进最后一个 Conv 的 Epilogue
    return identity + x

# =========================================================================
# [WMA-HPC] 全局编译句柄
# =========================================================================
_COMPILED_TEMP_CONV_FN = None

def _get_compiled_temp_conv_fn():
    global _COMPILED_TEMP_CONV_FN
    if _COMPILED_TEMP_CONV_FN is None:
        # 使用对应的环境变量控制
        if os.getenv("WMA_COMPILE_TEMPORAL_CONV_BLOCK") == "1":
            print("[WMA-HPC] Initializing GLOBAL compiled kernel for TemporalConvBlock...")
            _COMPILED_TEMP_CONV_FN = torch.compile(
                _global_temporal_conv_impl,
                mode="reduce-overhead",
                fullgraph=True
            )
        else:
            _COMPILED_TEMP_CONV_FN = _global_temporal_conv_impl
    return _COMPILED_TEMP_CONV_FN


# =========================================================================
# 修改后的类定义
# =========================================================================
class TemporalConvBlock(nn.Module):
    """
    Adapted from modelscope: https://github.com/modelscope/modelscope/blob/master/modelscope/models/multi_modal/video_synthesis/unet_sd.py
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 dropout=0.0,
                 spatial_aware=False):
        super(TemporalConvBlock, self).__init__()
        self._is_compiled = False
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Kernel & Padding Config
        th_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 3, 1)
        th_padding_shape = (1, 0, 0) if not spatial_aware else (1, 1, 0)
        tw_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 1, 3)
        tw_padding_shape = (1, 0, 0) if not spatial_aware else (1, 0, 1)

        # Layers Definition
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels), nn.SiLU(),
            nn.Conv3d(in_channels,
                      out_channels,
                      th_kernel_shape,
                      padding=th_padding_shape))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels,
                      in_channels,
                      tw_kernel_shape,
                      padding=tw_padding_shape))
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels,
                      in_channels,
                      th_kernel_shape,
                      padding=th_padding_shape))
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels,
                      in_channels,
                      tw_kernel_shape,
                      padding=tw_padding_shape))

        # Zero initialization for identity behavior
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        # 1. 获取全局唯一的编译函数
        fused_fn = _get_compiled_temp_conv_fn()

        # 2. 调用核心计算
        # 显式传入所有子模块，Dynamo 会自动追踪它们的参数
        # 移除了所有手动的 Layout 转换，完全信任 Inductor 对 Conv3d 的优化
        out = fused_fn(
            x,
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4
        )

        return out


class WMAModel(nn.Module):
    """
    The full World-Model-Action model.
    """

    def __init__(self,
                 in_channels: int,
                 model_channels: int,
                 out_channels: int,
                 num_res_blocks: int,
                 attention_resolutions: Sequence[int],
                 dropout: float = 0.0,
                 channel_mult: Sequence[int] = (1, 2, 4, 8),
                 conv_resample: bool = True,
                 dims: int = 2,
                 context_dim: int | None = None,
                 use_scale_shift_norm: bool = False,
                 resblock_updown: bool = False,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 transformer_depth: int = 1,
                 use_linear: bool = False,
                 use_checkpoint: bool = False,
                 temporal_conv: bool = False,
                 tempspatial_aware: bool = False,
                 temporal_attention: bool = True,
                 use_relative_position: bool = True,
                 use_causal_attention: bool = False,
                 temporal_length: int | None = None,
                 use_fp16: bool = False,
                 addition_attention: bool = False,
                 temporal_selfatt_only: bool = True,
                 image_cross_attention: bool = False,
                 cross_attention_scale_learnable: bool = False,
                 default_fs: int = 4,
                 fs_condition: bool = False,
                 n_obs_steps: int = 1,
                 num_stem_token: int = 1,
                 unet_head_config: OmegaConf | None = None,
                 stem_process_config: OmegaConf | None = None,
                 base_model_gen_only: bool = False):
        """
        Initialize the World-Model-Action network.

        Args:
            in_channels: Number of input channels to the backbone.
            model_channels: Base channel width for the UNet/backbone.
            out_channels: Number of output channels.
            num_res_blocks: Number of residual blocks per resolution stage.
            attention_resolutions: Resolutions at which to enable attention.
            dropout: Dropout probability used inside residual/attention blocks.
            channel_mult: Multipliers for channels at each resolution level.
            conv_resample: If True, use convolutional resampling for up/down sampling.
            dims: Spatial dimensionality of the backbone (1/2/3).
            context_dim: Optional context embedding dimension (for cross-attention).
            use_scale_shift_norm: Enable scale-shift (FiLM-style) normalization in blocks.
            resblock_updown: Use residual blocks for up/down sampling (instead of plain conv).
            num_heads: Number of attention heads (if >= 0). If -1, derive from num_head_channels.
            num_head_channels: Channels per attention head (if >= 0). If -1, derive from num_heads.
            transformer_depth: Number of transformer/attention blocks per stage.
            use_linear: Use linear attention variants where applicable.
            use_checkpoint: Enable gradient checkpointing in blocks to save memory.
            temporal_conv: Include temporal convolution along the time dimension.
            tempspatial_aware: If True, use time-space aware blocks.
            temporal_attention: Enable temporal self-attention.
            use_relative_position: Use relative position encodings in attention.
            use_causal_attention: Use causal (uni-directional) attention along time.
            temporal_length: Optional maximum temporal length expected by the model.
            use_fp16: Enable half-precision layers/normalization where supported.
            addition_attention: Add auxiliary attention modules.
            temporal_selfatt_only: Restrict attention to temporal-only (no spatial) if True.
            image_cross_attention: Enable cross-attention with image embeddings.
            cross_attention_scale_learnable: Make cross-attention scaling a learnable parameter.
            default_fs: Default frame-stride / fps.
            fs_condition: If True, condition on frame-stride/fps features.
            n_obs_steps: Number of observed steps used in conditioning heads.
            num_stem_token: Number of stem tokens for action tokenization.
            unet_head_config: OmegaConf for UNet heads (e.g., action/state heads).
            stem_process_config: OmegaConf for stem/preprocessor module.
            base_model_gen_only: Perform the generation using the base model with out action and state outputs.
        """

        super(WMAModel, self).__init__()
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        time_embed_dim = model_channels * 4
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        temporal_self_att_only = True
        self.addition_attention = addition_attention
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        self.cross_attention_scale_learnable = cross_attention_scale_learnable
        self.default_fs = default_fs
        self.fs_condition = fs_condition
        self.n_obs_steps = n_obs_steps
        self.num_stem_token = num_stem_token
        self.base_model_gen_only = base_model_gen_only

        # Time embedding blocks
        # 去噪声时间步序列的MLP多层感知机
        self.time_embed = nn.Sequential(
            # 第一层: 投影 (Projection)
            # 输入: (B, 320) -> 输出: (B, 1280)
            linear(model_channels, time_embed_dim),
            #激活函数
            nn.SiLU(),
            # 第二层: 混合 (Mixing)
            # 输入: (B, 1280) -> 输出: (B, 1280)  <-- 注意这里！不是变回 320
            linear(time_embed_dim, time_embed_dim),
        )
        if fs_condition:
            self.fps_embedding = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
            # 一种防守型的初始化策略，确保新加入的控制模块（FPS）
            # 在最开始不会破坏原有系统的稳定性，让模型从"无视 FPS"平滑过渡到"理解 FPS"。
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)
        # Input Block
        # 第一层输入, 首先将输入通道数通过卷积扩展到模型维度, dims为2指定使用nn.Conv2D
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        if self.addition_attention:
            # 开启 init_attn 的目的是：高频细节的时间一致性锁定。
            # U-Net 的深层（低分辨率）处理的是低频、语义信息（比如"这是一只猫"）。
            # U-Net 的浅层（高分辨率）处理的是高频、纹理信息（比如"猫毛的走向"、"光斑的闪烁"）。
            # 问题： 如果只在深层做时序 Attention，模型能保证"这只猫在每一帧都在"，
            # 但很难保证"每一帧的猫毛纹理都不闪烁"。因为高频信息在下采样过程中丢失了，深层的时序层"看不见"这些细节。
            self.init_attn = TimestepEmbedSequential(
                TemporalTransformer(model_channels,
                                    n_heads=8,
                                    d_head=num_head_channels,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    use_checkpoint=use_checkpoint,
                                    only_self_att=temporal_selfatt_only,
                                    causal_attention=False,
                                    relative_position=use_relative_position,
                                    temporal_length=temporal_length))

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch,
                             time_embed_dim,
                             dropout,
                             out_channels=mult * model_channels,
                             dims=dims,
                             use_checkpoint=use_checkpoint,
                             use_scale_shift_norm=use_scale_shift_norm,
                             tempspatial_aware=tempspatial_aware,
                             use_temporal_conv=temporal_conv)
                ]
                ch = mult * model_channels
                # 判断当前层级是否需要添加注意力模块
                # ds 代表当前的空间下采样倍数（Downsample Factor）。
                # 只有当当前的缩放比例在用户定义的 attention_resolutions 列表中时，程序才会执行后面的 MHA 参数计算
                # channels = num_head * head_dim
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        # 固定头数 (Fixed Head Count)
                        # 用户在外部指定了num_heads
                        dim_head = ch // num_heads
                    else:
                        # 固定每头维度 (Fixed Head Dimension)
                        # 用户指定了每个头的固定维度 num_head_channels（如常见的 64 或 80）
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            use_checkpoint=use_checkpoint,
                            disable_self_attn=False,
                            video_length=temporal_length,
                            agent_state_context_len=self.n_obs_steps,
                            agent_action_context_len=self.temporal_length *
                            num_stem_token,
                            image_cross_attention=self.image_cross_attention,
                            cross_attention_scale_learnable=self.
                            cross_attention_scale_learnable,
                        ))
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                use_checkpoint=use_checkpoint,
                                only_self_att=temporal_self_att_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch,
                                 time_embed_dim,
                                 dropout,
                                 out_channels=out_ch,
                                 dims=dims,
                                 use_checkpoint=use_checkpoint,
                                 use_scale_shift_norm=use_scale_shift_norm,
                                 down=True)
                        if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        layers = [
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm,
                     tempspatial_aware=tempspatial_aware,
                     use_temporal_conv=temporal_conv),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                use_linear=use_linear,
                use_checkpoint=use_checkpoint,
                disable_self_attn=False,
                video_length=temporal_length,
                agent_state_context_len=self.n_obs_steps,
                agent_action_context_len=self.temporal_length * num_stem_token,
                image_cross_attention=self.image_cross_attention,
                cross_attention_scale_learnable=self.
                cross_attention_scale_learnable)
        ]
        if self.temporal_attention:
            layers.append(
                TemporalTransformer(ch,
                                    num_heads,
                                    dim_head,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    use_linear=use_linear,
                                    use_checkpoint=use_checkpoint,
                                    only_self_att=temporal_self_att_only,
                                    causal_attention=use_causal_attention,
                                    relative_position=use_relative_position,
                                    temporal_length=temporal_length))
        layers.append(
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm,
                     tempspatial_aware=tempspatial_aware,
                     use_temporal_conv=temporal_conv))

        # Middle Block
        self.middle_block = TimestepEmbedSequential(*layers)

        # Output Block
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # ich 代表的是 来自编码器（Encoder/Input Blocks）对应层级的跳跃连接（Skip Connection）通道数 。
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich,
                             time_embed_dim,
                             dropout,
                             out_channels=mult * model_channels,
                             dims=dims,
                             use_checkpoint=use_checkpoint,
                             use_scale_shift_norm=use_scale_shift_norm,
                             tempspatial_aware=tempspatial_aware,
                             use_temporal_conv=temporal_conv)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            use_checkpoint=use_checkpoint,
                            disable_self_attn=False,
                            video_length=temporal_length,
                            agent_state_context_len=self.n_obs_steps,
                            image_cross_attention=self.image_cross_attention,
                            cross_attention_scale_learnable=self.
                            cross_attention_scale_learnable))
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                use_checkpoint=use_checkpoint,
                                only_self_att=temporal_self_att_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch,
                                 time_embed_dim,
                                 dropout,
                                 out_channels=out_ch,
                                 dims=dims,
                                 use_checkpoint=use_checkpoint,
                                 use_scale_shift_norm=use_scale_shift_norm,
                                 up=True)
                        if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # Action and state prediction unet
        unet_head_config['params']['context_dims'] = [
            mult * model_channels for mult in channel_mult
        ]
        self.action_unet = instantiate_from_config(unet_head_config)
        self.state_unet = instantiate_from_config(unet_head_config)

        # Initialize action token_projector
        self.action_token_projector = instantiate_from_config(
            stem_process_config)

        # Parallel streams for action_unet / state_unet (fork-join)
        self._stream_action = torch.cuda.Stream()
        self._stream_state = torch.cuda.Stream()
        self._export_mode = False  # Set True for torch.export (disables CUDA streams)

        # CUDA Graph state
        self._cuda_graph = None
        self._cuda_graph_warmup_count = 0
        self._CUDA_GRAPH_WARMUP_STEPS = 3
        self._static_x = None
        self._static_x_action = None
        self._static_x_state = None
        self._static_timesteps = None
        self._static_context = None
        self._static_context_action = None
        self._static_fs = None

        # [WMA-HPC] KV Cache state: cache processed context and per-layer KV
        self._kv_cache_mode = False
        self._kv_step_zero_pending = False
        self._current_ddim_step = 0  # Set by DDIM loop for per-step graph selection
        self._context_cache = None
        self._static_output = None

        # [WMA-HPC] Dual CUDA Graph state (prefill + decode, pre-captured at startup)
        self._cuda_graph_kv_prefill = None   # Graph P: step 0 (fills KV cache)
        self._cuda_graph_kv_decode = None    # Graph D: steps 1-49 (reads KV cache)
        self._static_kv_x = None
        self._static_kv_x_action = None
        self._static_kv_x_state = None
        self._static_kv_timesteps = None
        self._static_kv_context = None       # context input (Graph P only)
        self._static_kv_context_action = None
        self._static_kv_fs = None
        self._static_kv_prefill_output = None
        self._static_kv_decode_output = None

    def enable_kv_cache(self):
        """Enable KV caching for DDIM loop. Call before each DDIM sampling run.

        On the first call: sets up for fresh cache allocation.
        On subsequent calls: marks caches for refill (step 0 will copy_() new
        values into existing static buffers, preserving addresses for CUDA Graph).
        """
        self._kv_cache_mode = True
        self._kv_step_zero_pending = True  # step 0 needs to run eagerly
        for m in self.modules():
            if hasattr(m, '_kv_cache_enabled'):
                m._kv_cache_enabled = True
                m._kv_cache_fill = True  # refill on next call (step 0)

    def disable_kv_cache(self):
        """Disable KV caching between DDIM runs.

        Preserves the CUDA Graph and static buffers so they can be reused
        across iterations (graph reads from stable addresses via copy_()).
        Call reset_kv_cuda_graph() to fully release the graph.
        """
        self._kv_cache_mode = False
        # Keep _context_cache, _kv_cache, _cuda_graph_kv, _static_kv_*
        # alive — enable_kv_cache() will set _kv_cache_fill to refill them.
        for m in self.modules():
            if hasattr(m, '_kv_cache_enabled'):
                m._kv_cache_enabled = False

    def reset_kv_cuda_graph(self):
        """Fully release KV cache, CUDA Graphs, and all static buffers."""
        self._kv_cache_mode = False
        self._kv_step_zero_pending = False
        self._context_cache = None
        for m in self.modules():
            if hasattr(m, '_kv_cache_enabled'):
                m._kv_cache = None
                m._kv_cache_enabled = False
                m._kv_cache_fill = False
        self._cuda_graph_kv_prefill = None
        self._cuda_graph_kv_decode = None
        self._static_kv_x = None
        self._static_kv_x_action = None
        self._static_kv_x_state = None
        self._static_kv_timesteps = None
        self._static_kv_context = None
        self._static_kv_context_action = None
        self._static_kv_fs = None
        self._static_kv_prefill_output = None
        self._static_kv_decode_output = None

    def _capture_cuda_graph_kv_pair(self, x, x_action, x_state, timesteps,
                                    context, context_action, fs):
        """Capture prefill + decode CUDA Graph pair at daemon startup.

        Graph P (prefill): captures step-0 path — processes context, fills
        KV caches via copy_() into pre-allocated buffers.

        Graph D (decode): captures steps 1-49 path — reads context from
        _context_cache and K/V from per-layer caches (no projections).

        Both graphs share static input buffers for x, x_action, x_state,
        timesteps, context_action, fs.  Graph P additionally uses a static
        context buffer.  All tensor addresses are stable across replays.
        """
        # ── Shared static input buffers ───────────────────────────────────
        self._static_kv_x = x.clone()
        self._static_kv_x_action = x_action.clone()
        self._static_kv_x_state = x_state.clone()
        self._static_kv_timesteps = timesteps.clone()
        self._static_kv_context = context.clone()  # Graph P only
        self._static_kv_fs = fs.clone() if torch.is_tensor(fs) else fs
        self._static_kv_context_action = []
        for item in context_action:
            self._static_kv_context_action.append(
                item.clone() if torch.is_tensor(item) else item)

        def _set_kv_fill(fill: bool):
            for m in self.modules():
                if hasattr(m, '_kv_cache_fill'):
                    m._kv_cache_fill = fill

        # ── Graph P (prefill): processes context + fills KV caches ────────
        self._kv_cache_mode = True
        self._kv_step_zero_pending = True
        for m in self.modules():
            if hasattr(m, '_kv_cache_enabled'):
                m._kv_cache_enabled = True

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._CUDA_GRAPH_WARMUP_STEPS):
                _set_kv_fill(True)
                self._kv_step_zero_pending = True
                self._forward_impl(
                    self._static_kv_x, self._static_kv_x_action,
                    self._static_kv_x_state, self._static_kv_timesteps,
                    self._static_kv_context, self._static_kv_context_action,
                    None, self._static_kv_fs)
        torch.cuda.current_stream().wait_stream(s)

        # Re-set flags for actual capture
        _set_kv_fill(True)
        self._kv_step_zero_pending = True

        self._cuda_graph_kv_prefill = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph_kv_prefill):
            self._static_kv_prefill_output = self._forward_impl(
                self._static_kv_x, self._static_kv_x_action,
                self._static_kv_x_state, self._static_kv_timesteps,
                self._static_kv_context, self._static_kv_context_action,
                None, self._static_kv_fs)
        print("[WMA-HPC] Graph P (prefill) captured")

        # After capture: _context_cache and all _kv_cache are populated
        self._kv_step_zero_pending = False

        # ── Graph D (decode): reads from KV caches (no projections) ───────
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._CUDA_GRAPH_WARMUP_STEPS):
                self._forward_impl(
                    self._static_kv_x, self._static_kv_x_action,
                    self._static_kv_x_state, self._static_kv_timesteps,
                    None, self._static_kv_context_action,
                    None, self._static_kv_fs)
        torch.cuda.current_stream().wait_stream(s)

        self._cuda_graph_kv_decode = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph_kv_decode):
            self._static_kv_decode_output = self._forward_impl(
                self._static_kv_x, self._static_kv_x_action,
                self._static_kv_x_state, self._static_kv_timesteps,
                None, self._static_kv_context_action,
                None, self._static_kv_fs)
        print("[WMA-HPC] Graph D (decode, FP32) captured")

        # ── Graph D-BF16 (decode with BF16 autocast): full BF16 compute ──
        # Uses torch.autocast to run ALL operations (conv, attention, etc.)
        # in BF16 for maximum speed. Triton kernels (GroupNorm, GEGLU)
        # internally promote to FP32 for numerically sensitive ops.
        # BF16 attention via SM90-native SDPA cuDNN flash kernels.
        import unifolm_wma.modules.attention as _attn_mod
        _attn_mod._WMA_BF16_DECODE_ACTIVE = True

        # BF16 graph needs its own static input buffers (different memory addresses)
        self._static_kv_bf16_x = self._static_kv_x.clone()
        self._static_kv_bf16_x_action = self._static_kv_x_action.clone()
        self._static_kv_bf16_x_state = self._static_kv_x_state.clone()
        self._static_kv_bf16_timesteps = self._static_kv_timesteps.clone()
        self._static_kv_bf16_fs = self._static_kv_fs.clone() if torch.is_tensor(self._static_kv_fs) else self._static_kv_fs
        self._static_kv_bf16_context_action = []
        for item in self._static_kv_context_action:
            self._static_kv_bf16_context_action.append(
                item.clone() if torch.is_tensor(item) else item)

        _use_autocast_bf16 = os.getenv('WMA_AUTOCAST_BF16', '0') == '1'
        _autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if _use_autocast_bf16 else torch.autocast(device_type='cuda', enabled=False)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._CUDA_GRAPH_WARMUP_STEPS):
                with _autocast_ctx:
                    self._forward_impl(
                        self._static_kv_bf16_x, self._static_kv_bf16_x_action,
                        self._static_kv_bf16_x_state, self._static_kv_bf16_timesteps,
                        None, self._static_kv_bf16_context_action,
                        None, self._static_kv_bf16_fs)
        torch.cuda.current_stream().wait_stream(s)

        self._cuda_graph_kv_decode_bf16 = torch.cuda.CUDAGraph()
        with _autocast_ctx:
            with torch.cuda.graph(self._cuda_graph_kv_decode_bf16):
                self._static_kv_decode_bf16_output = self._forward_impl(
                    self._static_kv_bf16_x, self._static_kv_bf16_x_action,
                    self._static_kv_bf16_x_state, self._static_kv_bf16_timesteps,
                    None, self._static_kv_bf16_context_action,
                    None, self._static_kv_bf16_fs)
        _label = "BF16 autocast (full)" if _use_autocast_bf16 else "BF16 attention"
        print(f"[WMA-HPC] Graph D-BF16 (decode, {_label}) captured")

        _attn_mod._WMA_BF16_DECODE_ACTIVE = False

        # Number of early DDIM steps to use BF16 decode (0 = all FP32, 49 = all BF16)
        # Negative value: "FP32-first" mode — first N steps FP32, rest BF16
        #   e.g. -25: steps 1-25 FP32, steps 26-49 BF16
        _bf16_raw = int(os.getenv('WMA_BF16_STEPS', '0'))
        if _bf16_raw < 0:
            # FP32-first mode: first |N| steps FP32, rest BF16
            self._bf16_decode_steps = 0  # not used in reverse mode
            self._fp32_first_steps = abs(_bf16_raw)
            print(f"[WMA-HPC] Per-step mixed precision (FP32-first): steps 1-{self._fp32_first_steps} FP32, "
                  f"steps {self._fp32_first_steps+1}-49 BF16")
        else:
            self._bf16_decode_steps = _bf16_raw
            self._fp32_first_steps = 0
            if self._bf16_decode_steps > 0:
                print(f"[WMA-HPC] Per-step mixed precision: steps 1-{self._bf16_decode_steps} BF16, "
                      f"steps {self._bf16_decode_steps+1}-49 FP32")

    def _capture_cuda_graph(self, x, x_action, x_state, timesteps,
                            context, context_action, fs):
        """Warmup on side stream and capture the CUDA graph."""
        # Create static input buffers
        self._static_x = x.clone()
        self._static_x_action = x_action.clone()
        self._static_x_state = x_state.clone()
        self._static_timesteps = timesteps.clone()
        self._static_context = context.clone()
        # context_action is a mixed list: [Tensor, Tensor, bool, Tensor, Tensor]
        # Only first 2 (tensors) are used in forward; clone all tensors.
        self._static_context_action = []
        for item in context_action:
            self._static_context_action.append(
                item.clone() if torch.is_tensor(item) else item)
        self._static_fs = fs.clone() if torch.is_tensor(fs) else fs

        # Warmup on side stream (PyTorch best practice)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._CUDA_GRAPH_WARMUP_STEPS):
                self._forward_impl(
                    self._static_x, self._static_x_action,
                    self._static_x_state, self._static_timesteps,
                    self._static_context, self._static_context_action,
                    None, self._static_fs)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._static_output = self._forward_impl(
                self._static_x, self._static_x_action,
                self._static_x_state, self._static_timesteps,
                self._static_context, self._static_context_action,
                None, self._static_fs)
        print("[WMA-HPC] WMAModel CUDA Graph captured successfully")

    def forward(self,
                x: Tensor,
                x_action: Tensor,
                x_state: Tensor,
                timesteps: Tensor,
                context: Tensor | None = None,
                context_action: Tensor | None = None,
                features_adapter: Any = None,
                fs: Tensor | None = None,
                **kwargs) -> Tensor | tuple[Tensor, ...]:
        # [WMA-HPC] Dual CUDA Graph: Graph P (prefill) + Graph D (decode)
        # Graph P (step 0): processes context + fills KV caches via graph replay
        # Graph D (steps 1-49): reads from KV caches via graph replay
        # Both graphs pre-captured at daemon startup — zero eager execution.
        if self._kv_cache_mode:
            if self._cuda_graph_kv_prefill is not None:
                # ── Pre-captured dual-graph path ──────────────────────
                if self._kv_step_zero_pending:
                    # Step 0: replay Graph P (fills KV cache with new context)
                    self._static_kv_x.copy_(x)
                    self._static_kv_x_action.copy_(x_action)
                    self._static_kv_x_state.copy_(x_state)
                    self._static_kv_timesteps.copy_(timesteps)
                    self._static_kv_context.copy_(context)
                    for i, item in enumerate(context_action):
                        if torch.is_tensor(item):
                            self._static_kv_context_action[i].copy_(item)
                    if torch.is_tensor(fs) and self._static_kv_fs is not None:
                        self._static_kv_fs.copy_(fs)

                    self._cuda_graph_kv_prefill.replay()
                    self._kv_step_zero_pending = False

                    # Sync context_action to BF16 decode buffers (different
                    # memory addresses captured by the BF16 graph)
                    if self._bf16_decode_steps > 0 and hasattr(self, '_static_kv_bf16_context_action'):
                        for j, item in enumerate(self._static_kv_context_action):
                            if torch.is_tensor(item):
                                self._static_kv_bf16_context_action[j].copy_(item)
                        if torch.is_tensor(fs) and self._static_kv_bf16_fs is not None:
                            self._static_kv_bf16_fs.copy_(fs)

                    y, a_y, s_y = self._static_kv_prefill_output
                    return y.clone(), a_y.clone(), s_y.clone()

                # Steps 1-49: replay Graph D (FP32) or Graph D-BF16
                # _force_bf16_decode: int, extra BF16 step threshold for dm
                #   0 = use normal _bf16_decode_steps
                #   >0 = use this value as threshold (overrides _bf16_decode_steps)
                _dm_bf16 = getattr(self, '_force_bf16_decode', 0)
                _fp32_first = getattr(self, '_fp32_first_steps', 0)
                if _dm_bf16 > 0:
                    # DM override: BF16 for steps <= threshold
                    _use_bf16_graph = (
                        hasattr(self, '_cuda_graph_kv_decode_bf16')
                        and self._current_ddim_step <= _dm_bf16
                    )
                elif _fp32_first > 0:
                    # FP32-first mode: FP32 for early steps, BF16 for late steps
                    _use_bf16_graph = (
                        hasattr(self, '_cuda_graph_kv_decode_bf16')
                        and self._current_ddim_step > _fp32_first
                    )
                else:
                    # Normal mode: BF16 for early steps, FP32 for late steps
                    _bf16_threshold = self._bf16_decode_steps
                    _use_bf16_graph = (
                        _bf16_threshold > 0
                        and hasattr(self, '_cuda_graph_kv_decode_bf16')
                        and self._current_ddim_step <= _bf16_threshold
                    )
                if _use_bf16_graph:
                    # BF16 decode graph (early high-noise steps)
                    self._static_kv_bf16_x.copy_(x)
                    self._static_kv_bf16_x_action.copy_(x_action)
                    self._static_kv_bf16_x_state.copy_(x_state)
                    self._static_kv_bf16_timesteps.copy_(timesteps)
                    if torch.is_tensor(fs) and self._static_kv_bf16_fs is not None:
                        self._static_kv_bf16_fs.copy_(fs)

                    self._cuda_graph_kv_decode_bf16.replay()
                    y, a_y, s_y = self._static_kv_decode_bf16_output
                else:
                    # FP32 decode graph (late refinement steps)
                    self._static_kv_x.copy_(x)
                    self._static_kv_x_action.copy_(x_action)
                    self._static_kv_x_state.copy_(x_state)
                    self._static_kv_timesteps.copy_(timesteps)
                    if torch.is_tensor(fs) and self._static_kv_fs is not None:
                        self._static_kv_fs.copy_(fs)

                    self._cuda_graph_kv_decode.replay()
                    y, a_y, s_y = self._static_kv_decode_output

                return y.clone(), a_y.clone(), s_y.clone()

            else:
                # ── Eager KV cache path (WMA_KV_CACHE=1, no graph) ────
                result = self._forward_impl(x, x_action, x_state, timesteps,
                                            context, context_action,
                                            features_adapter, fs, **kwargs)
                if self._kv_step_zero_pending:
                    self._kv_step_zero_pending = False
                return result

        # Original single-graph path (non-KV-cache mode)
        if os.getenv("WMA_CUDA_GRAPH_WMA_MODEL") != "1":
            return self._forward_impl(x, x_action, x_state, timesteps,
                                      context, context_action,
                                      features_adapter, fs, **kwargs)

        if self._cuda_graph is None:
            # Warmup phase: run eagerly to stabilize allocations
            if self._cuda_graph_warmup_count < self._CUDA_GRAPH_WARMUP_STEPS:
                self._cuda_graph_warmup_count += 1
                return self._forward_impl(x, x_action, x_state, timesteps,
                                          context, context_action,
                                          features_adapter, fs)
            # Capture
            self._capture_cuda_graph(x, x_action, x_state, timesteps,
                                     context, context_action, fs)

        # Copy dynamic inputs to static buffers
        self._static_x.copy_(x)
        self._static_x_action.copy_(x_action)
        self._static_x_state.copy_(x_state)
        self._static_timesteps.copy_(timesteps)
        self._static_context.copy_(context)
        for i, item in enumerate(context_action):
            if torch.is_tensor(item):
                self._static_context_action[i].copy_(item)
        if torch.is_tensor(fs) and self._static_fs is not None:
            self._static_fs.copy_(fs)

        # Replay
        self._cuda_graph.replay()

        # Must clone: CFG calls apply_model twice and needs both results alive
        y, a_y, s_y = self._static_output
        return y.clone(), a_y.clone(), s_y.clone()

    def _forward_impl(self,
                      x: Tensor,
                      x_action: Tensor,
                      x_state: Tensor,
                      timesteps: Tensor,
                      context: Tensor | None = None,
                      context_action: Tensor | None = None,
                      features_adapter: Any = None,
                      fs: Tensor | None = None,
                      **kwargs) -> Tensor | tuple[Tensor, ...]:

        """
        Forward pass of the World-Model-Action backbone.

        Args:
            x: Input tensor (latent video), shape (B, C,...).
            x_action: action stream input.
            x_state: state stream input.
            timesteps: Diffusion timesteps, shape (B,) or scalar Tensor.
            context: conditioning context for cross-attention.
            context_action: conditioning context specific to action/state (implementation-specific).
            features_adapter: module or dict to adapt intermediate features.
            fs: frame-stride / fps conditioning.

        Returns:
            Tuple of Tensors for predictions:

        """

        b, _, t, _, _ = x.shape
        compute_dtype = self.dtype  # bf16 or fp32
        t_emb = timestep_embedding(timesteps,
                                   self.model_channels,
                                   repeat_only=False).type(compute_dtype)
        emb = self.time_embed(t_emb)

        # [WMA-HPC] Context processing is timestep-invariant; cache on first
        # DDIM step and reuse for steps 1-49 when KV cache is active.
        # Uses copy_() for refill to preserve tensor addresses for CUDA Graph.
        if self._kv_cache_mode and self._context_cache is not None and not self._kv_step_zero_pending:
            context = self._context_cache
        else:
            context = context.type(compute_dtype)
            bt, l_context, _ = context.shape
            if self.base_model_gen_only:
                assert l_context == 77 + self.n_obs_steps * 16, ">>> ERROR Context dim 1 ..."  ## NOTE HANDCODE
            else:
                if l_context == self.n_obs_steps + 77 + t * 16:
                    context_agent_state = context[:, :self.n_obs_steps]
                    context_text = context[:, self.n_obs_steps:self.n_obs_steps +
                                           77, :]
                    context_img = context[:, self.n_obs_steps + 77:, :]
                    context_agent_state = context_agent_state.repeat_interleave(
                        repeats=t, dim=0)
                    context_text = context_text.repeat_interleave(repeats=t, dim=0)
                    context_img = rearrange(context_img,
                                            'b (t l) c -> (b t) l c',
                                            t=t)
                    context = torch.cat(
                        [context_agent_state, context_text, context_img], dim=1)
                elif l_context == self.n_obs_steps + 16 + 77 + t * 16:
                    context_agent_state = context[:, :self.n_obs_steps]
                    context_agent_action = context[:, self.
                                                   n_obs_steps:self.n_obs_steps +
                                                   16, :]
                    context_agent_action = rearrange(
                        context_agent_action.unsqueeze(2), 'b t l d -> (b t) l d')
                    context_agent_action = self.action_token_projector(
                        context_agent_action)
                    context_agent_action = rearrange(context_agent_action,
                                                     '(b o) l d -> b o l d',
                                                     o=t)
                    context_agent_action = rearrange(context_agent_action,
                                                     'b o (t l) d -> b o t l d',
                                                     t=t)
                    context_agent_action = context_agent_action.permute(
                        0, 2, 1, 3, 4)
                    context_agent_action = rearrange(context_agent_action,
                                                     'b t o l d -> (b t) (o l) d')

                    context_text = context[:, self.n_obs_steps +
                                           16:self.n_obs_steps + 16 + 77, :]
                    context_text = context_text.repeat_interleave(repeats=t, dim=0)

                    context_img = context[:, self.n_obs_steps + 16 + 77:, :]
                    context_img = rearrange(context_img,
                                            'b (t l) c -> (b t) l c',
                                            t=t)
                    context_agent_state = context_agent_state.repeat_interleave(
                        repeats=t, dim=0)
                    context = torch.cat([
                        context_agent_state, context_agent_action, context_text,
                        context_img
                    ],
                                        dim=1)
            if self._kv_cache_mode:
                if self._context_cache is None:
                    self._context_cache = context.clone()
                else:
                    self._context_cache.copy_(context)
                context = self._context_cache

        emb = emb.repeat_interleave(repeats=t, dim=0)

        x = rearrange(x, 'b c t h w -> (b t) c h w')

        # Combine emb
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor([self.default_fs] * b,
                                  dtype=torch.long,
                                  device=x.device)
            fs_emb = timestep_embedding(fs,
                                        self.model_channels,
                                        repeat_only=False).type(compute_dtype)

            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        h = x.type(self.dtype)
        if h.ndim == 4:
            h = h.contiguous(memory_format=torch.channels_last)
        adapter_idx = 0
        # hs: hidden states, for skip connection
        hs = []
        # hs_a: hidden states for action
        hs_a = []
        for id, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context, batch_size=b)
            if id == 0 and self.addition_attention:
                # 此时空间分辨率 (H, W) 还是满的（最大分辨率），完全没有被下采样。
                # init_attn 立即对这T帧的高分辨率特征进行了一次时序混合（Temporal Mixing）。
                h = self.init_attn(h, emb, context=context, batch_size=b)
            # plug-in adapter features
            if ((id + 1) % 3 == 0) and features_adapter is not None:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            if id != 0:
                if isinstance(module[0], Downsample):
                    hs_a.append(
                        rearrange(hs[-1], '(b t) c h w -> b t c h w', t=t))
            hs.append(h)
        hs_a.append(rearrange(h, '(b t) c h w -> b t c h w', t=t))

        if features_adapter is not None:
            assert len(
                features_adapter) == adapter_idx, 'Wrong features_adapter'
        h = self.middle_block(h, emb, context=context, batch_size=b)
        hs_a.append(rearrange(h, '(b t) c h w -> b t c h w', t=t))

        hs_out = []
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context=context, batch_size=b)
            if isinstance(module[-1], Upsample):
                hs_a.append(
                    rearrange(hs_out[-1], '(b t) c h w -> b t c h w', t=t))
            hs_out.append(h)
        hs_a.append(rearrange(hs_out[-1], '(b t) c h w -> b t c h w', t=t))

        y = self.out(h)
        y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
        y = y.type(x.dtype)  # cast back to input dtype (e.g. fp32)

        if not self.base_model_gen_only and not getattr(self, '_skip_action_heads', False):
            ba, _, _ = x_action.shape
            ts_a = timesteps[:ba]
            ts_s = timesteps[:ba] if b > 1 else timesteps
            ctx_head = context_action[:2]
            # Cast backbone hidden states to fp32 for action/state heads
            hs_a_f = [h.float() for h in hs_a]

            # [WMA-HPC] Disable autocast for action/state heads — they need
            # FP32 precision.  When the BF16 CUDA graph is captured with
            # autocast, this ensures only the backbone runs in BF16.
            with torch.autocast(device_type='cuda', enabled=False):
                if self._export_mode:
                    # Sequential execution for torch.export (CUDA streams not traceable)
                    a_y = self.action_unet(x_action, ts_a, hs_a_f,
                                           ctx_head, **kwargs)
                    s_y = self.state_unet(x_state, ts_s, hs_a_f,
                                          ctx_head, **kwargs)
                else:
                    # Fork: run action_unet and state_unet on parallel CUDA streams
                    main_stream = torch.cuda.current_stream()
                    self._stream_action.wait_stream(main_stream)
                    self._stream_state.wait_stream(main_stream)

                    with torch.cuda.stream(self._stream_action):
                        a_y = self.action_unet(x_action, ts_a, hs_a_f,
                                               ctx_head, **kwargs)
                    with torch.cuda.stream(self._stream_state):
                        s_y = self.state_unet(x_state, ts_s, hs_a_f,
                                              ctx_head, **kwargs)

                    # Join: wait for both streams before continuing
                    main_stream.wait_stream(self._stream_action)
                    main_stream.wait_stream(self._stream_state)
        else:
            a_y = torch.zeros_like(x_action)
            s_y = torch.zeros_like(x_state)

        return y, a_y.type(x.dtype), s_y.type(x.dtype)
