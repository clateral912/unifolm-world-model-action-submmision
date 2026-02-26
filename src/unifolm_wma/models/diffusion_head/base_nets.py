"""
Contains torch Modules that correspond to basic network building blocks, like
MLP, RNN, and CNN backbones.
"""

import abc

import numpy as np
import torch
import torch.nn.functional as F
from unifolm_wma.ops.wrappers.wrapper import triton_spatial_softmax_online_nhwc


class Module(torch.nn.Module):
    """
    Base class for networks. The only difference from torch.nn.Module is that it
    requires implementing @output_shape.
    """

    @abc.abstractmethod
    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError


"""
================================================
Visual Backbone Networks
================================================
"""


class ConvBase(Module):
    """
    Base class for ConvNets.
    """

    def __init__(self):
        super(ConvBase, self).__init__()

    # dirty hack - re-implement to pass the buck onto subclasses from ABC parent
    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError(
                "Size mismatch: expect size %s, but got size %s"
                % (
                    str(self.output_shape(list(inputs.shape)[1:])),
                    str(list(x.shape)[1:]),
                )
            )
        return x

def _functional_spatial_softmax_impl(
    feature,
    # 权重参数
    conv_w, conv_b,
    # 核心参数
    temperature,
    pos_x, pos_y,
    # 配置参数
    num_kp: int,
    noise_std: float,
    output_variance: bool,
    training: bool
):
    """
    Stateless implementation of Spatial Softmax.
    Corrected dimension reduction logic.
    """
    # 1. Optional 1x1 Convolution
    if conv_w is not None:
        feature = F.conv2d(feature, conv_w, conv_b)

    # 2. Flatten Spatial Dimensions
    # [B, C, H, W] -> [B, C, H*W]
    # 这样 dim=0 是 Batch, dim=1 是 Keypoints, dim=2 是 Spatial
    feature = feature.flatten(2)

    # 3. Softmax over Spatial Dimension (dim=-1)
    # 确保是对每一个 Keypoint 的空间分布做归一化
    attention = F.softmax(feature / temperature, dim=-1)

    # 4. Expectation (Spatial Mean)
    # pos_x: [1, H*W] -> Broadcasts to [B, C, H*W]
    # Sum over dim=-1 (Spatial): [B, C, H*W] -> [B, C, 1]
    expected_x = torch.sum(pos_x * attention, dim=-1, keepdim=True)
    expected_y = torch.sum(pos_y * attention, dim=-1, keepdim=True)

    # [B, C, 2]
    expected_xy = torch.cat([expected_x, expected_y], dim=-1)

    # 此时 shape 已经是 [B, C, 2] (即 [B, num_kp, 2])
    # 为了保险起见，可以 reshape 一下，但通常不需要 view(-1) 了
    feature_keypoints = expected_xy.view(-1, num_kp, 2)

    # 5. Training Noise
    if training and noise_std > 0.0:
        noise = torch.randn_like(feature_keypoints) * noise_std
        feature_keypoints = feature_keypoints + noise

    # 6. Variance Calculation (Optional)
    if output_variance:
        # Sum over dim=-1
        expected_xx = torch.sum(pos_x * pos_x * attention, dim=-1, keepdim=True)
        expected_yy = torch.sum(pos_y * pos_y * attention, dim=-1, keepdim=True)
        expected_xy_cov = torch.sum(pos_x * pos_y * attention, dim=-1, keepdim=True)

        var_x = expected_xx - expected_x * expected_x
        var_y = expected_yy - expected_y * expected_y
        var_xy = expected_xy_cov - expected_x * expected_y

        # [B, C, 4] -> [B, C, 2, 2] -> Reshape to align with batch
        feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], dim=-1).view(-1, num_kp, 2, 2)
        return feature_keypoints, feature_covar

    return feature_keypoints


class SpatialSoftmax(torch.nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
        self,
        input_shape,
        num_kp=32,
        temperature=1.0,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            # 用 Linear 替代 Conv2d 1x1：对 NHWC 布局天然友好，
            # 输入 [N, H, W, in_c] -> 输出 [N, H, W, num_kp]，全程 NHWC 无需额外 permute
            self.proj = torch.nn.Linear(self._in_c, num_kp)
            self._num_kp = num_kp
        else:
            self.proj = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        # triton kernel 条件：不需要 output_variance
        # Linear 输出已是 NHWC 连续内存，可直接送入 triton kernel
        self._use_triton = not self.output_variance

        # triton kernel 需要 per-channel bias（Linear 已包含 bias，此处传零值）
        # persistent=False: 不进入 state_dict，避免旧 checkpoint 加载报错
        self.register_buffer(
            "_triton_bias",
            torch.zeros(self._num_kp, dtype=torch.float32),
            persistent=False
        )

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=True
            )
            self.register_parameter("temperature", temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=False
            )
            self.register_buffer("temperature", temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        self.kps = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        """
        兼容旧 checkpoint：自动将 Conv2d 的 nets.weight/bias 转换为 Linear 的 proj.weight/bias。
        Conv2d 1x1 weight: [out_c, in_c, 1, 1] -> Linear weight: [out_c, in_c]
        bias shape 不变: [out_c]
        """
        old_w_key = prefix + "nets.weight"
        old_b_key = prefix + "nets.bias"
        new_w_key = prefix + "proj.weight"
        new_b_key = prefix + "proj.bias"

        if old_w_key in state_dict:
            w = state_dict.pop(old_w_key)
            # Conv2d [out_c, in_c, 1, 1] -> Linear [out_c, in_c]
            if w.dim() == 4:
                w = w.squeeze(-1).squeeze(-1)
            state_dict[new_w_key] = w

        if old_b_key in state_dict:
            state_dict[new_b_key] = state_dict.pop(old_b_key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + "(num_kp={}, temperature={}, noise={})".format(
            self._num_kp, self.temperature.item(), self.noise_std
        )

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert len(input_shape) == 3
        assert input_shape[0] == self._in_c
        return [self._num_kp, 2]

    @torch.compiler.disable
    def _update_kps(self, kps):
        # 这个函数有副作用且只用于 debug/vis，不参与计算图
        # disable 标记它不被 trace，但在运行时会触发 graph break
        if isinstance(kps, tuple):
            self.kps = (kps[0].detach(), kps[1].detach())
        else:
            self.kps = kps.detach()

    def forward(self, feature):
        """
        Forward with triton kernel dispatch.
        全程 NHWC 流水线：NCHW -> permute -> Linear(NHWC) -> triton kernel(NHWC)
        仅在 output_variance=True 时回退到原有 PyTorch 实现。
        """
        if self._use_triton and feature.is_cuda:
            # 1. NCHW -> NHWC（唯一一次布局转换）
            feature_nhwc = feature.permute(0, 2, 3, 1).contiguous()

            # 2. Linear 投影（如需要），输出仍为 NHWC 连续内存
            if self.proj is not None:
                feature_nhwc = self.proj(feature_nhwc)  # [N, H, W, num_kp]

            # 3. Triton softmax + expectation，数据已就绪，零拷贝
            return triton_spatial_softmax_online_nhwc(
                feature_nhwc, self._triton_bias, self.temperature
            )

        # 回退路径（output_variance=True 时）
        # 将 Linear 权重重塑为 Conv2d 1x1 格式以复用已有函数
        if self.proj is not None:
            conv_w = self.proj.weight.unsqueeze(-1).unsqueeze(-1)
            conv_b = self.proj.bias
        else:
            conv_w = None
            conv_b = None

        return _functional_spatial_softmax_impl(
            feature,
            conv_w, conv_b,
            self.temperature,
            self.pos_x, self.pos_y,
            self._num_kp,
            self.noise_std,
            self.output_variance,
            self.training,
        )
