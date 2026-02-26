import torch
import triton
from unifolm_wma.ops.kernels.spatial_softmax import (
    SpatialSoftmax_NHWC_collect,
    SpatialSoftmax_NHWC_collect_onlinesoftmax,
    SpatialSoftmax_NHWC_normalize_expectation,
    get_rblock,
)


class SpatialSoftmaxTriton(torch.autograd.Function):
    """接受 NCHW 输入，内部转为 NHWC 后调用 triton kernel。"""
    @staticmethod
    def forward(ctx, feature, bias, temperature):
        assert feature.is_contiguous() or feature.stride(-1) != 1, "Input must be NCHW (or convertible)"

        N, C, H, W = feature.shape
        HxW = H * W

        feature_nhwc = feature.permute(0, 2, 3, 1).contiguous()

        XBLOCK = 128
        RBLOCK = get_rblock({'HxW': HxW})

        out_max = torch.empty((N * C), dtype=torch.float32, device=feature.device)
        out_sum = torch.empty((N * C), dtype=torch.float32, device=feature.device)
        out_exp_x = torch.zeros((N * C), dtype=torch.float32, device=feature.device)
        out_exp_y = torch.zeros((N * C), dtype=torch.float32, device=feature.device)

        # 2D grid: (channel blocks, batch) — 避免 XBLOCK > C 时跨 batch
        grid = (triton.cdiv(C, XBLOCK), N)

        SpatialSoftmax_NHWC_collect[grid](
            feature_nhwc, bias, temperature,
            out_max, out_sum,
            N, HxW, C,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK,
        )

        SpatialSoftmax_NHWC_normalize_expectation[grid](
            feature_nhwc, bias, temperature,
            out_max, out_sum,
            out_exp_x, out_exp_y,
            N, H, W, C,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK,
        )

        res_x = out_exp_x.view(N, C).unsqueeze(-1)
        res_y = out_exp_y.view(N, C).unsqueeze(-1)

        return torch.cat([res_x, res_y], dim=-1)


class SpatialSoftmaxTritonNHWC(torch.autograd.Function):
    """接受已经是 NHWC 布局的输入，零拷贝直接调用 triton kernel。"""
    @staticmethod
    def forward(ctx, feature_nhwc, bias, temperature):
        # feature_nhwc: [N, H, W, C] — 已经是 NHWC 连续内存
        N, H, W, C = feature_nhwc.shape
        HxW = H * W

        XBLOCK = 128
        RBLOCK = get_rblock({'HxW': HxW})

        out_max = torch.empty((N * C), dtype=torch.float32, device=feature_nhwc.device)
        out_sum = torch.empty((N * C), dtype=torch.float32, device=feature_nhwc.device)
        out_exp_x = torch.zeros((N * C), dtype=torch.float32, device=feature_nhwc.device)
        out_exp_y = torch.zeros((N * C), dtype=torch.float32, device=feature_nhwc.device)

        # 2D grid: (channel blocks, batch) — 避免 XBLOCK > C 时跨 batch
        grid = (triton.cdiv(C, XBLOCK), N)

        SpatialSoftmax_NHWC_collect[grid](
            feature_nhwc, bias, temperature,
            out_max, out_sum,
            N, HxW, C,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK,
        )

        SpatialSoftmax_NHWC_normalize_expectation[grid](
            feature_nhwc, bias, temperature,
            out_max, out_sum,
            out_exp_x, out_exp_y,
            N, H, W, C,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK,
        )

        res_x = out_exp_x.view(N, C).unsqueeze(-1)
        res_y = out_exp_y.view(N, C).unsqueeze(-1)

        return torch.cat([res_x, res_y], dim=-1)


class SpatialSoftmaxTritonOnline(torch.autograd.Function):
    """Online Softmax 变体：单 pass 同时计算 max 和 sum，减少一次全局内存读取。"""
    @staticmethod
    def forward(ctx, feature, bias, temperature):
        N, C, H, W = feature.shape
        HxW = H * W

        feature_nhwc = feature.permute(0, 2, 3, 1).contiguous()

        XBLOCK = 8
        RBLOCK = 16

        out_max = torch.empty((N * C), dtype=torch.float32, device=feature.device)
        out_sum = torch.empty((N * C), dtype=torch.float32, device=feature.device)
        out_exp_x = torch.zeros((N * C), dtype=torch.float32, device=feature.device)
        out_exp_y = torch.zeros((N * C), dtype=torch.float32, device=feature.device)

        grid = (triton.cdiv(C, XBLOCK), N)

        SpatialSoftmax_NHWC_collect_onlinesoftmax[grid](
            feature_nhwc, bias, temperature,
            out_max, out_sum,
            N, HxW, C,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK,
        )

        SpatialSoftmax_NHWC_normalize_expectation[grid](
            feature_nhwc, bias, temperature,
            out_max, out_sum,
            out_exp_x, out_exp_y,
            N, H, W, C,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK,
        )

        res_x = out_exp_x.view(N, C).unsqueeze(-1)
        res_y = out_exp_y.view(N, C).unsqueeze(-1)

        return torch.cat([res_x, res_y], dim=-1)


class SpatialSoftmaxTritonOnlineNHWC(torch.autograd.Function):
    """接受已经是 NHWC 布局的输入，零拷贝直接调用 triton kernel。"""
    @staticmethod
    def forward(ctx, feature_nhwc, bias, temperature):
        # feature_nhwc: [N, H, W, C] — 已经是 NHWC 连续内存
        N, H, W, C = feature_nhwc.shape
        HxW = H * W

        XBLOCK = 8
        RBLOCK = 16

        out_max = torch.empty((N * C), dtype=torch.float32, device=feature_nhwc.device)
        out_sum = torch.empty((N * C), dtype=torch.float32, device=feature_nhwc.device)
        out_exp_x = torch.zeros((N * C), dtype=torch.float32, device=feature_nhwc.device)
        out_exp_y = torch.zeros((N * C), dtype=torch.float32, device=feature_nhwc.device)

        # 2D grid: (channel blocks, batch) — 避免 XBLOCK > C 时跨 batch
        grid = (triton.cdiv(C, XBLOCK), N)

        SpatialSoftmax_NHWC_collect_onlinesoftmax[grid](
            feature_nhwc, bias, temperature,
            out_max, out_sum,
            N, HxW, C,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK,
            num_stages=4
        )

        SpatialSoftmax_NHWC_normalize_expectation[grid](
            feature_nhwc, bias, temperature,
            out_max, out_sum,
            out_exp_x, out_exp_y,
            N, H, W, C,
            XBLOCK=XBLOCK, RBLOCK=RBLOCK,
        )

        res_x = out_exp_x.view(N, C).unsqueeze(-1)
        res_y = out_exp_y.view(N, C).unsqueeze(-1)

        return torch.cat([res_x, res_y], dim=-1)



def triton_spatial_softmax(feature, bias, temperature):
    """NCHW 入口 — Naive 2-pass（内部会 permute）。"""
    return SpatialSoftmaxTriton.apply(feature, bias, temperature)


def triton_spatial_softmax_online(feature, bias, temperature):
    """NCHW 入口 — Online Softmax 1-pass collect（内部会 permute）。"""
    return SpatialSoftmaxTritonOnline.apply(feature, bias, temperature)

def triton_spatial_softmax_online_nhwc(feature, bias, temperature):
    """NCHW 入口 — Online Softmax 1-pass collect（NHWC 入口（零拷贝，数据已就绪））。"""
    return SpatialSoftmaxTritonOnlineNHWC.apply(feature, bias, temperature)


def triton_spatial_softmax_nhwc(feature_nhwc, bias, temperature):
    """NHWC 入口（零拷贝，数据已就绪）。"""
    return SpatialSoftmaxTritonNHWC.apply(feature_nhwc, bias, temperature)
