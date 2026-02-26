import torch
import torch.nn.functional as F
import os
from torch import nn, einsum
from einops import rearrange, repeat
from functools import partial
from contextlib import nullcontext

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# =========================================================================
# [WMA-HPC] FlashAttention-3 via PyTorch SDPA cuDNN backend (Hopper)
# =========================================================================
# WMA_FA3=1 enables cuDNN FA3 backend on Hopper (SM 9.0) GPUs.
# Falls back to xformers FA2 if unavailable or disabled.
# cuDNN FA3 uses TMA + warp specialization for ~1.5-2x over FA2.
_WMA_FA3_ENABLED = os.getenv("WMA_FA3", "0") == "1"
_SDPA_FA3_AVAILABLE = False

if _WMA_FA3_ENABLED:
    try:
        if (torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 9
                and hasattr(torch.backends.cuda, 'cudnn_sdp_enabled')
                and torch.backends.cuda.cudnn_sdp_enabled()):
            _SDPA_FA3_AVAILABLE = True
            print("[WMA-HPC] FA3 enabled: using PyTorch SDPA cuDNN backend on Hopper")
        else:
            print("[WMA-HPC] FA3 requested but cuDNN SDP not available, falling back to xformers")
    except Exception:
        print("[WMA-HPC] FA3 detection failed, falling back to xformers")

def _sdpa_attention(q, k, v, attn_bias=None):
    """Drop-in replacement for xformers.ops.memory_efficient_attention.

    Accepts xformers layout (B, S, H, D) and converts to SDPA layout (B, H, S, D).
    attn_bias: (B*factor, H, S_q, S_kv) or (B*factor, 1, 1, S_kv) broadcastable tensor, or None.
    """
    # (B, S, H, D) -> (B, H, S, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

    # (B, H, S, D) -> (B, S, H, D)
    return out.transpose(1, 2)


# =========================================================================
# [WMA-HPC] Per-step reduced-precision attention for dual CUDA graph decode
# =========================================================================
# When _WMA_BF16_DECODE_ACTIVE is True, attention uses reduced-precision SDPA
# (SM90-native cuDNN flash attention) instead of FP32 xformers (SM80 cutlass).
# This flag is set during CUDA graph capture and remains constant during
# graph replay --- the captured graph always uses the path that was active at
# capture time.
#
# WMA_ATTENTION_DTYPE: "bf16" (default) or "fp16".
# FP16 has 10 mantissa bits vs BF16's 7, giving better precision for
# attention scores at the same cuDNN flash attention speed.
_WMA_BF16_DECODE_ACTIVE = False
_WMA_REDUCED_ATTN_DTYPE = torch.bfloat16 if os.getenv('WMA_ATTENTION_DTYPE', 'bf16') != 'fp16' else torch.float16
if os.getenv('WMA_ATTENTION_DTYPE') == 'fp16':
    print(f"[WMA-HPC] Reduced-precision attention dtype: float16 (10-bit mantissa)")


def _efficient_attention_bf16(q, k, v, attn_bias=None):
    """Reduced-precision attention via SDPA cuDNN flash attention (SM90-native).

    Casts FP32 Q/K/V to reduced precision (BF16 or FP16), runs SDPA,
    casts output back to FP32. Accepts xformers layout (B, S, H, D).
    ~8x faster than FP32 xformers cutlass on H200.
    """
    _dtype = _WMA_REDUCED_ATTN_DTYPE
    q_bf = q.to(_dtype).transpose(1, 2)  # (B, H, S, D)
    k_bf = k.to(_dtype).transpose(1, 2)
    v_bf = v.to(_dtype).transpose(1, 2)
    if attn_bias is not None:
        attn_bias = attn_bias.to(_dtype)
    out = F.scaled_dot_product_attention(q_bf, k_bf, v_bf, attn_mask=attn_bias)
    return out.transpose(1, 2).to(q.dtype)  # back to (B, S, H, D) FP32


def _efficient_attention(q, k, v, attn_bias=None):
    """Unified attention dispatch: FA3 (SDPA cuDNN) or xformers FA2.

    Accepts 4D xformers layout (B, S, H, D).
    When _WMA_BF16_DECODE_ACTIVE, uses BF16 SDPA for ~8x attention speedup.
    """
    if _WMA_BF16_DECODE_ACTIVE:
        return _efficient_attention_bf16(q, k, v, attn_bias=attn_bias)
    if _SDPA_FA3_AVAILABLE:
        return _sdpa_attention(q, k, v, attn_bias=attn_bias)
    return xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)


def _efficient_attention_3d(q, k, v, b, heads, dim_head, attn_bias=None):
    """Attention dispatch for 3D layout (B*H, S, D) used by fallback path.

    Reshapes to 4D, calls unified dispatch, reshapes back to (B, S, H*D).
    """
    s_q = q.shape[1]
    # (B*H, S, D) -> (B, H, S, D) -> (B, S, H, D)
    q4 = q.view(b, heads, s_q, dim_head).transpose(1, 2)
    k4 = k.view(b, heads, k.shape[1], dim_head).transpose(1, 2)
    v4 = v.view(b, heads, v.shape[1], dim_head).transpose(1, 2)
    out4 = _efficient_attention(q4, k4, v4, attn_bias=attn_bias)
    # (B, S, H, D) -> (B, S, H*D)
    return out4.reshape(b, s_q, heads * dim_head)

from unifolm_wma.utils.common import (
    checkpoint,
    exists,
    default,
)
from unifolm_wma.utils.basics import zero_module


class RelativePosition(nn.Module):
    """https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py"""

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units)
        )
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings

def _global_fused_project_impl(
    context,
    kv_ins, kv_ip, kv_as, kv_aa,
    len_state, len_action, len_text
):
    # Slice logic
    context_agent_state = context[:, : len_state, :]

    start_action = len_state
    end_action = len_state + len_action
    context_agent_action = context[:, start_action : end_action, :]

    start_text = end_action
    end_text = end_action + len_text
    context_ins = context[:, start_text : end_text, :]

    start_image = end_text
    context_image = context[:, start_image :, :]

    # Fused Projections
    k, v = kv_ins(context_ins)
    k_ip, v_ip = kv_ip(context_image)
    k_as, v_as = kv_as(context_agent_state)
    k_aa, v_aa = kv_aa(context_agent_action)

    return k, v, k_ip, v_ip, k_as, v_as, k_aa, v_aa

# 2. Output fusion logic
def _global_fused_combine_impl(
    out, out_ip, out_as, out_aa,
    to_out_layer,
    scale_ip, scale_as, scale_aa,
    alpha_ctx, alpha_cas, alpha_caa,
    use_learnable
):
    if use_learnable:
        combined = (
            out
            + scale_ip * out_ip * (torch.tanh(alpha_ctx) + 1)
            + scale_as * out_as * (torch.tanh(alpha_cas) + 1)
            + scale_aa * out_aa * (torch.tanh(alpha_caa) + 1)
        )
    else:
        combined = (
            out
            + scale_ip * out_ip
            + scale_as * out_as
            + scale_aa * out_aa
        )

    return to_out_layer(combined)

# =========================================================================
# [WMA-HPC] Batched 4-Modality Projection & Attention Support
# =========================================================================

def _project_and_pad_subroutine(module, x, max_len, heads, dim_head):
    """Project -> Reshape to xformers 4D layout (B, N, H, D) -> Pad to max_len.

    Uses KVFusion's fused_linear to produce K/V in one shot, then view as (B, N, 2, H, D)
    and slice to get (B, N, H, D) -- native xformers layout, no permute needed.
    Inductor will fuse view + pad + cat into a single kernel.
    """
    kv = module.fused_linear(x)  # (B, N, 2*H*D)
    b, n, _ = kv.shape

    # View to (B, N, 2, H, D) -- zero-copy reshape
    kv = kv.view(b, n, 2, heads, dim_head)
    k = kv[:, :, 0]  # (B, N, H, D) -- native xformers layout, no permute needed
    v = kv[:, :, 1]  # (B, N, H, D)

    if n < max_len:
        pad_len = max_len - n
        # F.pad starts from the last dim: (D_l, D_r, H_l, H_r, N_l, N_r)
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
    return k, v


def _global_fused_project_impl_batched(
    context,
    kv_ins, kv_ip, kv_as, kv_aa,
    len_state, len_action, len_text, len_image,
    heads, dim_head
):
    """Batched 4-modality projection.

    Slice context -> project each -> pad to max_len -> cat along batch dim.
    Returns k_all, v_all with shape (4*B, max_len, H, D), native xformers layout.
    Inductor will auto-fuse view/pad/cat for Zero-Copy.
    """
    # Context slicing
    context_agent_state = context[:, :len_state, :]
    start_action = len_state
    end_action = len_state + len_action
    context_agent_action = context[:, start_action:end_action, :]
    start_text = end_action
    end_text = end_action + len_text
    context_ins = context[:, start_text:end_text, :]
    context_image = context[:, end_text:, :]

    max_len = max(len_state, len_action, len_text, len_image)

    # Project & Pad each modality -- Inductor fuses view/pad/cat
    k_txt, v_txt = _project_and_pad_subroutine(kv_ins, context_ins, max_len, heads, dim_head)
    k_img, v_img = _project_and_pad_subroutine(kv_ip, context_image, max_len, heads, dim_head)
    k_as, v_as = _project_and_pad_subroutine(kv_as, context_agent_state, max_len, heads, dim_head)
    k_aa, v_aa = _project_and_pad_subroutine(kv_aa, context_agent_action, max_len, heads, dim_head)

    # Cat along batch dim -- Inductor Zero-Copy Concatenation
    # Order: text, image, state, action (matches _make_batched_bias lens_list)
    k_all = torch.cat([k_txt, k_img, k_as, k_aa], dim=0)  # (4*B, max_len, H, D)
    v_all = torch.cat([v_txt, v_img, v_as, v_aa], dim=0)

    return k_all, v_all


def _make_batched_bias(b, max_len, lens_list, device, dtype, attn_mask_aa=None):
    """Construct batched 4-modality attention bias.

    Args:
        b: batch size
        max_len: padded max KV sequence length
        lens_list: [len_text, len_image, len_state, len_action], order matches cat
        device, dtype: tensor placement
        attn_mask_aa: (B, 1, 1, action_len) block-diagonal mask, or None

    Returns:
        bias: (4*B, 1, 1, max_len) -- broadcastable to (4*B, H, Q_len, max_len)
    """
    # Initialize to -inf (mask all)
    bias = torch.full((4 * b, 1, 1, max_len), float("-inf"), device=device, dtype=dtype)

    # Unmask valid region per modality
    current_b = 0
    for i, length in enumerate(lens_list):
        bias[current_b : current_b + b, :, :, :length] = 0.0
        current_b += b

    # Overlay AgentAction block-diagonal mask (4th modality, index=3)
    if attn_mask_aa is not None:
        aa_start = 3 * b
        aa_len = lens_list[3]
        # attn_mask_aa: 0.0 = attend, -inf = block -- overwrites the 0.0 set above
        bias[aa_start : aa_start + b, :, :, :aa_len] = attn_mask_aa

    return bias


# =========================================================================
# [WMA-HPC] Compiled Function Handle Management
# =========================================================================

_COMPILED_BATCHED_PROJECT_FN = None

def _get_compiled_batched_proj_fn():
    """Get compiled handle for batched projection (lazy init, global singleton)."""
    global _COMPILED_BATCHED_PROJECT_FN
    if _COMPILED_BATCHED_PROJECT_FN is None:
        if os.getenv("WMA_COMPILE_CROSS_ATTN") == "1":
            print("[WMA-HPC] Initializing GLOBAL compiled kernel for batched 4-modality projection...")
            _COMPILED_BATCHED_PROJECT_FN = torch.compile(
                _global_fused_project_impl_batched,
                mode="reduce-overhead",
                fullgraph=True
            )
        else:
            _COMPILED_BATCHED_PROJECT_FN = _global_fused_project_impl_batched
    return _COMPILED_BATCHED_PROJECT_FN

# 3. Execute global compilation (only once)
# Controlled by env var to avoid compilation in non-HPC environments
_COMPILED_PROJECT_FN = None
_COMPILED_COMBINE_FN = None

def _get_compiled_fns():
    global _COMPILED_PROJECT_FN, _COMPILED_COMBINE_FN
    if _COMPILED_PROJECT_FN is None:
        if os.getenv("WMA_COMPILE_CROSS_ATTN") == "1":
            print("[WMA-HPC] Initializing GLOBAL compiled kernels for CrossAttention...")
            _COMPILED_PROJECT_FN = torch.compile(_global_fused_project_impl, mode="reduce-overhead", fullgraph=True)
            _COMPILED_COMBINE_FN = torch.compile(_global_fused_combine_impl, mode="reduce-overhead", fullgraph=True)
        else:
            # Fallback to original functions (no compilation)
            _COMPILED_PROJECT_FN = _global_fused_project_impl
            _COMPILED_COMBINE_FN = _global_fused_combine_impl
    return _COMPILED_PROJECT_FN, _COMPILED_COMBINE_FN

class KVFusion(nn.Module):
    """
    [WMA-HPC Optimization]
    Fuses originally separate K and V projection layers into a single Linear layer.
    """
    def __init__(self, in_features, out_features, original_k_name, original_v_name, bias=False):
        super().__init__()
        self.fused_linear = nn.Linear(in_features, out_features * 2, bias=bias)
        self.original_k_name = original_k_name
        self.original_v_name = original_v_name

    def forward(self, x):
        kv = self.fused_linear(x)
        k, v = kv.chunk(2, dim=-1)
        return k, v


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        relative_position=False,
        temporal_length=None,
        video_length=None,
        agent_state_context_len=2,
        agent_action_context_len=16,
        image_cross_attention=False,
        image_cross_attention_scale=1.0,
        agent_state_cross_attention_scale=1.0,
        agent_action_cross_attention_scale=1.0,
        cross_attention_scale_learnable=False,
        text_context_len=77,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.kv_ins = KVFusion(context_dim, inner_dim, 'to_k', 'to_v')
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

        self.relative_position = relative_position
        if self.relative_position:
            assert temporal_length is not None
            self.relative_position_k = RelativePosition(
                num_units=dim_head, max_relative_position=temporal_length
            )
            self.relative_position_v = RelativePosition(
                num_units=dim_head, max_relative_position=temporal_length
            )
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.agent_state_cross_attention_scale = agent_state_cross_attention_scale
        self.agent_action_cross_attention_scale = agent_action_cross_attention_scale
        self.text_context_len = text_context_len
        self.agent_state_context_len = agent_state_context_len
        self.agent_action_context_len = agent_action_context_len
        self.cross_attention_scale_learnable = cross_attention_scale_learnable

        # [WMA-HPC] KV Cache: skip redundant K/V projections across DDIM steps
        self._kv_cache_enabled = False
        self._kv_cache = None  # populated on first step, reused on steps 1-49
        self._kv_cache_fill = False  # True = fill/refill cache on next call (step 0)

        if self.image_cross_attention:
            # self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            # self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            # self.to_k_as = nn.Linear(context_dim, inner_dim, bias=False)
            # self.to_v_as = nn.Linear(context_dim, inner_dim, bias=False)
            # self.to_k_aa = nn.Linear(context_dim, inner_dim, bias=False)
            # self.to_v_aa = nn.Linear(context_dim, inner_dim, bias=False)
            self.kv_ip = KVFusion(context_dim, inner_dim, 'to_k_ip', 'to_v_ip')
            self.kv_as = KVFusion(context_dim, inner_dim, 'to_k_as', 'to_v_as')
            self.kv_aa = KVFusion(context_dim, inner_dim, 'to_k_aa', 'to_v_aa')
            if cross_attention_scale_learnable:
                self.register_parameter("alpha_ctx", nn.Parameter(torch.tensor(0.0)))
                self.register_parameter("alpha_cas", nn.Parameter(torch.tensor(0.0)))
                self.register_parameter("alpha_caa", nn.Parameter(torch.tensor(0.0)))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        [WMA-HPC Critical Fix]
        Intercept state_dict during parent module loading.
        We have the absolutely correct prefix (e.g. 'model...attn1.').
        We can precisely find the old 'to_k' and fuse it into the new 'kv_ins.fused_linear'.
        """
        def patch_fusion(attr_name, k_name, v_name):
            key_k = prefix + k_name + '.weight'
            key_v = prefix + v_name + '.weight'
            key_fused = prefix + attr_name + '.fused_linear.weight'

            if key_k in state_dict and key_v in state_dict:
                w_k = state_dict[key_k]
                w_v = state_dict[key_v]
                with torch.no_grad():
                    state_dict[key_fused] = torch.cat([w_k, w_v], dim=0)
                del state_dict[key_k]
                del state_dict[key_v]

        patch_fusion('kv_ins', 'to_k', 'to_v')
        if self.image_cross_attention:
            patch_fusion('kv_ip', 'to_k_ip', 'to_v_ip')
            patch_fusion('kv_as', 'to_k_as', 'to_v_as')
            patch_fusion('kv_aa', 'to_k_aa', 'to_v_aa')

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    # =========================================================================
    # [WMA-HPC] Pruned forward variants
    # =========================================================================
    # Specialized, dead-branch-pruned versions of forward/efficient_forward.
    # Dispatched at runtime when parameter validation confirms the config
    # matches pruning conditions. Falls back to original if any check fails.
    # See attention_pruned.py for full analysis of each scenario.

    def _pruned_spatial_self_attn_efficient_forward(self, x):
        """Scenario A: SpatialTransformer.attn1 -- Spatial Self-Attention (xformers)"""
        q = self.to_q(x)
        k, v = self.kv_ins(x)

        b, _, _ = q.shape
        h = self.heads
        d = self.dim_head
        n = q.shape[1]

        q = q.view(b, n, h, d)
        k = k.view(b, n, h, d)
        v = v.view(b, n, h, d)

        # 2. Attention dispatch: FA3 (SDPA cuDNN) or xformers FA2
        out = _efficient_attention(q, k, v)

        # 3. Output shape is [B, N, H, D], just view back
        out = out.view(b, n, h * d)

        return self.to_out(out)

    def _pruned_spatial_cross_attn_efficient_forward(self, x, context):
        """Scenario B: SpatialTransformer.attn2 -- Full Multimodal Cross-Attention (xformers)

        Optimized: 4-modality Batched Attention, eliminates memory copies and serial kernel launches.
        - KV projection outputs native xformers 4D layout (B, N, H, D), no permute/contiguous
        - 4 modalities padded to max_len then cat along batch dim, single xformers call
        - Padding regions masked via attn_bias (-inf)

        [WMA-HPC] KV Cache: context is constant across 50 DDIM steps, K/V projections and attn_bias
        computed only on first step and cached, subsequent 49 steps reuse directly, eliminating ~3400 redundant Linear projections.
        """
        _, fused_combine_fn = _get_compiled_fns()

        b, n, _ = x.shape
        h = self.heads
        d = self.dim_head

        # 1. Query: project + reshape to xformers 4D layout (B, N, H, D)
        #    Query depends on x (changes each step), cannot be cached
        q = self.to_q(x).view(b, n, h, d)
        # Replicate 4 copies for batched attention (physical copy cost much less than 4 kernel launches)
        q_all = q.repeat(4, 1, 1, 1)  # (4*B, N, H, D)

        # 2. KV + attn_bias: only depends on context (constant in DDIM loop) -- cacheable
        if self._kv_cache is not None and not self._kv_cache_fill:
            k_all, v_all, attn_bias = self._kv_cache
        else:
            batched_proj_fn = _get_compiled_batched_proj_fn()

            # Batched KV projection: slice -> project -> pad -> cat
            len_image = context.shape[1] - (
                self.agent_state_context_len
                + self.agent_action_context_len
                + self.text_context_len
            )
            k_all, v_all = batched_proj_fn(
                context,
                self.kv_ins, self.kv_ip, self.kv_as, self.kv_aa,
                self.agent_state_context_len, self.agent_action_context_len,
                self.text_context_len, len_image,
                h, d,
            )
            max_len = k_all.shape[1]

            # Construct AgentAction block-diagonal mask: (B, 1, 1, action_len)
            action_len = self.agent_action_context_len
            num_token = action_len // 16
            start_positions = (
                (torch.arange(b, device=x.device) % 16) + 1
            ) * num_token
            col_indices = torch.arange(action_len, device=x.device)
            mask_2d = col_indices.unsqueeze(0) >= start_positions.unsqueeze(1)
            aa_mask = torch.zeros(b, 1, 1, action_len, device=x.device, dtype=q.dtype)
            aa_mask[:, 0, 0, :].masked_fill_(mask_2d, float("-inf"))

            # Build combined attention bias (4*B, 1, 1, max_len)
            lens_list = [
                self.text_context_len, len_image,
                self.agent_state_context_len, action_len,
            ]
            attn_bias = _make_batched_bias(b, max_len, lens_list, x.device, q.dtype, aa_mask)
            # expand is zero-copy (only changes stride), no extra memory
            attn_bias = attn_bias.expand(4 * b, h, n, max_len)

            # [WMA-HPC] Cache KV + bias: first time allocate, then copy_() to keep addresses stable
            if self._kv_cache_enabled:
                if self._kv_cache is None:
                    # First time: allocate static buffers
                    self._kv_cache = (k_all.clone(), v_all.clone(), attn_bias.clone())
                else:
                    # Refill: copy_() into existing buffers (same addresses for CUDA Graph)
                    self._kv_cache[0].copy_(k_all)
                    self._kv_cache[1].copy_(v_all)
                    self._kv_cache[2].copy_(attn_bias)
                k_all, v_all, attn_bias = self._kv_cache
                self._kv_cache_fill = False

        # 3. Single 4-modality Attention (replaces original 4 serial calls)
        out_all = _efficient_attention(q_all, k_all, v_all, attn_bias=attn_bias)
        # out_all: (4*B, N, H, D) -> (4*B, N, H*D)
        out_all = out_all.reshape(4 * b, n, h * d)

        # 4. Split back to per-modality outputs + fused combine
        o_txt, o_img, o_as, o_aa = out_all.chunk(4, dim=0)

        return fused_combine_fn(
            o_txt, o_img, o_as, o_aa,
            self.to_out,
            self.image_cross_attention_scale,
            self.agent_state_cross_attention_scale,
            self.agent_action_cross_attention_scale,
            None, None, None,
            False,
        )

    def _pruned_temporal_self_attn_forward(self, x):
        """Scenario C/D: TemporalTransformer.attn1 & attn2 -- Temporal Self-Attention (einsum)"""
        h = self.heads

        q = self.to_q(x)
        k, v = self.kv_ins(x)

        b, n, _ = q.shape
        h, d = self.heads, self.dim_head
        q = q.view(b, n, h, d).transpose(1, 2)
        k = k.view(b, n, h, d).transpose(1, 2)
        v = v.view(b, n, h, d).transpose(1, 2)

        # [WMA-HPC] Reduced-precision SDPA for temporal attention when decode is active
        if _WMA_BF16_DECODE_ACTIVE:
            _dtype = _WMA_REDUCED_ATTN_DTYPE
            q = q.to(_dtype)
            k = k.to(_dtype)
            v = v.to(_dtype)
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            out = out.to(x.dtype)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Output Reshape
        out = out.transpose(1, 2).contiguous().view(b, n, h * d)

        return self.to_out(out)

    def forward(self, x, context=None, mask=None):
        # [WMA-HPC] Pruned dispatch: temporal self-attention (Scenario C/D)
        if (context is None
                and mask is None
                and not self.relative_position
                and not self.image_cross_attention):
            return self._pruned_temporal_self_attn_forward(x)

        spatial_self_attn = context is None
        k_ip, v_ip, out_ip = None, None, None
        k_as, v_as, out_as = None, None, None
        k_aa, v_aa, out_aa = None, None, None

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            assert 1 > 2, (
                ">>> ERROR: should setup xformers and use efficient_forward ..."
            )
            context_agent_state = context[:, : self.agent_state_context_len, :]
            context_agent_action = context[
                :,
                self.agent_state_context_len : self.agent_state_context_len
                + self.agent_action_context_len,
                :,
            ]
            context_ins = context[
                :,
                self.agent_state_context_len
                + self.agent_action_context_len : self.agent_state_context_len
                + self.agent_action_context_len
                + self.text_context_len,
                :,
            ]
            context_image = context[
                :,
                self.agent_state_context_len
                + self.agent_action_context_len
                + self.text_context_len :,
                :,
            ]

            # [FIXED] All replaced with kv_fusion calls
            k, v = self.kv_ins(context_ins)
            k_ip, v_ip = self.kv_ip(context_image)
            k_as, v_as = self.kv_as(context_agent_state)
            k_aa, v_aa = self.kv_aa(context_agent_action)
        else:
            if not spatial_self_attn:
                context_ins = context[:, : self.text_context_len, :]
                # [FIXED] Use kv_ins
                k, v = self.kv_ins(context_ins)
            else:
                # Self Attention
                k, v = self.kv_ins(context)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
        )

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum("b t d, t s d -> b t s", q, k2) * self.scale
            sim += sim2
        del k

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            sim.masked_fill_(~(mask > 0.5), max_neg_value)

        sim = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum("b t s, t s d -> b t d", sim, v2)
            out += out2
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        if k_ip is not None and k_as is not None and k_aa is not None:
            ## for image cross-attention
            k_ip, v_ip = map(
                lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (k_ip, v_ip)
            )
            sim_ip = torch.einsum("b i d, b j d -> b i j", q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum("b i j, b j d -> b i d", sim_ip, v_ip)
            out_ip = rearrange(out_ip, "(b h) n d -> b n (h d)", h=h)

            ## for agent state cross-attention
            k_as, v_as = map(
                lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (k_as, v_as)
            )
            sim_as = torch.einsum("b i d, b j d -> b i j", q, k_as) * self.scale
            del k_as
            sim_as = sim_as.softmax(dim=-1)
            out_as = torch.einsum("b i j, b j d -> b i d", sim_as, v_as)
            out_as = rearrange(out_as, "(b h) n d -> b n (h d)", h=h)

            ## for agent action cross-attention
            k_aa, v_aa = map(
                lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (k_aa, v_aa)
            )
            sim_aa = torch.einsum("b i d, b j d -> b i j", q, k_aa) * self.scale
            del k_aa
            sim_aa = sim_aa.softmax(dim=-1)
            out_aa = torch.einsum("b i j, b j d -> b i d", sim_aa, v_aa)
            out_aa = rearrange(out_aa, "(b h) n d -> b n (h d)", h=h)

        if out_ip is not None and out_as is not None and out_aa is not None:
            if self.cross_attention_scale_learnable:
                out = (
                    out
                    + self.image_cross_attention_scale
                    * out_ip
                    * (torch.tanh(self.alpha_ctx) + 1)
                    + self.agent_state_cross_attention_scale
                    * out_as
                    * (torch.tanh(self.alpha_cas) + 1)
                    + self.agent_action_cross_attention_scale
                    * out_aa
                    * (torch.tanh(self.alpha_caa) + 1)
                )
            else:
                out = (
                    out
                    + self.image_cross_attention_scale * out_ip
                    + self.agent_state_cross_attention_scale * out_as
                    + self.agent_action_cross_attention_scale * out_aa
                )

        return self.to_out(out)

    def efficient_forward(self, x, context=None, mask=None):
        # [WMA-HPC] Pruned dispatch
        if mask is None:
            # Scenario A: spatial self-attention, no multimodal branches
            if context is None and not self.image_cross_attention:
                return self._pruned_spatial_self_attn_efficient_forward(x)
            # Scenario B: full multimodal cross-attention, non-learnable scales
            # Use >= instead of == to handle decoder case: decoder's
            # agent_action_context_len (default 16) doesn't match the actual
            # action token count (256), so the "image" portion absorbs the
            # overflow. The batched pruned path computes len_image dynamically
            # as context.shape[1] - (state + action + text), so it handles
            # both encoder (image=video_length) and decoder (image>video_length)
            # correctly with the same slicing as the fallback Case 3.
            if (context is not None
                    and self.image_cross_attention
                    and not self.cross_attention_scale_learnable
                    and self.video_length is not None
                    and context.shape[1] >= (self.agent_state_context_len
                                             + self.agent_action_context_len
                                             + self.text_context_len
                                             + self.video_length)):
                return self._pruned_spatial_cross_attn_efficient_forward(x, context)

        # [WMA-HPC] Fallback to original implementation
        fused_proj_fn, fused_combine_fn = _get_compiled_fns()

        spatial_self_attn = context is None
        k, v, out = None, None, None
        k_ip, v_ip, out_ip = None, None, None
        k_as, v_as, out_as = None, None, None
        k_aa, v_aa, out_aa = None, None, None

        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            # Case 1: Text + Image Only
            if context.shape[1] == self.text_context_len + self.video_length:
                context_ins = context[:, : self.text_context_len, :]
                context_image = context[:, self.text_context_len :, :]
                # [FIXED] Fix attribute call errors
                k, v = self.kv_ins(context_ins)
                k_ip, v_ip = self.kv_ip(context_image)

            # Case 2: State + Text + Image
            elif (context.shape[1] == self.agent_state_context_len + self.text_context_len + self.video_length):
                context_agent_state = context[:, : self.agent_state_context_len, :]
                context_ins = context[:, self.agent_state_context_len : self.agent_state_context_len + self.text_context_len, :]
                context_image = context[:, self.agent_state_context_len + self.text_context_len :, :]
                # [FIXED] Fix attribute call errors
                k, v = self.kv_ins(context_ins)
                k_ip, v_ip = self.kv_ip(context_image)
                k_as, v_as = self.kv_as(context_agent_state)

            # Case 3: Action + State + Text + Image (Full Multimodal)
            else:
                # [WMA-HPC] KV Cache for fallback path
                if self._kv_cache is not None and not self._kv_cache_fill:
                    k, v, k_ip, v_ip, k_as, v_as, k_aa, v_aa, attn_mask_aa = self._kv_cache
                else:
                    # [HPC] Call global compiled function
                    k, v, k_ip, v_ip, k_as, v_as, k_aa, v_aa = fused_proj_fn(
                        context,
                        self.kv_ins, self.kv_ip, self.kv_as, self.kv_aa,
                        self.agent_state_context_len, self.agent_action_context_len, self.text_context_len
                    )

                    attn_mask_aa = self._get_attn_mask_aa(
                        x.shape[0], q.shape[1], k_aa.shape[1],
                        block_size=16, device=k_aa.device,
                    )

                    if self._kv_cache_enabled:
                        if self._kv_cache is None:
                            self._kv_cache = (k.clone(), v.clone(), k_ip.clone(), v_ip.clone(),
                                              k_as.clone(), v_as.clone(), k_aa.clone(), v_aa.clone(),
                                              attn_mask_aa.clone())
                        else:
                            self._kv_cache[0].copy_(k); self._kv_cache[1].copy_(v)
                            self._kv_cache[2].copy_(k_ip); self._kv_cache[3].copy_(v_ip)
                            self._kv_cache[4].copy_(k_as); self._kv_cache[5].copy_(v_as)
                            self._kv_cache[6].copy_(k_aa); self._kv_cache[7].copy_(v_aa)
                            self._kv_cache[8].copy_(attn_mask_aa)
                        k, v, k_ip, v_ip, k_as, v_as, k_aa, v_aa, attn_mask_aa = self._kv_cache
                        self._kv_cache_fill = False
        else:
            # [FIXED] Fix context_ins undefined variable issue
            if not spatial_self_attn:
                input_context = context[:, :self.text_context_len, :]
            else:
                input_context = context

            # [FIXED] Fix attribute call errors
            k, v = self.kv_ins(input_context)

        b, _, _ = q.shape
        q = (
            q.unsqueeze(3)
            .reshape(b, q.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, q.shape[1], self.dim_head)
            .contiguous()
        )

        if k is not None:
             k, v = map(lambda t: t.unsqueeze(3).reshape(b, t.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(b * self.heads, t.shape[1], self.dim_head).contiguous(), (k, v))
             out = _efficient_attention_3d(q, k, v, b, self.heads, self.dim_head)

        if k_ip is not None:
             k_ip, v_ip = map(lambda t: t.unsqueeze(3).reshape(b, t.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(b * self.heads, t.shape[1], self.dim_head).contiguous(), (k_ip, v_ip))
             out_ip = _efficient_attention_3d(q, k_ip, v_ip, b, self.heads, self.dim_head)

        if k_as is not None:
             k_as, v_as = map(lambda t: t.unsqueeze(3).reshape(b, t.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(b * self.heads, t.shape[1], self.dim_head).contiguous(), (k_as, v_as))
             out_as = _efficient_attention_3d(q, k_as, v_as, b, self.heads, self.dim_head)

        if k_aa is not None:
             k_aa, v_aa = map(lambda t: t.unsqueeze(3).reshape(b, t.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(b * self.heads, t.shape[1], self.dim_head).contiguous(), (k_aa, v_aa))

             attn_mask_aa = attn_mask_aa.unsqueeze(1).expand(b, self.heads, attn_mask_aa.shape[1], attn_mask_aa.shape[2])
             attn_mask_aa = attn_mask_aa.to(q.dtype)

             out_aa = _efficient_attention_3d(q, k_aa, v_aa, b, self.heads, self.dim_head, attn_bias=attn_mask_aa)

        # -----------------------------------------------------------
        # Output Fusion Phase
        # -----------------------------------------------------------
        # [WMA-HPC] Ensure all tensors have consistent shapes, using original input x shape as fallback
        # x.shape = [B, N, C]
        zero_tensor = x.new_zeros(x.shape)

        out = zero_tensor if out is None else out
        out_ip = zero_tensor if out_ip is None else out_ip
        out_as = zero_tensor if out_as is None else out_as
        out_aa = zero_tensor if out_aa is None else out_aa

        # Core logic: only use Compiled Kernel when truly full multimodal (Action/State/Image all present)
        multimodal_ready = (k_ip is not None) and (k_as is not None) and (k_aa is not None)

        if multimodal_ready:
            # [HPC] Call global compiled function
            # Note: all parameters must be explicitly passed
            return fused_combine_fn(
                out, out_ip, out_as, out_aa,
                self.to_out, # Pass Module
                self.image_cross_attention_scale,
                self.agent_state_cross_attention_scale,
                self.agent_action_cross_attention_scale,
                self.alpha_ctx if hasattr(self, 'alpha_ctx') else None,
                self.alpha_cas if hasattr(self, 'alpha_cas') else None,
                self.alpha_caa if hasattr(self, 'alpha_caa') else None,
                self.cross_attention_scale_learnable # Bool
            )

        # [WMA-HPC] Fallback branch
        # Even if not full multimodal, add up existing partial results
        # Note: out_ip etc. are already zero_tensor, so addition is safe
        if self.cross_attention_scale_learnable:
             out = (
                out
                + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha_ctx) + 1)
                + self.agent_state_cross_attention_scale * out_as * (torch.tanh(self.alpha_cas) + 1)
                + self.agent_action_cross_attention_scale * out_aa * (torch.tanh(self.alpha_caa) + 1)
            )
        else:
            out = (
                out
                + self.image_cross_attention_scale * out_ip
                + self.agent_state_cross_attention_scale * out_as
                + self.agent_action_cross_attention_scale * out_aa
            )

        return self.to_out(out)

    def _get_attn_mask_aa(self, b, l1, l2, block_size=16, device=None):
        num_token = l2 // block_size
        start_positions = (
            (torch.arange(b, device=device) % block_size) + 1
        ) * num_token
        col_indices = torch.arange(l2, device=device)
        mask_2d = col_indices.unsqueeze(0) >= start_positions.unsqueeze(1)
        mask = mask_2d.unsqueeze(1).expand(b, l1, l2)
        attn_mask = torch.zeros_like(mask, dtype=torch.float)
        attn_mask[mask] = float("-inf")
        return attn_mask

def _global_ff_impl(x, norm_module, ff_module):
    """
    Stateless implementation of the FeedForward block logic.
    Fused path: x + FeedForward(LayerNorm(x))
    """
    # Inductor will auto-fuse Norm -> FF -> Add together
    # Note: ff_module internally typically contains Linear -> Act -> Dropout -> Linear
    # torch.compile can see through these layers for full-graph optimization
    return ff_module(norm_module(x)) + x


# =========================================================================
# [WMA-HPC] Global compiled handle management
# =========================================================================
_COMPILED_FF_FN = None

def _get_compiled_ff_fn():
    global _COMPILED_FF_FN
    if _COMPILED_FF_FN is None:
        if os.getenv("WMA_COMPILE_FF_BLOCK", "0") == "1":
            print("[WMA-HPC] Initializing GLOBAL compiled kernel for BasicTransformerBlock FF...")

            _COMPILED_FF_FN = torch.compile(
                _global_ff_impl,
                mode="reduce-overhead",
                fullgraph=True
            )
        else:
            _COMPILED_FF_FN = _global_ff_impl
    return _COMPILED_FF_FN

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attention_cls=None,
        video_length=None,
        agent_state_context_len=2,
        agent_action_context_len=16,
        image_cross_attention=False,
        image_cross_attention_scale=1.0,
        cross_attention_scale_learnable=False,
        text_context_len=77,
    ):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )
        #
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            video_length=video_length,
            agent_state_context_len=agent_state_context_len,
            agent_action_context_len=agent_action_context_len,
            image_cross_attention=image_cross_attention,
            image_cross_attention_scale=image_cross_attention_scale,
            cross_attention_scale_learnable=cross_attention_scale_learnable,
            text_context_len=text_context_len,
        )
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, **kwargs):
        # implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (
            x,
        )  # should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        return checkpoint(
            self._forward, input_tuple, self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None, mask=None):
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                mask=mask,
            )
            + x
        )
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        fused_ff_fn = _get_compiled_ff_fn()

        x = fused_ff_fn(x, self.norm3, self.ff)

        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        disable_self_attn=False,
        use_linear=False,
        video_length=None,
        agent_state_context_len=2,
        agent_action_context_len=16,
        image_cross_attention=False,
        cross_attention_scale_learnable=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    attention_cls=attention_cls,
                    video_length=video_length,
                    agent_state_context_len=agent_state_context_len,
                    agent_action_context_len=agent_action_context_len,
                    image_cross_attention=image_cross_attention,
                    cross_attention_scale_learnable=cross_attention_scale_learnable,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear
        self.compute_dtype = None  # set to torch.bfloat16 to enable bf16 attention

    def forward(self, x, context=None, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        _amp = (torch.amp.autocast('cuda', dtype=self.compute_dtype)
                if self.compute_dtype else nullcontext())
        with _amp:
            if not self.use_linear:
                x = self.proj_in(x)
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            if self.use_linear:
                x = self.proj_in(x)
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, context=context, **kwargs)
            if self.use_linear:
                x = self.proj_out(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            if not self.use_linear:
                x = self.proj_out(x)
        if self.compute_dtype:
            x = x.to(x_in.dtype)
        return x + x_in


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=True,
        use_linear=False,
        only_self_att=True,
        causal_attention=False,
        causal_block_size=1,
        relative_position=False,
        temporal_length=None,
    ):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Conv1d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )
        if not use_linear:
            self.proj_in = nn.Conv1d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert temporal_length is not None
            attention_cls = partial(
                CrossAttention, relative_position=True, temporal_length=temporal_length
            )
        else:
            attention_cls = partial(CrossAttention, temporal_length=temporal_length)
        if self.causal_attention:
            assert temporal_length is not None
            self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attention_cls=attention_cls,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear
        self.compute_dtype = None  # set to torch.bfloat16 to enable bf16 attention

    def forward(self, x, context=None):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        _amp = (torch.amp.autocast('cuda', dtype=self.compute_dtype)
                if self.compute_dtype else nullcontext())
        with _amp:
            x = rearrange(x, "b c t h w -> (b h w) c t").contiguous()

            if not self.use_linear:
                x = self.proj_in(x)
            x = rearrange(x, "bhw c t -> bhw t c").contiguous()
            if self.use_linear:
                x = self.proj_in(x)

            temp_mask = None
            if self.causal_attention:
                # Slice the from mask map
                temp_mask = self.mask[:, :t, :t].to(x.device)

            if temp_mask is not None:
                mask = temp_mask.to(x.device)
                mask = repeat(mask, "l i j -> (l bhw) i j", bhw=b * h * w)
            else:
                mask = None

            if self.only_self_att:
                # NOTE: if no context is given, cross-attention defaults to self-attention
                for i, block in enumerate(self.transformer_blocks):
                    x = block(x, mask=mask)
                x = rearrange(x, "(b hw) t c -> b hw t c", b=b).contiguous()
            else:
                x = rearrange(x, "(b hw) t c -> b hw t c", b=b).contiguous()
                context = rearrange(
                    context, "(b t) l con -> b t l con", t=t
                ).contiguous()
                for i, block in enumerate(self.transformer_blocks):
                    # Calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                    for j in range(b):
                        context_j = repeat(
                            context[j], "t l con -> (t r) l con", r=(h * w) // t, t=t
                        ).contiguous()
                        # Note: causal mask will not applied in cross-attention case
                        x[j] = block(x[j], context=context_j)

            if self.use_linear:
                x = self.proj_out(x)
                x = rearrange(x, "b (h w) t c -> b c t h w", h=h, w=w).contiguous()
            if not self.use_linear:
                x = rearrange(x, "b hw t c -> (b hw) c t").contiguous()
                x = self.proj_out(x)
                x = rearrange(x, "(b h w) c t -> b c t h w", b=b, h=h, w=w).contiguous()

        if self.compute_dtype:
            x = x.to(x_in.dtype)
        return x + x_in


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv,
            "b (qkv heads c) h w -> qkv b heads c (h w)",
            heads=self.heads,
            qkv=3,
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # Attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_
