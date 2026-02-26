"""WMA Inference Daemon -- persistent server holding loaded model."""

# [WMA-HPC] Redirect Triton cache to writable dir (must be before any triton import)
import os as _os
if not _os.access(_os.path.expanduser('~/.triton/cache'), _os.W_OK):
    _tc = '/tmp/triton_cache'
    _os.makedirs(_tc, exist_ok=True)
    _os.environ['TRITON_CACHE_DIR'] = _tc

import argparse, os, sys, glob, signal, time, traceback
import pandas as pd
import random
import torch
import torchvision
import h5py
import numpy as np
import logging
import einops
import warnings
try:
    import imageio
except ImportError:
    pass  # imageio not used in daemon code path

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict, deque
from torch import nn, Tensor
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file
from multiprocessing import shared_memory

from unifolm_wma.models.samplers.ddim import DDIMSampler
from unifolm_wma.utils.utils import instantiate_from_config
import torch._dynamo

torch._dynamo.config.accumulated_cache_size_limit = 2048

# NOTE: cudnn.benchmark is temporarily enabled during CUDA Graph warmup
# (inside _capture_cuda_graph_kv_pair) to auto-tune conv algorithms, then
# disabled before actual capture to avoid deadlocks.

# IPC protocol
from wma_ipc import (
    SHM_NAME, SHM_SIZE, PID_FILE, LOG_FILE,
    STATE_IDLE, STATE_REQUEST, STATE_RUNNING, STATE_DONE, STATE_ERROR,
    read_ctrl, write_ctrl, read_state, write_state, cleanup_shm,
)

# eval_utils (same directory)
from eval_utils import populate_queues


# ========================================================================
# Migrated functions (verbatim from world_model_interaction.py)
# ========================================================================

def compile_kernel():
    # All torch.compile(reduce-overhead) flags are disabled — they create nested CUDA
    # Graphs that conflict with the manual WMAModel-level CUDA Graph capture.
    # os.environ["WMA_COMPILE_OBS"] = "1"
    # os.environ["WMA_COMPILE_CONV1D_BLOCK"] = "1"
    # os.environ["WMA_COMPILE_TEMPORAL_CONV_BLOCK"] = "1"
    # os.environ["WMA_COMPILE_CROSS_ATTN"] = "1"
    # os.environ["WMA_COMPILE_FF_BLOCK"] = "1"

    # ConditionalUnet1D-level CUDA Graph is subsumed by WMAModel-level graph.
    # os.environ["WMA_CUDA_GRAPH_ACTION_UNET"] = "1"

    # Two-phase CUDA Graph: KV cache fills on step 0, graph captures
    # the KV-hot path and replays for steps 1-49. This subsumes both
    # the old single-graph and KV-cache-only modes.
    if os.environ.get('WMA_KV_CUDA_GRAPH') == '1':
        print("[WMA-HPC] Two-phase CUDA Graph: KV cache + graph replay", flush=True)
        os.environ['WMA_KV_CACHE'] = '1'
        os.environ.pop("WMA_CUDA_GRAPH_WMA_MODEL", None)
    elif os.environ.get('WMA_KV_CACHE') == '1':
        print("[WMA-HPC] KV Cache enabled — auto-disabling WMAModel CUDA Graph", flush=True)
        os.environ.pop("WMA_CUDA_GRAPH_WMA_MODEL", None)
    elif not os.environ.get('WMA_NO_CUDA_GRAPH'):
        print("[WMA-HPC] Set WMAModel manual CUDA Graph flag ...", flush=True)
        os.environ["WMA_CUDA_GRAPH_WMA_MODEL"] = "1"
    else:
        print("[WMA-HPC] SKIPPED CUDA Graph (WMA_NO_CUDA_GRAPH)", flush=True)

    print("[WMA-HPC] Skip OpenCLIP pretrained loading (weights from safetensors) ...", flush=True)
    os.environ["WMA_SKIP_CLIP_PRETRAINED"] = "1"

    print("[WMA-HPC] Skip EMA deepcopy (inference only) ...", flush=True)
    os.environ["WMA_INFERENCE"] = "1"


def get_device_from_parameters(module: nn.Module) -> torch.device:
    return next(iter(module.parameters())).device


def load_model_checkpoint(model: nn.Module, ckpt: str) -> nn.Module:
    """Load model weights from checkpoint file.

    Supports .safetensors (mmap zero-copy) and legacy .ckpt formats.
    If .ckpt path given but same-name .safetensors exists, auto-uses latter.
    """
    import time as _time

    safetensors_path = ckpt.rsplit(".", 1)[0] + ".safetensors" if not ckpt.endswith(".safetensors") else ckpt
    use_safetensors = os.path.exists(safetensors_path)

    nhwc_path = safetensors_path.rsplit(".", 1)[0] + "_nhwc.safetensors"
    if os.path.exists(nhwc_path):
        safetensors_path = nhwc_path
        use_safetensors = True

    t0 = _time.perf_counter()
    if use_safetensors:
        print(f'>>> Loading safetensors: {safetensors_path}')
        from safetensors import safe_open
        sf = safe_open(safetensors_path, framework="pt", device="cpu")
        is_nhwc = sf.metadata().get("nhwc_format") == "1" if sf.metadata() else False
        state_dict = {k: sf.get_tensor(k) for k in sf.keys()}
        if os.environ.get("WMA_INFERENCE") == "1":
            ema_keys = [k for k in state_dict if k.startswith("dp_ema_model.")]
            if ema_keys:
                print(f'>>> WMA_INFERENCE=1: filtering {len(ema_keys)} dp_ema_model.* keys')
                for k in ema_keys:
                    del state_dict[k]
        if is_nhwc:
            print(f'>>> NHWC format detected, permuting back to contiguous layout ...')
            for k, v in state_dict.items():
                if v.dim() == 4:
                    state_dict[k] = v.permute(0, 3, 1, 2).contiguous()
                elif v.dim() == 5:
                    state_dict[k] = v.permute(0, 4, 1, 2, 3).contiguous()
                elif v.dim() == 3:
                    state_dict[k] = v.permute(0, 2, 1).contiguous()
        model.load_state_dict(state_dict, strict=True, assign=True)
    else:
        print(f'>>> Loading ckpt: {ckpt}')
        state_dict = torch.load(ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
            # Filter EMA keys when WMA_INFERENCE=1 (EMA model is not instantiated)
            if os.environ.get("WMA_INFERENCE") == "1":
                ema_keys = [k for k in state_dict if k.startswith("dp_ema_model.")]
                if ema_keys:
                    print(f'>>> WMA_INFERENCE=1: filtering {len(ema_keys)} dp_ema_model.* keys from ckpt')
                    for k in ema_keys:
                        del state_dict[k]
            try:
                model.load_state_dict(state_dict, strict=True)
            except:
                new_pl_sd = OrderedDict()
                for k, v in state_dict.items():
                    new_pl_sd[k] = v
                for k in list(new_pl_sd.keys()):
                    if "framestride_embed" in k:
                        new_key = k.replace("framestride_embed", "fps_embedding")
                        new_pl_sd[new_key] = new_pl_sd[k]
                        del new_pl_sd[k]
                model.load_state_dict(new_pl_sd, strict=True)
        else:
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]] = state_dict['module'][key]
            model.load_state_dict(new_pl_sd)

    elapsed = _time.perf_counter() - t0
    print(f'>>> model checkpoint loaded in {elapsed:.1f}s.')
    return model


def save_results(video: Tensor, filename: str, fps: int = 8) -> None:
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4)
    frame_grids = [
        torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0)
        for framesheet in video
    ]
    grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(filename, grid, fps=fps,
                               video_codec='h264', options={'crf': '10'})


def get_init_frame_path(data_dir: str, sample: dict) -> str:
    rel_video_fp = os.path.join(sample['data_dir'],
                                str(sample['videoid']) + '.png')
    full_image_fp = os.path.join(data_dir, 'images', rel_video_fp)
    return full_image_fp


def get_transition_path(data_dir: str, sample: dict) -> str:
    rel_transition_fp = os.path.join(sample['data_dir'],
                                     str(sample['videoid']) + '.h5')
    full_transition_fp = os.path.join(data_dir, 'transitions',
                                      rel_transition_fp)
    return full_transition_fp


def prepare_init_input(start_idx, init_frame_path, transition_dict,
                       frame_stride, wma_data, video_length=16,
                       n_obs_steps=2):
    indices = [start_idx + frame_stride * i for i in range(video_length)]
    init_frame = Image.open(init_frame_path).convert('RGB')
    init_frame = torch.tensor(np.array(init_frame)).unsqueeze(0).permute(
        3, 0, 1, 2).float()

    if start_idx < n_obs_steps - 1:
        state_indices = list(range(0, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]
        num_padding = n_obs_steps - 1 - start_idx
        first_slice = states[0:1, :]
        padding = first_slice.repeat(num_padding, 1)
        states = torch.cat((padding, states), dim=0)
    else:
        state_indices = list(range(start_idx - n_obs_steps + 1, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]

    actions = transition_dict['action'][indices, :]

    ori_state_dim = states.shape[-1]
    ori_action_dim = actions.shape[-1]

    frames_action_state_dict = {
        'action': actions,
        'observation.state': states,
    }
    frames_action_state_dict = wma_data.normalizer(frames_action_state_dict)
    frames_action_state_dict = wma_data.get_uni_vec(
        frames_action_state_dict,
        transition_dict['action_type'],
        transition_dict['state_type'],
    )

    if wma_data.spatial_transform is not None:
        init_frame = wma_data.spatial_transform(init_frame)
    init_frame = (init_frame / 255 - 0.5) * 2

    data = {
        'observation.image': init_frame,
    }
    data.update(frames_action_state_dict)
    return data, ori_state_dim, ori_action_dim


def get_latent_z(model, videos: Tensor) -> Tensor:
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z



class _Gap2Graph:
    """CUDA graph for gap2: VAE decode wm latents → pixel video.

    Only captures VAE decode (the dominant cost at ~15-25ms).  All other
    conditioning (CLIP encode, state/action projections, VAE encode for
    img_cat_cond) runs through the normal queue-based flow inside
    image_guided_synthesis_sim_mode to preserve exact numerical equivalence
    with the original inference pipeline.
    """
    _WARMUP = 3

    def __init__(self, model, noise_shape):
        self.model = model
        device = next(model.parameters()).device
        B, C_lat, T, H_lat, W_lat = noise_shape

        # Static input buffer (caller copies into this before replay)
        self._in_wm_samples = torch.empty(
            B, C_lat, T, H_lat, W_lat, device=device)

        self._out = None
        self._graph = None
        self._stream = torch.cuda.Stream(device=device)

    def _impl(self):
        return self.model.decode_first_stage(self._in_wm_samples)

    def capture(self):
        s = self._stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._WARMUP):
                self._impl()
        torch.cuda.current_stream().wait_stream(s)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._out = self._impl()
        print("[WMA-HPC] Gap2 CUDA Graph captured (VAE decode only)")

    def run(self, wm_samples):
        """Copy wm latents, replay graph, return decoded pixel video.

        Returns a cloned tensor safe to hold across replays.
        """
        self._in_wm_samples.copy_(wm_samples)
        self._graph.replay()
        return self._out.clone()


def image_guided_synthesis_sim_mode(
        model, prompts, observation, noise_shape,
        action_cond_step=16, n_samples=1, ddim_steps=50,
        ddim_eta=1.0, unconditional_guidance_scale=1.0,
        fs=None, text_input=True, timestep_spacing='uniform',
        guidance_rescale=0.0, sim_mode=True,
        skip_decode=False,
        ddim_sampler=None,
        cached_cond=None,
        **kwargs):
    """
    Args:
        skip_decode: If True, skip VAE decode (return None for videos).
            Use when gap2 will decode separately via CUDA graph.
        ddim_sampler: Optional pre-created DDIMSampler to reuse across calls.
        cached_cond: Optional dict with pre-computed conditioning embeddings
            to avoid redundant CLIP/VAE forward passes within the same iteration.
            Keys: 'cond_img_emb', 'img_cat_cond', 'cond_state_emb'
    Returns:
        (batch_variants, actions, states, img_cond, samples)
        - samples: raw DDIM latent output (before VAE decode).
    """
    b, _, t, _, _ = noise_shape
    # [WMA-HPC] Reuse DDIMSampler across calls (schedule cached internally)
    if ddim_sampler is None:
        ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]

    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    # --- Conditioning ---
    # [WMA-HPC] Reuse pre-computed embeddings if available
    if cached_cond is not None:
        cond_img_emb = cached_cond['cond_img_emb']
        img_cat_cond = cached_cond['img_cat_cond']
        cond_state_emb = cached_cond['cond_state_emb']
    else:
        img = observation['observation.images.top'].permute(0, 2, 1, 3, 4)
        cond_img = rearrange(img, 'b o c h w -> (b o) c h w')[-1:]
        cond_img_emb = model.embedder(cond_img)
        cond_img_emb = model.image_proj_model(cond_img_emb)

        img_cat_cond = None
        if model.model.conditioning_key == 'hybrid':
            z = get_latent_z(model, img.permute(0, 2, 1, 3, 4))
            img_cat_cond = z[:, :, -1:, :, :]
            img_cat_cond = repeat(img_cat_cond,
                                  'b c t h w -> b c (repeat t) h w',
                                  repeat=noise_shape[2])

        cond_state_emb = model.state_projector(observation['observation.state'])
        cond_state_emb = cond_state_emb + model.agent_state_pos_emb

    img_cond = {'cond_img_emb': cond_img_emb, 'img_cat_cond': img_cat_cond}
    cond = {"c_concat": [img_cat_cond]}

    # Text / state / action conditioning
    if not text_input:
        prompts = [""] * batch_size
    cond_ins_emb = model.get_learned_conditioning(prompts)

    # [WMA-HPC] Action embedding always recomputed (changes between dm/wm)
    cond_action_emb = model.action_projector(observation['action'])
    cond_action_emb = cond_action_emb + model.agent_action_pos_emb

    if not sim_mode:
        cond_action_emb = torch.zeros_like(cond_action_emb)

    cond["c_crossattn"] = [
        torch.cat(
            [cond_state_emb, cond_action_emb, cond_ins_emb, cond_img_emb],
            dim=1)
    ]
    cond["c_crossattn_action"] = [
        observation['observation.images.top'][:, :,
                                              -model.n_obs_steps_acting:],
        observation['observation.state'][:, -model.n_obs_steps_acting:],
        sim_mode,
        False,
    ]

    uc = None
    kwargs.update({"unconditional_conditioning_img_nonetext": None})
    cond_mask = None
    cond_z0 = None
    if ddim_sampler is not None:
        samples, actions, states, intermedia = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            cfg_img=None,
            mask=cond_mask,
            x0=cond_z0,
            fs=fs,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            store_intermediates=False,
            **kwargs)

        if skip_decode:
            batch_variants = None
        else:
            batch_images = model.decode_first_stage(samples)
            batch_variants = batch_images

    return batch_variants, actions, states, img_cond, samples


# ========================================================================
# Daemon-specific functions
# ========================================================================

def init_model(args):
    """One-time model initialization. Returns (model, config, data, noise_shape, device)."""
    compile_kernel()

    config = OmegaConf.load(args.config)
    config['model']['params']['wma_config']['params']['use_checkpoint'] = False

    # Skip nn.init (same monkey-patch as original)
    import torch.nn.init as _init
    import torch.nn.modules.activation as _act_mod
    _noop = lambda tensor, *a, **kw: tensor
    _init_names = ['uniform_', 'normal_', 'kaiming_uniform_', 'kaiming_normal_',
                   'xavier_uniform_', 'xavier_normal_', 'constant_', 'zeros_',
                   'ones_', 'trunc_normal_', 'orthogonal_']
    _orig = {n: getattr(_init, n) for n in _init_names if hasattr(_init, n)}
    for n in _orig:
        setattr(_init, n, _noop)
    _act_orig = {}
    for n in _init_names:
        if hasattr(_act_mod, n):
            _act_orig[n] = getattr(_act_mod, n)
            setattr(_act_mod, n, _noop)

    model = instantiate_from_config(config.model)

    for n, fn in _orig.items():
        setattr(_init, n, fn)
    for n, fn in _act_orig.items():
        setattr(_act_mod, n, fn)

    model.perframe_ae = args.perframe_ae
    # Also accept .safetensors variant if .ckpt doesn't exist (broken symlinks)
    _st_path = args.ckpt_path.rsplit(".", 1)[0] + ".safetensors" if not args.ckpt_path.endswith(".safetensors") else args.ckpt_path
    assert os.path.exists(args.ckpt_path) or os.path.exists(_st_path), f"Checkpoint not found: {args.ckpt_path} (also checked {_st_path})"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    logging.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()

    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda(0)

    # --- Ablation env vars ---
    # WMA_NO_BF16_ATTENTION=1  → skip bf16 attention
    # WMA_NO_FUSED_GN=1        → skip fused GroupNorm
    # WMA_NO_FUSED_GEGLU=1     → skip fused GEGLU
    # WMA_NO_CHANNELS_LAST=1   → skip channels_last (+ conv1d→conv2d)
    # WMA_NO_BF16_BACKBONE=1   → skip bf16 backbone conversion
    no_bf16_attn = os.environ.get('WMA_NO_BF16_ATTENTION')
    no_fused_gn = os.environ.get('WMA_NO_FUSED_GN')
    no_fused_geglu = os.environ.get('WMA_NO_FUSED_GEGLU')
    no_channels_last = os.environ.get('WMA_NO_CHANNELS_LAST')
    no_bf16_backbone = os.environ.get('WMA_NO_BF16_BACKBONE')

    if not no_bf16_attn:
        from unifolm_wma.ops.wrappers.bf16_attention import enable_bf16_attention
        _attn_dtype_str = os.environ.get('WMA_ATTENTION_DTYPE', 'bf16')
        _attn_dtype = torch.float16 if _attn_dtype_str == 'fp16' else torch.bfloat16
        n_bf16 = enable_bf16_attention(model.model.diffusion_model, target_levels='all', dtype=_attn_dtype)
        print(f">>> Enabled {_attn_dtype_str} attention on {n_bf16} transformer modules")
    else:
        print(">>> SKIPPED bf16 attention (WMA_NO_BF16_ATTENTION)")

    if not no_fused_gn:
        from unifolm_wma.ops.wrappers.fused_groupnorm import enable_fused_groupnorm
        n_fused = enable_fused_groupnorm(model.model.diffusion_model)
        n_fused_vae = enable_fused_groupnorm(model.first_stage_model)
        print(f">>> Enabled fused GroupNorm on {n_fused} backbone + {n_fused_vae} VAE module(s)")
    else:
        print(">>> SKIPPED fused GroupNorm (WMA_NO_FUSED_GN)")

    if not no_fused_geglu:
        from unifolm_wma.ops.wrappers.fused_geglu import enable_fused_geglu
        n_geglu = enable_fused_geglu(model.model.diffusion_model)
        print(f">>> Enabled fused GEGLU on {n_geglu} modules")
    else:
        print(">>> SKIPPED fused GEGLU (WMA_NO_FUSED_GEGLU)")

    if not no_channels_last:
        from unifolm_wma.ops.wrappers.conv1d_to_conv2d import enable_conv2d_nhwc
        n_conv2d = enable_conv2d_nhwc(model.model.diffusion_model.action_unet)
        print(f">>> Conv1d→Conv2d: {n_conv2d} modules upgraded in action_unet")
    else:
        print(">>> SKIPPED Conv1d→Conv2d + channels_last (WMA_NO_CHANNELS_LAST)")

    if not no_bf16_backbone:
        model.model.diffusion_model.to(torch.bfloat16)
        model.model.diffusion_model.dtype = torch.bfloat16
        model.model.diffusion_model.action_unet.to(torch.float32)
        model.model.diffusion_model.state_unet.to(torch.float32)
        print(">>> Converted diffusion backbone to bf16 (heads kept fp32)")
        # NOTE: PyTorch LayerNorm CUDA kernel already computes mean/var in fp32
        # accumulators even for bf16 input. Autocast also gives identical results
        # because every LN output is immediately consumed by a bf16 matmul.
        # The 5.8dB bf16-vs-fp32 gap is from accumulated quantization across ALL
        # tensor boundaries (50 DDIM steps × 30+ layers), not any single op.
    else:
        print(">>> SKIPPED bf16 backbone (WMA_NO_BF16_BACKBONE)")

    if not no_channels_last:
        from unifolm_wma.ops.wrappers.fused_groupnorm import enable_channels_last
        n_cl = enable_channels_last(model.model.diffusion_model)
        print(f">>> Enabled channels_last ({n_cl} params)")
    model._use_channels_last = not no_channels_last

    # NOTE: VAE stays fp32. bf16 VAE causes PSNR to drop from 29dB to 16.7dB.
    # The DiagonalGaussianDistribution.sample() in encoder is precision-sensitive.

    device = get_device_from_parameters(model)

    assert (args.height % 16 == 0) and (args.width % 16 == 0)
    assert args.bs == 1

    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    noise_shape = [args.bs, channels, args.video_length, h, w]

    # [WMA-HPC] cuDNN benchmark warmup: run a few forward passes to let
    # cudnn.benchmark find optimal conv algorithms before CUDA Graph capture.
    # The captured graph then uses the best algorithms for every replay.
    # NOTE: cuDNN benchmark is NOT enabled by default — benchmarked algorithms
    # cause numerical instability in CUDA graph replay (Winograd / workspace
    # corruption). Only enable manually if needed for profiling.
    if torch.backends.cudnn.benchmark:
        import time as _time
        _t0 = _time.time()
        dm = model.model.diffusion_model
        B, T = args.bs, args.video_length
        L = model.n_obs_steps_imagen + 16 + 77 + T * 16
        n_act = model.n_obs_steps_acting
        with torch.inference_mode():
            dummy_x = torch.randn(B, channels * 2, T, h, w, device=device)
            dummy_x_action = torch.randn(B, T, model.agent_action_dim, device=device)
            dummy_x_state = torch.randn(B, T, model.agent_state_dim, device=device)
            dummy_timesteps = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)
            dummy_context = torch.randn(B, L, 1024, device=device)
            dummy_context_action = [
                torch.randn(B, 3, n_act, args.height, args.width, device=device),
                torch.randn(B, n_act, model.agent_state_dim, device=device),
                False,
                False,
            ]
            dummy_fs = torch.tensor([8] * B, dtype=torch.long, device=device)
            # 3 warmup calls: first benchmarks convs, subsequent confirm
            # NOTE: features_adapter=None (default), fs must be passed by keyword
            for _w in range(3):
                dm(dummy_x, dummy_x_action, dummy_x_state,
                   dummy_timesteps, dummy_context, dummy_context_action,
                   None, dummy_fs)
            torch.cuda.synchronize()
        _t1 = _time.time()
        print(f">>> cuDNN benchmark warmup done in {_t1-_t0:.1f}s", flush=True)

    # Pre-capture CUDA Graph at init time (avoids first-request latency)
    if os.environ.get("WMA_CUDA_GRAPH_WMA_MODEL") == "1":
        dm = model.model.diffusion_model
        B, T = args.bs, args.video_length
        L = model.n_obs_steps_imagen + 16 + 77 + T * 16  # context seq len
        n_act = model.n_obs_steps_acting
        with torch.inference_mode():
            dummy_x = torch.randn(B, channels * 2, T, h, w, device=device)
            dummy_x_action = torch.randn(B, T, model.agent_action_dim, device=device)
            dummy_x_state = torch.randn(B, T, model.agent_state_dim, device=device)
            dummy_timesteps = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)
            dummy_context = torch.randn(B, L, 1024, device=device)
            dummy_context_action = [
                torch.randn(B, 3, n_act, args.height, args.width, device=device),
                torch.randn(B, n_act, model.agent_state_dim, device=device),
                False,
                False,
            ]
            dummy_fs = torch.tensor([8] * B, dtype=torch.long, device=device)
            dm._capture_cuda_graph(
                dummy_x, dummy_x_action, dummy_x_state,
                dummy_timesteps, dummy_context, dummy_context_action, dummy_fs)
        print(">>> WMAModel CUDA Graph pre-captured at init time")

    # Batch all 16 frames through VAE decode instead of sequential per-frame
    no_cuda_graph = os.environ.get('WMA_NO_CUDA_GRAPH')
    # Gap2 (VAE decode CUDA graph) is useful even in eager model mode
    no_gap2 = os.environ.get('WMA_NO_GAP2')
    _use_gap2 = not no_cuda_graph or (not no_gap2 and no_cuda_graph)
    if _use_gap2:
        model.perframe_ae = False
        print(f">>> Set perframe_ae=False (batch VAE decode, was {args.perframe_ae})")
    else:
        model.perframe_ae = args.perframe_ae
        print(f">>> perframe_ae={args.perframe_ae} (CUDA Graph disabled, keep original)")

    # Pre-capture Gap2 CUDA Graph (VAE decode only)
    gap2 = None
    if _use_gap2:
        gap2 = _Gap2Graph(model, noise_shape)
        with torch.inference_mode():
            gap2.capture()
    else:
        print(">>> SKIPPED Gap2 CUDA Graph")

    # [WMA-HPC] Cross-Attention KV Cache: cache K/V projections across DDIM steps
    if os.environ.get('WMA_KV_CACHE') == '1':
        model._use_kv_cache = True
        print(">>> KV Cache enabled for cross-attention (WMA_KV_CACHE=1)")

    # Pre-capture dual CUDA Graph pair (prefill + decode) for KV cache mode
    if os.environ.get('WMA_KV_CUDA_GRAPH') == '1':
        dm = model.model.diffusion_model
        B, T = args.bs, args.video_length
        L = model.n_obs_steps_imagen + 16 + 77 + T * 16
        n_act = model.n_obs_steps_acting
        with torch.inference_mode():
            dummy_x = torch.randn(B, channels * 2, T, h, w, device=device)
            dummy_x_action = torch.randn(B, T, model.agent_action_dim, device=device)
            dummy_x_state = torch.randn(B, T, model.agent_state_dim, device=device)
            dummy_timesteps = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)
            dummy_context = torch.randn(B, L, 1024, device=device)
            dummy_context_action = [
                torch.randn(B, 3, n_act, args.height, args.width, device=device),
                torch.randn(B, n_act, model.agent_state_dim, device=device),
                False,
                False,
            ]
            dummy_fs = torch.tensor([8] * B, dtype=torch.long, device=device)
            dm._capture_cuda_graph_kv_pair(
                dummy_x, dummy_x_action, dummy_x_state,
                dummy_timesteps, dummy_context, dummy_context_action, dummy_fs)
        print(">>> KV CUDA Graph pair (prefill+decode) pre-captured at init time")

    # [WMA-HPC] Pre-create DDIMSampler for reuse across all requests
    ddim_sampler = DDIMSampler(model)
    # Pre-compute schedule with default params (50 steps, uniform_trailing, eta=1.0)
    ddim_sampler.make_schedule(ddim_num_steps=50, ddim_discretize='uniform_trailing', ddim_eta=1.0, verbose=False)
    print(">>> DDIMSampler pre-created and schedule cached")

    # [WMA-HPC] Extended BF16 decode for dm calls (WMA_DM_DDIM_STEPS).
    # The dm call only needs actions — video output is discarded.
    # dm may tolerate more BF16 steps than wm.
    # -1 = all decode steps BF16, >0 = BF16 step threshold
    dm_ddim_steps = int(os.environ.get('WMA_DM_DDIM_STEPS', '0'))
    if dm_ddim_steps == -1:
        print(">>> All-BF16 decode for dm calls (49 BF16 steps)")
    elif dm_ddim_steps > 0:
        print(f">>> Extended BF16 decode for dm calls ({dm_ddim_steps} BF16 steps)")

    # [WMA-HPC] Extended BF16 for wm calls (WMA_WM_BF16_STEPS).
    # wm video may tolerate more BF16 than dm actions.
    wm_bf16_steps = int(os.environ.get('WMA_WM_BF16_STEPS', '0'))
    if wm_bf16_steps > 0:
        print(f">>> Extended BF16 decode for wm calls ({wm_bf16_steps} BF16 steps)")

    # [WMA-HPC] Reduced total DDIM steps for dm (WMA_DM_NUM_STEPS).
    # DM only needs actions (video discarded). Fewer DDIM steps = faster DM.
    # 0 = use same steps as WM (default), >0 = total steps for DM DDIM loop.
    # Uses a DEDICATED DDIMSampler to avoid schedule contamination with WM sampler.
    dm_num_steps = int(os.environ.get('WMA_DM_NUM_STEPS', '0'))
    dm_ddim_sampler = None
    if dm_num_steps > 0:
        dm_ddim_sampler = DDIMSampler(model)
        dm_ddim_sampler.make_schedule(ddim_num_steps=dm_num_steps, ddim_discretize='uniform_trailing', ddim_eta=1.0, verbose=False)
        print(f">>> Reduced DM DDIM steps: {dm_num_steps} (vs {50} for WM), dedicated sampler created")

    return model, config, data, noise_shape, device, gap2, ddim_sampler, dm_ddim_steps, wm_bf16_steps, dm_num_steps, dm_ddim_sampler


def run_inference_request(params, model, config, data, noise_shape, device,
                          gap2, ddim_sampler=None, dm_ddim_steps=0, wm_bf16_steps=0,
                          dm_num_steps=0, dm_ddim_sampler=None):
    """Execute a single inference request. params is a dict from read_ctrl()."""
    # Build a namespace-like object from params for compatibility
    class Args:
        pass
    args = Args()
    args.savedir = params["savedir"]
    args.prompt_dir = params["prompt_dir"]
    args.dataset = params["dataset"]
    args.ddim_steps = params["ddim_steps"]
    args.ddim_eta = params["ddim_eta"]
    args.unconditional_guidance_scale = params["uncond_scale"]
    args.guidance_rescale = params["guidance_rescale"]
    args.frame_stride = [params["frame_stride"]]
    args.n_iter = params["n_iter"]
    args.exe_steps = params["exe_steps"]
    args.n_action_steps = params["n_action_steps"]
    args.video_length = params["video_length"]
    args.height = params["height"]
    args.width = params["width"]
    args.bs = params["bs"]
    args.save_fps = params["save_fps"]
    args.perframe_ae = bool(params["perframe_ae"])
    args.zero_pred_state = bool(params["zero_pred_state"])
    args.timestep_spacing = params["timestep_spacing"]
    args.seed = params["seed"]

    seed_everything(args.seed)

    # Create output dirs
    os.makedirs(args.savedir + '/inference', exist_ok=True)

    # Load CSV
    csv_path = os.path.join(args.prompt_dir, f"{args.dataset}.csv")
    df = pd.read_csv(csv_path)

    # -- Migrated inference loop (from original run_inference lines 677-914) --
    with torch.inference_mode():
        for idx in range(0, len(df)):
            sample = df.iloc[idx]

            init_frame_path = get_init_frame_path(args.prompt_dir, sample)
            ori_fps = float(sample['fps'])

            video_save_dir = args.savedir + f"/inference/sample_{sample['videoid']}"
            os.makedirs(video_save_dir, exist_ok=True)
            os.makedirs(video_save_dir + '/dm', exist_ok=True)
            os.makedirs(video_save_dir + '/wm', exist_ok=True)

            transition_path = get_transition_path(args.prompt_dir, sample)
            with h5py.File(transition_path, 'r') as h5f:
                transition_dict = {}
                for key in h5f.keys():
                    transition_dict[key] = torch.tensor(h5f[key][()])
                for key in h5f.attrs.keys():
                    transition_dict[key] = h5f.attrs[key]

            for fs in args.frame_stride:
                sample_save_dir = f'{video_save_dir}/dm/{fs}'
                os.makedirs(sample_save_dir, exist_ok=True)
                sample_save_dir = f'{video_save_dir}/wm/{fs}'
                os.makedirs(sample_save_dir, exist_ok=True)

                wm_video = []
                cond_obs_queues = {
                    "observation.images.top":
                    deque(maxlen=model.n_obs_steps_imagen),
                    "observation.state": deque(maxlen=model.n_obs_steps_imagen),
                    "action": deque(maxlen=args.video_length),
                }

                start_idx = 0
                model_input_fs = ori_fps // fs
                batch, ori_state_dim, ori_action_dim = prepare_init_input(
                    start_idx,
                    init_frame_path,
                    transition_dict,
                    fs,
                    data.test_datasets[args.dataset],
                    n_obs_steps=model.n_obs_steps_imagen)
                observation = {
                    'observation.images.top':
                    batch['observation.image'].permute(1, 0, 2,
                                                       3)[-1].unsqueeze(0),
                    'observation.state':
                    batch['observation.state'][-1].unsqueeze(0),
                    'action':
                    torch.zeros_like(batch['action'][-1]).unsqueeze(0)
                }
                observation = {
                    key: observation[key].to(device, non_blocking=True)
                    for key in observation
                }
                if model._use_channels_last:
                    for key, val in observation.items():
                        if val.dim() == 4:
                            observation[key] = val.to(memory_format=torch.channels_last)
                        elif val.dim() == 3:
                            observation[key] = val.unsqueeze(-1).to(memory_format=torch.channels_last).squeeze(-1)

                cond_obs_queues = populate_queues(cond_obs_queues, observation)

                # [WMA-HPC] Async save: overlap H.264 encoding with GPU
                from concurrent.futures import ThreadPoolExecutor
                _save_pool = ThreadPoolExecutor(max_workers=1)
                _save_futures = []

                for itr in tqdm(range(args.n_iter)):
                    # --- Decision-Making (dm) ---
                    torch.compiler.cudagraph_mark_step_begin()
                    print(f'>>> Step {itr}: generating actions ...')
                    observation = {
                        'observation.images.top':
                        torch.stack(list(
                            cond_obs_queues['observation.images.top']),
                                    dim=1).permute(0, 2, 1, 3, 4),
                        'observation.state':
                        torch.stack(list(
                            cond_obs_queues['observation.state']),
                                    dim=1),
                        'action':
                        torch.stack(list(cond_obs_queues['action']),
                                    dim=1),
                    }
                    observation = {
                        key: observation[key].to(device, non_blocking=True)
                        for key in observation
                    }
                    if model._use_channels_last:
                        for key, val in observation.items():
                            if val.dim() == 5:
                                observation[key] = val.to(
                                    memory_format=torch.channels_last_3d)
                            elif val.dim() == 4:
                                observation[key] = val.to(
                                    memory_format=torch.channels_last)
                            elif val.dim() == 3:
                                observation[key] = val.unsqueeze(-1).to(
                                    memory_format=torch.channels_last
                                ).squeeze(-1)

                    # [WMA-HPC] Pre-compute shared conditioning for dm+wm
                    # CLIP image embed, VAE encode, and state embed are the
                    # same for both dm and wm calls within one iteration
                    # (only action embedding differs)
                    _no_dedup = os.environ.get('WMA_NO_COND_DEDUP')
                    _cached_cond = None
                    if not _no_dedup:
                        _img = observation['observation.images.top'].permute(0, 2, 1, 3, 4)
                        _cond_img = rearrange(_img, 'b o c h w -> (b o) c h w')[-1:]
                        _cond_img_emb = model.embedder(_cond_img)
                        _cond_img_emb = model.image_proj_model(_cond_img_emb)
                        _img_cat_cond = None
                        if model.model.conditioning_key == 'hybrid':
                            _z = get_latent_z(model, _img.permute(0, 2, 1, 3, 4))
                            _img_cat_cond = _z[:, :, -1:, :, :]
                            _img_cat_cond = repeat(_img_cat_cond,
                                                  'b c t h w -> b c (repeat t) h w',
                                                  repeat=noise_shape[2])
                        _cond_state_emb = model.state_projector(observation['observation.state'])
                        _cond_state_emb = _cond_state_emb + model.agent_state_pos_emb

                        _cached_cond = {
                            'cond_img_emb': _cond_img_emb,
                            'img_cat_cond': _img_cat_cond,
                            'cond_state_emb': _cond_state_emb,
                        }

                    # [WMA-HPC] Extended BF16 decode for dm: video is discarded
                    # (skip_decode=True), so actions may tolerate more BF16 steps.
                    # dm_ddim_steps > 0: use as BF16 step threshold for dm
                    # dm_ddim_steps == -1: use 49 (all decode steps BF16)
                    if dm_ddim_steps != 0:
                        _dm_bf16_steps = 49 if dm_ddim_steps == -1 else dm_ddim_steps
                        model.model.diffusion_model._force_bf16_decode = _dm_bf16_steps
                    # [WMA-HPC] Reduced DM DDIM steps: fewer total steps for DM
                    # since only actions matter (video discarded via skip_decode=True).
                    # When using fewer steps, force all decode steps to BF16.
                    _dm_steps = dm_num_steps if dm_num_steps > 0 else args.ddim_steps
                    _dm_sampler = dm_ddim_sampler if dm_ddim_sampler is not None else ddim_sampler
                    if dm_num_steps > 0 and dm_ddim_steps == 0:
                        # Auto-enable all-BF16 for reduced-step DM
                        model.model.diffusion_model._force_bf16_decode = dm_num_steps
                    _, pred_actions, _, _, _ = \
                        image_guided_synthesis_sim_mode(
                            model, sample['instruction'], observation,
                            noise_shape,
                            action_cond_step=args.exe_steps,
                            ddim_steps=_dm_steps,
                            ddim_eta=args.ddim_eta,
                            unconditional_guidance_scale=args.unconditional_guidance_scale,
                            fs=model_input_fs,
                            timestep_spacing=args.timestep_spacing,
                            guidance_rescale=args.guidance_rescale,
                            sim_mode=False,
                            skip_decode=True,
                            ddim_sampler=_dm_sampler,
                            cached_cond=_cached_cond)
                    if dm_ddim_steps != 0 or dm_num_steps > 0:
                        model.model.diffusion_model._force_bf16_decode = 0

                    # Push predicted actions to queue (for wm observation)
                    for idx in range(len(pred_actions[0])):
                        obs_act = {'action': pred_actions[0][idx:idx + 1]}
                        obs_act['action'][:, ori_action_dim:] = 0.0
                        cond_obs_queues = populate_queues(
                            cond_obs_queues, obs_act)

                    # Build wm observation from queues
                    observation = {
                        'observation.images.top':
                        torch.stack(list(
                            cond_obs_queues['observation.images.top']),
                                    dim=1).permute(0, 2, 1, 3, 4),
                        'observation.state':
                        torch.stack(list(cond_obs_queues['observation.state']),
                                    dim=1),
                        'action':
                        torch.stack(list(cond_obs_queues['action']), dim=1),
                    }
                    observation = {
                        key: observation[key].to(device, non_blocking=True)
                        for key in observation
                    }

                    # --- World Model (wm) ---
                    torch.compiler.cudagraph_mark_step_begin()
                    print(f'>>> Step {itr}: interacting with world model ...')
                    use_gap2 = gap2 is not None
                    # [WMA-HPC] Use more BF16 steps for wm (video tolerates more
                    # BF16 than dm's action quality). wm_bf16_steps > 0 overrides.
                    if wm_bf16_steps > 0:
                        model.model.diffusion_model._force_bf16_decode = wm_bf16_steps
                    # [WMA-HPC] Reuse cached conditioning for wm call
                    # (same CLIP/VAE/state, only action changes)
                    pred_videos_1, _, pred_states, _, wm_samples = \
                        image_guided_synthesis_sim_mode(
                            model, "", observation, noise_shape,
                            action_cond_step=args.exe_steps,
                            ddim_steps=args.ddim_steps,
                            ddim_eta=args.ddim_eta,
                            unconditional_guidance_scale=args.unconditional_guidance_scale,
                            fs=model_input_fs,
                            text_input=False,
                            timestep_spacing=args.timestep_spacing,
                            guidance_rescale=args.guidance_rescale,
                            skip_decode=use_gap2,
                            ddim_sampler=ddim_sampler,
                            cached_cond=_cached_cond)
                    if wm_bf16_steps > 0:
                        model.model.diffusion_model._force_bf16_decode = 0

                    # Decode wm latents: CUDA graph or direct
                    if use_gap2:
                        pred_videos_1 = gap2.run(wm_samples)

                    # Update queues with decoded wm output (for next wm obs)
                    for idx in range(args.exe_steps):
                        obs_wm = {
                            'observation.images.top':
                            pred_videos_1[0][:, idx:idx + 1].permute(
                                1, 0, 2, 3),
                            'observation.state':
                            torch.zeros_like(pred_states[0][idx:idx + 1]) if
                            args.zero_pred_state else
                            pred_states[0][idx:idx + 1],
                            'action':
                            torch.zeros_like(pred_actions[0][-1:])
                        }
                        obs_wm['observation.state'][
                            :, ori_state_dim:] = 0.0
                        cond_obs_queues = populate_queues(
                            cond_obs_queues, obs_wm)

                    # [WMA-HPC] Async save: transfer to CPU then encode in background
                    sample_video_file = \
                        f'{video_save_dir}/wm/{fs}/itr-{itr}.mp4'
                    _vid_cpu = pred_videos_1.cpu()
                    _save_futures.append(
                        _save_pool.submit(save_results, _vid_cpu,
                                          sample_video_file, args.save_fps))

                    print('>' * 24)
                    wm_video.append(
                        pred_videos_1[:, :, :args.exe_steps].cpu())

                # [WMA-HPC] Wait for all background saves to complete
                for fut in _save_futures:
                    fut.result()
                _save_pool.shutdown(wait=False)

                full_video = torch.cat(wm_video, dim=2)
                sample_full_video_file = f"{video_save_dir}/../{sample['videoid']}_full_fs{fs}.mp4"
                save_results(full_video, sample_full_video_file, fps=args.save_fps)


# ========================================================================
# Daemon entry point
# ========================================================================

def get_daemon_parser():
    """Daemon startup args -- subset needed for model init."""
    parser = argparse.ArgumentParser(description="WMA Inference Daemon")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--perframe_ae", action='store_true', default=False)
    parser.add_argument("--foreground", action='store_true', default=False,
                        help="Run in foreground (no double-fork, logs to stdout). "
                             "Useful for development and debugging.")
    return parser


def _daemonize():
    """Double-fork to fully detach from parent process tree.

    This ensures tools like nsys won't wait for the daemon after the
    client exits, because the daemon becomes a child of init/systemd
    rather than the client.
    """
    # First fork: parent exits immediately, child continues
    pid = os.fork()
    if pid > 0:
        os._exit(0)  # Parent exits — client's Popen sees child terminate

    # New session leader (detach from terminal)
    os.setsid()

    # Second fork: session leader exits, grandchild can never acquire a tty
    pid = os.fork()
    if pid > 0:
        os._exit(0)

    # Now running as fully detached daemon (grandchild)


def main():
    parser = get_daemon_parser()
    args = parser.parse_args()

    foreground = args.foreground

    if not foreground:
        # 0. Double-fork to detach from parent (avoids nsys re-parent wait)
        _daemonize()

    # 1. Write PID file (use new PID after double-fork)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    if not foreground:
        # 2. Redirect stdout/stderr to log file
        # os.dup2 only changes the OS-level fd target; Python's sys.stdout
        # TextIOWrapper keeps its own (fully-buffered) buffer, so print()
        # output gets stuck in an 8 KB buffer and never reaches the log.
        # Fix: replace sys.stdout/stderr with line-buffered file objects
        # AND dup2 fd 1/2 so C-level writes (CUDA, tqdm) also go to the log.
        _log_out = open(LOG_FILE, "a", buffering=1)  # line-buffered
        _log_err = open(LOG_FILE, "a", buffering=1)  # separate fd for stderr
        os.dup2(_log_out.fileno(), 1)
        os.dup2(_log_err.fileno(), 2)
        sys.stdout = _log_out
        sys.stderr = _log_err

    print(f"[Daemon] PID={os.getpid()}, starting...", flush=True)

    # 3. Signal handler for graceful shutdown
    def _shutdown(signum, frame):
        print(f"[Daemon] Received signal {signum}, shutting down...", flush=True)
        cleanup_shm()
        try:
            os.unlink(PID_FILE)
        except FileNotFoundError:
            pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # 4. Load model (one-time, expensive)
    model, config, data, noise_shape, device, gap2, ddim_sampler, dm_ddim_steps, wm_bf16_steps, dm_num_steps, dm_ddim_sampler = init_model(args)

    # 5. Record daemon defaults for compatibility check
    daemon_defaults = {
        "config_path": os.path.abspath(args.config),
        "ckpt_path": os.path.abspath(args.ckpt_path),
        "height": args.height,
        "width": args.width,
    }

    # 6. Create SHM control block
    cleanup_shm()  # Remove stale SHM if exists
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
    # Daemon manages SHM lifecycle itself — don't let resource_tracker interfere
    from multiprocessing.resource_tracker import unregister as _rt_unregister
    _rt_unregister(f"/{SHM_NAME}", "shared_memory")
    shm.buf[:SHM_SIZE] = b"\x00" * SHM_SIZE  # Zero-fill
    write_state(shm, STATE_IDLE)

    print("[Daemon] Model loaded, entering listen loop.", flush=True)
    print("[Daemon] Ready.", flush=True)

    # 7. Listen loop
    try:
        while True:
            # Poll for REQUEST
            while read_state(shm) != STATE_REQUEST:
                time.sleep(0.05)

            params = read_ctrl(shm)
            print(f"[Daemon] Request received from PID={params['client_pid']}", flush=True)

            # Validate compatibility
            mismatches = []
            for key in ("config_path", "ckpt_path", "height", "width"):
                req_val = params[key]
                if key in ("config_path", "ckpt_path"):
                    req_val = os.path.abspath(req_val) if req_val else ""
                daemon_val = daemon_defaults[key]
                if isinstance(daemon_val, str):
                    if req_val != daemon_val:
                        mismatches.append(f"{key}({daemon_val} vs {req_val})")
                else:
                    if req_val != daemon_val:
                        mismatches.append(f"{key}({daemon_val} vs {req_val})")
            if mismatches:
                err = f"Param mismatch: {', '.join(mismatches)}. Restart daemon."
                print(f"[Daemon] ERROR: {err}", flush=True)
                write_ctrl(shm, state=STATE_ERROR, exit_code=1, error_msg=err)
                continue

            write_state(shm, STATE_RUNNING)

            try:
                run_inference_request(params, model, config, data, noise_shape, device, gap2, ddim_sampler,
                                     dm_ddim_steps, wm_bf16_steps, dm_num_steps, dm_ddim_sampler)
                write_ctrl(shm, state=STATE_DONE, exit_code=0)
                print("[Daemon] Request completed successfully.", flush=True)
            except Exception:
                tb = traceback.format_exc()
                print(f"[Daemon] ERROR:\n{tb}", flush=True)
                write_ctrl(shm, state=STATE_ERROR, exit_code=1,
                           error_msg=tb[:511])
    finally:
        shm.close()


if __name__ == "__main__":
    main()
