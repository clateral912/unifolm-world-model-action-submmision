import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        # =======OPTIMIZATION: CUDA GRAPH COMPATIBLE TIMESTEP EMBEDDING=======
        # Create tensor directly on target device to avoid .to() allocation
        # during CUDA graph capture (which forbids dynamic memory allocation).
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32,
                         device=timesteps.device) /
            half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def make_beta_schedule(schedule,
                       n_timestep,
                       linear_start=1e-4,
                       linear_end=2e-2,
                       cosine_s=8e-3):
    if schedule == "linear":
        betas = (torch.linspace(linear_start**0.5,
                                linear_end**0.5,
                                n_timestep,
                                dtype=torch.float64)**2)

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep +
            cosine_s)
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start,
                               linear_end,
                               n_timestep,
                               dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start,
                               linear_end,
                               n_timestep,
                               dtype=torch.float64)**0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


# =======OPTIMIZATION: ADVANCED TIMESTEP SCHEDULING (logSNR, dp_optimal)==========
# These scheduling methods place DDIM timesteps more intelligently than uniform
# spacing, concentrating steps in regions of high score-function curvature.
def _logsnr_from_alphas_cumprod(alphas_cumprod):
    """Compute logSNR = log(alpha_cumprod / (1 - alpha_cumprod)) for each timestep."""
    return np.log(alphas_cumprod / (1.0 - alphas_cumprod + 1e-12) + 1e-12)


def _make_logsnr_timesteps(num_ddim_timesteps, num_ddpm_timesteps, alphas_cumprod):
    """Generate timesteps uniformly spaced in logSNR space."""
    logsnr = _logsnr_from_alphas_cumprod(alphas_cumprod)
    logsnr_max = logsnr[0]
    logsnr_min = logsnr[-1]
    target_logsnr = np.linspace(logsnr_min, logsnr_max, num_ddim_timesteps)
    ddim_timesteps = np.zeros(num_ddim_timesteps, dtype=np.int64)
    for i, target in enumerate(target_logsnr):
        ddim_timesteps[i] = np.argmin(np.abs(logsnr - target))
    ddim_timesteps = np.unique(ddim_timesteps)
    if len(ddim_timesteps) < num_ddim_timesteps:
        all_ts = np.arange(num_ddpm_timesteps)
        remaining = np.setdiff1d(all_ts, ddim_timesteps)
        logsnr_remaining = logsnr[remaining]
        target_full = np.linspace(logsnr_min, logsnr_max, num_ddim_timesteps)
        for target in target_full:
            if len(ddim_timesteps) >= num_ddim_timesteps:
                break
            idx = np.argmin(np.abs(logsnr_remaining - target))
            candidate = remaining[idx]
            if candidate not in ddim_timesteps:
                ddim_timesteps = np.sort(np.append(ddim_timesteps, candidate))
    ddim_timesteps = ddim_timesteps[:num_ddim_timesteps]
    return ddim_timesteps


def _make_dp_optimal_timesteps(num_ddim_timesteps, num_ddpm_timesteps, alphas_cumprod):
    """Find optimal DDIM timestep subset via dynamic programming.

    Minimizes total curvature-weighted squared logSNR gap.
    """
    T = num_ddpm_timesteps
    M = num_ddim_timesteps
    if M >= T:
        return np.arange(T, dtype=np.int64)
    if M <= 2:
        return np.array([0, T - 1], dtype=np.int64)[:M]
    logsnr = _logsnr_from_alphas_cumprod(alphas_cumprod).astype(np.float64)
    d2 = np.zeros(T, dtype=np.float64)
    d2[1:-1] = np.abs(logsnr[2:] - 2.0 * logsnr[1:-1] + logsnr[:-2])
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    kernel_size = max(T // 50, 3)
    if kernel_size > 1:
        pad_l = kernel_size // 2
        pad_r = kernel_size - pad_l - 1
        padded = np.pad(d2, (pad_l, pad_r), mode='edge')
        d2 = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode='valid')[:T]
    d2 = d2 / (d2.mean() + 1e-12)
    d2_cumsum = np.concatenate(([0.0], np.cumsum(d2)))
    num_interior = M - 2
    if num_interior == 0:
        return np.array([0, T - 1], dtype=np.int64)

    def costs_from(a, b_arr):
        delta = logsnr[a] - logsnr[b_arr]
        avg_curv = (d2_cumsum[b_arr + 1] - d2_cumsum[a]) / np.maximum(b_arr - a + 1, 1)
        return delta ** 2 * (1.0 + avg_curv)

    INF = np.float64(1e30)
    dp_prev = np.full(T, INF, dtype=np.float64)
    parent = np.full((num_interior, T), -1, dtype=np.int64)
    for t in range(1, T - 1):
        dp_prev[t] = costs_from(0, np.array([t]))[0]
    for m in range(1, num_interior):
        dp_cur = np.full(T, INF, dtype=np.float64)
        for t in range(m + 1, T - 1):
            s_arr = np.arange(max(1, m), t)
            valid_mask = dp_prev[s_arr] < INF
            if not np.any(valid_mask):
                continue
            s_valid = s_arr[valid_mask]
            delta = logsnr[s_valid] - logsnr[t]
            avg_curv = (d2_cumsum[t + 1] - d2_cumsum[s_valid]) / np.maximum(t - s_valid + 1, 1)
            step_costs = delta ** 2 * (1.0 + avg_curv)
            total_costs = dp_prev[s_valid] + step_costs
            best_idx = np.argmin(total_costs)
            dp_cur[t] = total_costs[best_idx]
            parent[m][t] = s_valid[best_idx]
        dp_prev = dp_cur
    interior_range = np.arange(1, T - 1)
    valid = dp_prev[interior_range] < INF
    if not np.any(valid):
        return _make_logsnr_timesteps(M, T, alphas_cumprod)
    ir_valid = interior_range[valid]
    final_costs = dp_prev[ir_valid] + costs_from(ir_valid, np.full_like(ir_valid, T - 1))
    best_last = ir_valid[np.argmin(final_costs)]
    path = [best_last]
    for m in range(num_interior - 1, 0, -1):
        path.append(parent[m][path[-1]])
    path.reverse()
    ddim_timesteps = np.array([0] + path + [T - 1], dtype=np.int64)
    return ddim_timesteps
# =======END OPTIMIZATION=======


def make_ddim_timesteps(ddim_discr_method,
                        num_ddim_timesteps,
                        num_ddpm_timesteps,
                        verbose=True,
                        alphas_cumprod=None):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
        steps_out = ddim_timesteps + 1
    elif ddim_discr_method == 'uniform_trailing':
        c = num_ddpm_timesteps / num_ddim_timesteps
        ddim_timesteps = np.flip(np.round(np.arange(num_ddpm_timesteps, 0,
                                                    -c))).astype(np.int64)
        steps_out = ddim_timesteps - 1
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8),
                                       num_ddim_timesteps))**2).astype(int)
        steps_out = ddim_timesteps + 1
    # =======OPTIMIZATION: ADDITIONAL SCHEDULING METHODS=======
    elif ddim_discr_method == 'logSNR':
        assert alphas_cumprod is not None, \
            "logSNR schedule requires alphas_cumprod"
        if isinstance(alphas_cumprod, torch.Tensor):
            alphas_cumprod = alphas_cumprod.cpu().numpy()
        ddim_timesteps = _make_logsnr_timesteps(
            num_ddim_timesteps, num_ddpm_timesteps, alphas_cumprod)
        steps_out = ddim_timesteps + 1
    elif ddim_discr_method == 'dp_optimal':
        assert alphas_cumprod is not None, \
            "dp_optimal schedule requires alphas_cumprod"
        if isinstance(alphas_cumprod, torch.Tensor):
            alphas_cumprod = alphas_cumprod.cpu().numpy()
        ddim_timesteps = _make_dp_optimal_timesteps(
            num_ddim_timesteps, num_ddpm_timesteps, alphas_cumprod)
        steps_out = ddim_timesteps + 1
    # =======END OPTIMIZATION=======
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"'
        )

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    # steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums,
                                  ddim_timesteps,
                                  eta,
                                  verbose=True):
    # select alphas for computing the variance schedule
    # print(f'ddim_timesteps={ddim_timesteps}, len_alphacums={len(alphacums)}')
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] +
                             alphacums[ddim_timesteps[:-1]].tolist())

    # according the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(
            f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}'
        )
        print(
            f'For the chosen value of eta, which is {eta}, '
            f'this results in the following sigma_t schedule for ddim sampler {sigmas}'
        )
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)

    Args:
        betas (`numpy.ndarray`):
            the betas that the scheduler is being initialized with.

    Returns:
        `numpy.ndarray`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_bar_sqrt = np.sqrt(alphas_cumprod)

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].copy()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 -
                                            alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = np.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)),
                                   keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # Rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # Mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (
        1 - guidance_rescale) * noise_cfg
    return noise_cfg
