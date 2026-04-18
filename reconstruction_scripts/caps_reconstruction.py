"""
This module consolidates the custom TV denoisers and PnP-ADMM reconstruction
code.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.restoration import denoise_tv_chambolle as sk_denoise_tv_chambolle
from skimage.restoration import denoise_wavelet

from caps_tools import A_, A_t_

Array = np.ndarray


def _as_supported_float(image: Array) -> Array:
    """Convert an array to a stable floating dtype for numerical processing."""
    image = np.asarray(image)
    if image.dtype.kind != "f":
        image = img_as_float(image)
    if image.dtype == np.float16:
        return image.astype(np.float32, copy=False)
    if image.dtype in (np.float32, np.float64):
        return image
    return image.astype(np.float64, copy=False)


def _light_smooth(
    image: Array,
    mode: Optional[str] = None,
    sigma: float = 1.0,
    blend: float = 0.30,
    parity_strength: float = 0.75,
) -> Array:
    """Apply an optional light post-cleanup to a single 2D slice.

    Parameters
    ----------
    image
        Input 2D denoised slice.
    mode
        One of ``None``, ``'light_gaussian'``, ``'parity_cancel'``, or
        ``'gaussian_plus_parity'``.
    sigma
        Gaussian sigma for blur-based cleanup.
    blend
        Blend ratio in ``[0, 1]`` for the Gaussian-smoothed image.
    parity_strength
        Strength of global checkerboard cancellation.
    """
    if mode is None:
        return image

    if mode == "light_gaussian":
        low = gaussian_filter(image, sigma=sigma, mode="reflect")
        return (1.0 - blend) * image + blend * low

    if mode == "parity_cancel":
        h, w = image.shape
        yy, xx = np.indices((h, w))
        checker = ((yy + xx) % 2) * 2.0 - 1.0
        alpha = np.mean(image * checker)
        return image - parity_strength * alpha * checker

    if mode == "gaussian_plus_parity":
        low = gaussian_filter(image, sigma=sigma, mode="reflect")
        image2 = (1.0 - blend) * image + blend * low

        h, w = image2.shape
        yy, xx = np.indices((h, w))
        checker = ((yy + xx) % 2) * 2.0 - 1.0
        alpha = np.mean(image2 * checker)
        return image2 - parity_strength * alpha * checker

    raise ValueError(f"Unknown post-processing mode: {mode}")


def denoise_tv_chambolle_caps(
    image: Array,
    weight: float = 0.1,
    eps: float = 2.0e-4,
    max_num_iter: int = 200,
    *,
    channel_axis: int = -1,
    post_mode: Optional[str] = None,
    post_sigma: float = 1.0,
    post_blend: float = 0.30,
    post_parity_strength: float = 0.75,
) -> Array:
    """Apply slice-wise Chambolle TV denoising to a 3D CAPS stack.

    Each slice along ``channel_axis`` is denoised independently as a 2D image.
    This matches the original CAPS slice-wise denoising logic.
    """
    image = np.asarray(image)
    input_dtype = image.dtype
    image = _as_supported_float(image)

    if channel_axis is None:
        raise ValueError("For CAPS denoising, set channel_axis to the slice axis.")

    channel_axis = channel_axis % image.ndim
    moved = np.moveaxis(image, channel_axis, -1)
    out = np.empty_like(moved)

    for idx in range(moved.shape[-1]):
        denoised = sk_denoise_tv_chambolle(
            moved[..., idx],
            weight=weight,
            eps=eps,
            max_num_iter=max_num_iter,
            channel_axis=None,
        )
        out[..., idx] = _light_smooth(
            denoised,
            mode=post_mode,
            sigma=post_sigma,
            blend=post_blend,
            parity_strength=post_parity_strength,
        )

    out = np.moveaxis(out, -1, channel_axis)
    if input_dtype.kind == "f":
        out = out.astype(input_dtype, copy=False)
    return out


def _denoise_tv_chambolle_nd_h1(
    image: Array,
    weight: float = 0.1,
    eps: float = 2.0e-4,
    max_num_iter: int = 200,
    h1_weight: float = 0.0,
    h1_num_iter: int = 1,
) -> Array:
    """Custom Chambolle TV denoiser with optional internal H1-style diffusion."""
    ndim = image.ndim
    p = np.zeros((ndim,) + image.shape, dtype=image.dtype)
    g = np.zeros_like(p)
    d = np.zeros_like(image)
    iteration = 0

    while iteration < max_num_iter:
        if iteration > 0:
            d = -p.sum(0)
            slices_d = [slice(None)] * ndim
            slices_p = [slice(None)] * (ndim + 1)

            for axis in range(ndim):
                slices_d[axis] = slice(1, None)
                slices_p[axis + 1] = slice(0, -1)
                slices_p[0] = axis
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[axis] = slice(None)
                slices_p[axis + 1] = slice(None)

            out = image + d
        else:
            out = image.copy()

        if h1_weight > 0 and h1_num_iter > 0:
            alpha = min(float(h1_weight), 0.24)
            for _ in range(h1_num_iter):
                if ndim == 2:
                    padded = np.pad(out, 1, mode="reflect")
                    center = padded[1:-1, 1:-1]
                    lap = (
                        padded[:-2, 1:-1]
                        + padded[2:, 1:-1]
                        + padded[1:-1, :-2]
                        + padded[1:-1, 2:]
                        - 4.0 * center
                    )
                    out = out + alpha * lap
                else:
                    lap = np.zeros_like(out)
                    for axis in range(ndim):
                        pad_width = [(0, 0)] * ndim
                        pad_width[axis] = (1, 1)
                        padded = np.pad(out, pad_width, mode="reflect")

                        center = [slice(1, -1)] * ndim
                        minus = [slice(1, -1)] * ndim
                        plus = [slice(1, -1)] * ndim
                        minus[axis] = slice(0, -2)
                        plus[axis] = slice(2, None)

                        lap += (
                            padded[tuple(minus)]
                            + padded[tuple(plus)]
                            - 2.0 * padded[tuple(center)]
                        )
                    out = out + alpha * lap

        energy = (d**2).sum()

        slices_g = [slice(None)] * (ndim + 1)
        for axis in range(ndim):
            slices_g[axis + 1] = slice(0, -1)
            slices_g[0] = axis
            g[tuple(slices_g)] = np.diff(out, axis=axis)
            slices_g[axis + 1] = slice(None)

        norm = np.sqrt((g**2).sum(axis=0))[np.newaxis, ...]
        energy += weight * norm.sum()

        if h1_weight > 0:
            energy += 0.5 * h1_weight * (g**2).sum()

        tau = 1.0 / (2.0 * ndim)
        norm *= tau / weight
        norm += 1.0
        p -= tau * g
        p /= norm

        energy /= float(image.size)
        if iteration == 0:
            energy_init = energy
            energy_previous = energy
        else:
            if np.abs(energy_previous - energy) < eps * energy_init:
                break
            energy_previous = energy

        iteration += 1

    return out


def sk_denoise_tv_chambolle_h1(
    image: Array,
    weight: float = 0.1,
    eps: float = 2.0e-4,
    max_num_iter: int = 200,
    *,
    channel_axis: Optional[int] = None,
    h1_weight: float = 0.0,
    h1_num_iter: int = 1,
) -> Array:
    """Apply the custom H1-regularized Chambolle denoiser."""
    input_dtype = np.asarray(image).dtype
    image = _as_supported_float(image)

    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        moved = np.moveaxis(image, channel_axis, -1)
        out = np.empty_like(moved)

        for idx in range(moved.shape[-1]):
            out[..., idx] = _denoise_tv_chambolle_nd_h1(
                moved[..., idx],
                weight=weight,
                eps=eps,
                max_num_iter=max_num_iter,
                h1_weight=h1_weight,
                h1_num_iter=h1_num_iter,
            )

        out = np.moveaxis(out, -1, channel_axis)
    else:
        out = _denoise_tv_chambolle_nd_h1(
            image,
            weight=weight,
            eps=eps,
            max_num_iter=max_num_iter,
            h1_weight=h1_weight,
            h1_num_iter=h1_num_iter,
        )

    if input_dtype.kind == "f":
        out = out.astype(input_dtype, copy=False)
    return out


def _shift_up(array: Array) -> Array:
    """Shift an array upward with Neumann boundary handling."""
    return np.concatenate([array[:1, ...], array[:-1, ...]], axis=0)


def _shift_left(array: Array) -> Array:
    """Shift an array leftward with Neumann boundary handling."""
    return np.concatenate([array[:, :1, ...], array[:, :-1, ...]], axis=1)


def _grad(image: Array) -> Tuple[Array, Array]:
    """Forward finite differences with replicated boundaries."""
    grad_x = np.concatenate([image[:, 1:, ...], image[:, -1:, ...]], axis=1) - image
    grad_y = np.concatenate([image[1:, ...], image[-1:, ...]], axis=0) - image
    return grad_x, grad_y


def _div(px: Array, py: Array) -> Array:
    """Adjoint divergence operator for ``_grad``."""
    return (px - _shift_left(px)) + (py - _shift_up(py))


def _shrink2(
    dx: Array,
    dy: Array,
    thresh: float,
    vectorial: bool = True,
    eps: float = 1e-12,
) -> Tuple[Array, Array]:
    """Isotropic shrinkage for split-Bregman TV denoising."""
    if dx.ndim == 3:
        if vectorial:
            norm = np.sqrt((dx**2 + dy**2).sum(axis=2))
            scale = np.maximum(0.0, 1.0 - thresh / np.maximum(norm, eps))
            scale = scale[..., None]
        else:
            norm = np.sqrt(dx**2 + dy**2)
            scale = np.maximum(0.0, 1.0 - thresh / np.maximum(norm, eps))
    else:
        norm = np.sqrt(dx**2 + dy**2)
        scale = np.maximum(0.0, 1.0 - thresh / np.maximum(norm, eps))
    return dx * scale, dy * scale


def _laplacian_spectrum(height: int, width: int) -> Array:
    """Return the Fourier spectrum of the 2D discrete Laplacian kernel."""
    kernel = np.zeros((height, width), dtype=np.float32)
    kernel[0, 0] = -4.0
    kernel[0, 1] = 1.0
    kernel[0, -1] = 1.0
    kernel[1, 0] = 1.0
    kernel[-1, 0] = 1.0
    return np.fft.fft2(kernel).real


def tvdenoise_split_bregman(
    image: Array,
    lam: float = 20.0,
    max_iter: int = 1000,
    tol: float = 1e-3,
    gamma1: float = 5.0,
    vectorial: bool = True,
    channel_axis: Optional[int] = None,
    dtype_out: Optional[np.dtype] = None,
) -> Array:
    """Denoise an image using split-Bregman TV with an L2 fidelity term.

    Smaller ``lam`` yields stronger denoising, matching the original MATLAB-style
    convention used in the user code.
    """
    image = np.asarray(image)
    if channel_axis is not None:
        image = np.moveaxis(image, channel_axis, -1)

    image_work = image.astype(np.float32, copy=False) if image.dtype.kind != "f" else image.copy()

    height, width = image_work.shape[:2]
    channels = 1 if image_work.ndim == 2 else image_work.shape[2]

    u = image_work.copy()
    dx = np.zeros_like(u)
    dy = np.zeros_like(u)
    b1x = np.zeros_like(u)
    b1y = np.zeros_like(u)

    lap_hat = _laplacian_spectrum(height, width)
    alpha = lam / gamma1
    denom = alpha - lap_hat
    inv_denom = 1.0 / np.maximum(denom, 1e-12)
    if channels > 1:
        inv_denom = inv_denom[..., None]

    tol_scaled = tol * np.linalg.norm(image_work.reshape(-1), ord=2)

    for iteration in range(int(max_iter)):
        u_last = u

        div_term = _div(dx - b1x, dy - b1y)
        rhs = alpha * image_work - div_term

        if channels == 1:
            u = np.real(ifft2(inv_denom * fft2(rhs)))
        else:
            u_hat = fft2(rhs, axes=(0, 1))
            u = np.real(ifft2(inv_denom * u_hat, axes=(0, 1)))

        ux, uy = _grad(u)
        temp_x = ux + b1x
        temp_y = uy + b1y
        dx, dy = _shrink2(temp_x, temp_y, 1.0 / gamma1, vectorial=vectorial)

        b1x = temp_x - dx
        b1y = temp_y - dy

        if iteration >= 2:
            diff = np.linalg.norm((u - u_last).reshape(-1), ord=2)
            if diff <= tol_scaled:
                break

    if channel_axis is not None:
        u = np.moveaxis(u, -1, channel_axis)

    if dtype_out is not None:
        u = u.astype(dtype_out, copy=False)
    return u


def _normalize_for_metrics(reference: Array, estimate: Array) -> Tuple[Array, Array]:
    """Normalize volumes for PSNR/SSIM evaluation with zero-safe scaling."""
    ref_max = np.max(reference)
    est_max = np.max(estimate)
    ref_norm = reference if ref_max == 0 else reference / ref_max
    est_norm = estimate if est_max == 0 else estimate / est_max
    return ref_norm, est_norm


def _apply_denoiser(
    image: Array,
    denoiser: str,
    tv_weight: float,
    tv_iter_max: int,
    sigma: Optional[float],
    multichannel: bool,
) -> Array:
    """Dispatch the denoising step used in the PnP-ADMM loop."""
    denoiser_name = denoiser.lower()

    if denoiser_name == "tv":
        return sk_denoise_tv_chambolle_h1(
            image,
            weight=tv_weight,
            eps=2e-4,
            max_num_iter=tv_iter_max,
            channel_axis=-1,
            h1_weight=1e-1,
            h1_num_iter=1,
        )

    if denoiser_name == "wavelet":
        channel_axis = -1 if multichannel else None
        return denoise_wavelet(
            image,
            sigma=0.01 if sigma is None else sigma,
            channel_axis=channel_axis,
        )

    raise ValueError(f"Unsupported denoiser: {denoiser}")


def admm_optimize(
    kf: int,
    y: Array,
    mask: Array,
    resultsdir: Optional[str],
    orig_name: str,
    mask_name: str,
    _lambda: float = 1.0,
    rho: float = 10.0,
    denoiser: str = "tv",
    iter_max: int = 50,
    noise_estimate: bool = False,
    tv_weight: float = 0.0,
    sigma: Optional[float] = None,
    tv_iter_max: int = 5,
    multichannel: bool = True,
    x0: Optional[Array] = None,
    model: Optional[object] = None,
    X_orig: Optional[Array] = None,
    show_iqa: bool = True,
    clip_range: Tuple[float, float] = (0.0, 2.0),
    verbose: bool = False,
) -> Tuple[Array, List[float], List[float]]:
    """Run PnP-ADMM reconstruction for one coded frame block.

    Legacy arguments such as ``resultsdir``, ``orig_name``, ``mask_name``,
    ``_lambda``, ``noise_estimate``, and ``model`` are retained for backward
    compatibility with the existing calling code.
    """
    del kf, resultsdir, orig_name, mask_name, _lambda, noise_estimate, model

    phi_sum = np.sum(np.square(mask), axis=2)
    phi_sum[phi_sum == 0] = 1.0

    forward = lambda x: A_(x, mask)
    adjoint = lambda data: A_t_(data, mask)

    if x0 is None:
        x0 = adjoint(y)

    x = x0.copy()
    theta = x0.copy()
    b = np.zeros_like(x0)
    psnr_history: List[float] = []
    ssim_history: List[float] = []

    for iteration in range(iter_max):
        yb = forward(theta - b)
        x = (theta - b) + adjoint((y - yb) / (phi_sum + rho))

        theta = _apply_denoiser(
            x + b,
            denoiser=denoiser,
            tv_weight=tv_weight,
            tv_iter_max=tv_iter_max,
            sigma=sigma,
            multichannel=multichannel,
        )

        theta = np.clip(theta, clip_range[0], clip_range[1])
        b = b + (x - theta)

        if show_iqa and X_orig is not None:
            x_ref, x_est = _normalize_for_metrics(X_orig, x)
            psnr_history.append(compare_psnr(x_ref, x_est, data_range=1.0))
            ssim_history.append(compare_ssim(x_ref, x_est, data_range=1.0))
        elif verbose and (iteration + 1) % 5 == 0:
            print(f"  ADMM-{denoiser.upper()} iteration {iteration + 1:3d}.")

    return x, psnr_history, ssim_history


def pnp_admm(
    meas: Array,
    mask: Array,
    resultsdir: Optional[str],
    orig_name: str,
    mask_name: str,
    v0: Optional[Array] = None,
    orig: Optional[Array] = None,
    ini_frame: int = 0,
    num_frame: int = 1,
    orig_inten: float = 10000.0,
    maskdirection: str = "plain",
    **args,
) -> Tuple[Array, float, List[List[float]], List[List[float]]]:
    """Reconstruct one or more coded frame blocks with PnP-ADMM."""
    del orig_inten

    num_row, num_col, num_mask = mask.shape
    recon = np.zeros((num_row, num_col, num_mask * num_frame), dtype=np.float32)
    psnr_all: List[List[float]] = []
    ssim_all: List[List[float]] = []
    begin_time = time.time()

    direction = maskdirection.lower()
    if direction not in {"plain", "updown", "downup"}:
        raise ValueError("maskdirection must be 'plain', 'updown', or 'downup'.")

    for kf in range(num_frame):
        if orig is not None:
            start = (kf + ini_frame) * num_mask
            stop = (kf + ini_frame + 1) * num_mask
            orig_k = orig[:, :, start:stop]
        else:
            orig_k = None

        meas_k = meas[:, :, kf + ini_frame]

        if v0 is None:
            v0_k = None
        else:
            v0_k = v0[:, :, kf * num_mask : (kf + 1) * num_mask]
            reverse = (direction == "updown" and (kf + ini_frame) % 2 == 1) or (
                direction == "downup" and (kf + ini_frame) % 2 == 0
            )
            if reverse:
                v0_k = v0_k[:, :, ::-1]

        x_k, psnr_k, ssim_k = admm_optimize(
            kf,
            meas_k,
            mask,
            resultsdir,
            orig_name,
            mask_name,
            x0=v0_k,
            X_orig=orig_k,
            **args,
        )

        reverse = (direction == "updown" and (kf + ini_frame) % 2 == 1) or (
            direction == "downup" and (kf + ini_frame) % 2 == 0
        )
        if reverse:
            x_k = x_k[:, :, ::-1]
            psnr_k = psnr_k[::-1]
            ssim_k = ssim_k[::-1]

        recon[:, :, kf * num_mask : (kf + 1) * num_mask] = x_k
        psnr_all.append(psnr_k)
        ssim_all.append(ssim_k)

    elapsed = time.time() - begin_time
    return recon, elapsed, psnr_all, ssim_all


def process_one_frame(
    kf: int,
    meas: Array,
    mask: Array,
    orig_k: Optional[Array],
    ini_frame: int,
    orig_inten: float,
    denoiser: str,
    iter_num: int,
    tv_weight: float,
    rho: float,
    tv_iter_max: int = 5,
) -> Tuple[int, Array, float, List[List[float]], List[List[float]]]:
    """Reconstruct a single coded frame block using ``pnp_admm``.

    This helper keeps the original return signature so it can still be used in
    a parallel processing workflow.
    """
    meas_k = meas[:, :, ini_frame + kf : ini_frame + kf + 1]
    x_k, run_time_k, psnr_k, ssim_k = pnp_admm(
        meas_k,
        mask,
        resultsdir=None,
        orig_name="",
        mask_name="",
        v0=None,
        orig=orig_k,
        ini_frame=0,
        num_frame=1,
        orig_inten=orig_inten,
        maskdirection="plain",
        _lambda=tv_weight * rho,
        rho=rho,
        denoiser=denoiser,
        iter_max=iter_num,
        tv_weight=tv_weight,
        tv_iter_max=tv_iter_max,
    )
    return kf, x_k, run_time_k, psnr_k, ssim_k


def calculate_psnr(original: Array, reconstructed: Array, max_value: float) -> float:
    """Compute PSNR from a fixed maximum intensity value."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(max_value**2 / mse)


def calculate_ssim(
    original: Array,
    reconstructed: Array,
    L: float = 65536,
    K1: float = 0.01,
    K2: float = 0.03,
    sigma: float | Sequence[float] = 1.5,
) -> float:
    """Compute SSIM using Gaussian local statistics."""
    sigma_value = sigma if isinstance(sigma, (list, tuple)) else (sigma,) * original.ndim
    filter_args = {"sigma": sigma_value, "truncate": 3.5, "mode": "reflect"}

    mean_orig = gaussian_filter(original, **filter_args)
    mean_recon = gaussian_filter(reconstructed, **filter_args)
    var_orig = gaussian_filter(original**2, **filter_args) - mean_orig**2
    var_recon = gaussian_filter(reconstructed**2, **filter_args) - mean_recon**2
    cov = gaussian_filter(original * reconstructed, **filter_args) - mean_orig * mean_recon

    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    numerator = (2 * mean_orig * mean_recon + c1) * (2 * cov + c2)
    denominator = (mean_orig**2 + mean_recon**2 + c1) * (var_orig + var_recon + c2)
    return float(np.mean(numerator / denominator))


__all__ = [
    "admm_optimize",
    "calculate_psnr",
    "calculate_ssim",
    "denoise_tv_chambolle_caps",
    "pnp_admm",
    "process_one_frame",
    "sk_denoise_tv_chambolle_h1",
    "tvdenoise_split_bregman",
]
