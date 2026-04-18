"""
This module provides helper routines for TIFF I/O, forward/adjoint operators,
and saving reconstruction outputs for CAPS workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import tifffile
from PIL import Image

ArrayLike = Union[np.ndarray, Sequence[np.ndarray]]
PathLike = Union[str, Path]


__all__ = [
    "read_tif",
    "save_as_bigtiff",
    "save_as_tif",
    "save_tiff_with_spacing",
    "A_",
    "A_t_",
    "finalize_result",
]


def _ensure_directory(path: PathLike) -> Path:
    """Create a directory if needed and return it as a ``Path`` object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stack_to_zyx(images: ArrayLike) -> np.ndarray:
    """Convert a stack to ImageJ-friendly ``(Z, Y, X)`` order.

    Accepted inputs
    ---------------
    - list/tuple of 2D arrays, interpreted as sequential slices
    - 3D array in ``(Y, X, Z)`` order
    - 3D array already in ``(Z, Y, X)`` order
    """
    arr = np.asarray(images)

    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            raise ValueError("images must not be empty")
        if np.asarray(images[0]).ndim != 2:
            raise ValueError("Expected a sequence of 2D slices")
        return np.stack([np.asarray(im) for im in images], axis=0)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D data, got shape {arr.shape}")

    # Heuristic: if the last axis is smallest, treat input as (Y, X, Z).
    if arr.shape[-1] <= min(arr.shape[0], arr.shape[1]):
        return np.moveaxis(arr, -1, 0)

    return arr


def read_tif(path: PathLike) -> np.ndarray:
    """Read a TIFF stack and return it in ``(Y, X, Z)`` order."""
    stack = tifffile.imread(path)
    stack = np.asarray(stack)

    if stack.ndim == 2:
        return stack[..., np.newaxis]
    if stack.ndim != 3:
        raise ValueError(f"Expected a 2D image or 3D TIFF stack, got {stack.ndim}D data")

    return np.moveaxis(stack, 0, -1)


def save_as_bigtiff(images: ArrayLike, resultsdir: PathLike, save_name: str) -> Path:
    """Save a stack as a compressed BigTIFF file.

    Parameters
    ----------
    images
        Image stack provided as a sequence of 2D slices or a 3D array.
    resultsdir
        Output directory.
    save_name
        File stem without extension.
    """
    output_dir = _ensure_directory(resultsdir)
    output_path = output_dir / f"{save_name}.tif"
    stack_zyx = _stack_to_zyx(images)

    tifffile.imwrite(
        output_path,
        stack_zyx,
        bigtiff=True,
        compression="deflate",
        metadata={"axes": "ZYX"},
        imagej=True,
    )
    return output_path


def save_as_tif(images: ArrayLike, resultsdir: PathLike, save_name: str) -> Path:
    """Save a 3D stack as an ImageJ-compatible TIFF in ``ZYX`` order."""
    output_dir = _ensure_directory(resultsdir)
    output_path = output_dir / f"{save_name}.tif"
    stack_zyx = _stack_to_zyx(images)

    tifffile.imwrite(
        output_path,
        stack_zyx,
        imagej=True,
        metadata={"axes": "ZYX"},
        compression="deflate",
    )
    return output_path


def save_tiff_with_spacing(
    save_name: PathLike,
    images: ArrayLike,
    x: float,
    y: float,
    z: float,
) -> Path:
    """Save a TIFF stack with voxel spacing metadata.

    Parameters
    ----------
    save_name
        Full output path including extension.
    images
        Image stack provided as a sequence of 2D slices or a 3D array.
    x, y, z
        Voxel sizes in micrometers.
    """
    output_path = Path(save_name)
    _ensure_directory(output_path.parent)

    stack_zyx = _stack_to_zyx(images).astype(np.uint16, copy=False)
    metadata = {
        "spacing": z,
        "unit": "um",
        "axes": "ZYX",
    }
    resolution = (1.0 / x, 1.0 / y)

    tifffile.imwrite(
        output_path,
        stack_zyx,
        imagej=True,
        metadata=metadata,
        compression="deflate",
        resolution=resolution,
    )
    return output_path


def A_(x: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Forward measurement operator for CAPS/CS-LSM.

    Multiple axial planes are encoded into a single measurement by summing the
    element-wise product between the volume ``x`` and mask stack ``phi`` over
    the axial dimension.
    """
    x = np.asarray(x)
    phi = np.asarray(phi)
    if x.shape != phi.shape:
        raise ValueError(f"x and phi must have the same shape, got {x.shape} and {phi.shape}")
    return np.sum(x * phi, axis=2)


def A_t_(y: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Adjoint of the CAPS/CS-LSM forward measurement operator."""
    y = np.asarray(y)
    phi = np.asarray(phi)
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got shape {y.shape}")
    if phi.ndim != 3:
        raise ValueError(f"phi must be 3D, got shape {phi.shape}")
    if y.shape != phi.shape[:2]:
        raise ValueError(
            f"Spatial dimensions of y must match phi, got {y.shape} and {phi.shape[:2]}"
        )
    return np.repeat(y[:, :, np.newaxis], phi.shape[2], axis=2) * phi


def finalize_result(
    result_3d: np.ndarray,
    runtime: Optional[float],
    perframe_psnr,
    perframe_ssim,
    orig,
    Cr: int,
    resultsdir: PathLike,
    save_name: str,
    iframe: int = 0,
    nframe: int = 1,
    orig_inten: float = 255,
    show_res_flag: int = 1,
    save_res_flag: int = 1,
    **kwargs,
) -> Optional[Path]:
    """Save a reconstruction stack as a 16-bit TIFF.

    The signature is kept compatible with the legacy pipeline, although only
    the saving branch is used in the cleaned implementation.
    """
    del runtime, perframe_psnr, perframe_ssim, orig, Cr, iframe, nframe, show_res_flag, kwargs

    if not save_res_flag:
        return None

    output_dir = _ensure_directory(resultsdir)
    output_path = output_dir / f"{save_name}.tif"

    result_3d = np.asarray(result_3d)
    data_range = result_3d.max() - result_3d.min()
    if data_range == 0:
        image_data = np.zeros_like(result_3d, dtype=np.uint16)
    else:
        scaled = orig_inten * (result_3d - result_3d.min()) / data_range
        image_data = scaled.astype(np.uint16)

    images = [Image.fromarray(image_data[:, :, i], mode="I;16") for i in range(image_data.shape[2])]
    images[0].save(
        str(output_path),
        save_all=True,
        append_images=images[1:],
        compression="tiff_deflate",
    )

    print(f"Results TIFF saved to: {output_path}\n")
    return output_path
