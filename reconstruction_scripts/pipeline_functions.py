"""
Utility functions for the post-reconstruction CAPS processing pipeline
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff
from scipy.interpolate import interp1d
from scipy.ndimage import zoom


# -----------------------------------------------------------------------------
# Rolling-shutter simulation
# -----------------------------------------------------------------------------

def create_triangle(profile: np.ndarray, start_index: int, width: int) -> None:
    """Write a triangular peak into a 1D profile in place."""
    if width <= 0:
        return

    for i in range(width):
        if i <= width // 2:
            profile[start_index + i] = (i + 1) / (width // 2 + 1)
        else:
            profile[start_index + i] = (width - i) / (width // 2 + 1)


def create_trapezoid(
    profile: np.ndarray,
    start_index: int,
    width: int,
    peak_width: int,
) -> None:
    """Write a trapezoid into a 1D profile in place."""
    if width <= 0 or peak_width <= 0:
        return

    end_index = start_index + width
    half_peak = peak_width // 2

    for i in range(half_peak):
        profile[start_index + i] = (i + 1) / (half_peak + 1)

    for i in range(half_peak):
        profile[end_index - half_peak + i] = (half_peak - i) / (half_peak + 1)

    profile[start_index + half_peak : end_index - half_peak] = 1.0


def create_line_profiles(
    trapezoid_widths: list[int],
    profile_length: int = 404,
) -> list[np.ndarray]:
    """
    Create rolling-shutter line profiles.

    The first width defines the triangle peak width. The remaining widths define
    trapezoids. The final profile is all ones.
    """
    if len(trapezoid_widths) == 0:
        raise ValueError("trapezoid_widths must contain at least one value.")

    line_profiles: list[np.ndarray] = []

    peak_width = trapezoid_widths[0]
    trapezoid_w_list = trapezoid_widths[1:]

    line_profile_1 = np.zeros(profile_length, dtype=np.float32)
    start_tri = (profile_length - peak_width) // 2
    create_triangle(line_profile_1, start_tri, peak_width)
    line_profiles.append(line_profile_1)

    for width in trapezoid_w_list:
        line_profile = np.zeros(profile_length, dtype=np.float32)
        start_index = (profile_length - width) // 2
        create_trapezoid(line_profile, start_index, width, peak_width)
        line_profiles.append(line_profile)

    line_profiles.append(np.ones(profile_length, dtype=np.float32))
    return line_profiles


def generate_rolling_shutter_stack(CR: int, realH: int, realW: int) -> np.ndarray:
    """
    Generate an in-memory rolling-shutter simulation stack.

    Returns
    -------
    ndarray
        Shape (num_profiles, realH, realW), dtype float32.
    """
    known_trapezoids = {
        10: [40, 80, 120, 160, 200, 240, 280, 320, 360],
        15: [26, 54, 82, 109, 136, 163, 191, 218, 246, 274, 301, 328, 356, 382],
        20: [20, 40, 60, 81, 102, 122, 144, 164, 184, 204, 226, 246, 266, 286,
             308, 328, 348, 368, 390],
        30: [14, 28, 40, 54, 68, 82, 96, 110, 122, 136, 150, 164, 178, 192, 204,
             218, 232, 246, 260, 274, 286, 300, 314, 328, 342, 356, 368, 382, 396],
    }

    if CR not in known_trapezoids:
        raise ValueError(f"No trapezoid widths defined for CR={CR}.")

    line_profiles = create_line_profiles(
        known_trapezoids[CR],
        profile_length=realH,
    )
    num_profiles = len(line_profiles)

    rolling_stack = np.zeros((num_profiles, realH, realW), dtype=np.float32)
    for i, line_prof in enumerate(line_profiles):
        rolling_stack[i] = np.tile(line_prof.reshape(realH, 1), (1, realW))

    return rolling_stack


# -----------------------------------------------------------------------------
# Rolling-shutter calibration
# -----------------------------------------------------------------------------

def normalize_stack(stack: np.ndarray) -> np.ndarray:
    """Normalize a stack to [0, 1]."""
    stack = np.asarray(stack, dtype=np.float32)
    min_val = float(np.min(stack))
    max_val = float(np.max(stack))
    return (stack - min_val) / (max_val - min_val + 1e-12)


def compute_mask_coding(calibrated_rolling_normalized: np.ndarray) -> np.ndarray:
    """
    Convert a cumulative rolling-shutter stack into slice-wise coding masks.
    """
    rolling = np.asarray(calibrated_rolling_normalized, dtype=np.float32)
    mask_coding = np.zeros_like(rolling, dtype=np.float32)

    mask_coding[0] = rolling[0]
    for i in range(1, rolling.shape[0]):
        diff_slice = rolling[i] - rolling[i - 1]
        diff_slice[diff_slice < 0] = 0

        mn = float(diff_slice.min())
        mx = float(diff_slice.max())
        if mx > mn:
            diff_slice = (diff_slice - mn) / (mx - mn + 1e-12)

        mask_coding[i] = diff_slice

    return mask_coding


def generate_calibrated_result(
    result_stack: np.ndarray,
    mask_coding: np.ndarray,
) -> np.ndarray:
    """
    Generate the calibrated output stack using the rolling-shutter coding masks.
    """
    result_stack = np.asarray(result_stack, dtype=np.float32)
    mask_coding = np.asarray(mask_coding, dtype=np.float32)

    num_masks = mask_coding.shape[0]
    num_results = result_stack.shape[0]
    num_outputs = num_results - num_masks

    if num_outputs <= 0:
        raise ValueError(
            "result_stack must contain more slices than the number of masks."
        )

    calibrated = np.zeros((num_outputs, *result_stack.shape[1:]), dtype=np.float32)

    for i in range(num_outputs):
        if (i + 1) % 1000 == 0:
            print(f"  Calibrating slice {i + 1}/{num_outputs} ...")

        idx = i % num_masks
        mask_sum_1 = np.sum(mask_coding[: idx + 1], axis=0)
        mask_sum_2 = np.sum(mask_coding[idx + 1 :], axis=0)

        calibrated[i] = (
            result_stack[i + num_masks] * mask_sum_1
            + result_stack[i] * mask_sum_2
        )

    return np.clip(calibrated, 0, 65535).astype(np.uint16)


def rolling_cali_result_in_memory(
    rolling_stack: np.ndarray,
    result_stack: np.ndarray,
) -> np.ndarray:
    """Run rolling-shutter calibration entirely in memory."""
    rolling_norm = normalize_stack(rolling_stack)
    mask_coding = compute_mask_coding(rolling_norm)
    return generate_calibrated_result(result_stack, mask_coding)


# -----------------------------------------------------------------------------
# Reslicing
# -----------------------------------------------------------------------------

def reslice_camera_scanning_mismatch(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Reslice a 3D stack along z to address camera-scanning mismatch.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    image = tiff.imread(str(input_path))
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {image.shape}.")

    n_frames, height, width = image.shape
    print(
        "[reslice_camera_scanning_mismatch] Original: "
        f"{n_frames} x {height} x {width}"
    )

    target_depth = int(0.00199761 * 500 * n_frames)
    print(f"Reslicing from {n_frames} to {target_depth} slices.")

    zoom_factors = (target_depth / n_frames, 1, 1)
    rescaled_image = zoom(image, zoom_factors, order=1).astype(np.uint16)

    tiff.imwrite(str(output_path), rescaled_image)
    print(f"[reslice_camera_scanning_mismatch] Saved: {output_path}")

    with tiff.TiffFile(str(output_path)) as tif_out:
        shape = tif_out.series[0].shape
        dtype = tif_out.series[0].dtype
    print(f"Resliced shape: {shape} — dtype: {dtype}")


# -----------------------------------------------------------------------------
# Sine-based z interpolation
# -----------------------------------------------------------------------------

def generate_sin_diff_weights(N: int) -> np.ndarray:
    """Build brightness-correction weights for a half-cycle sine exposure."""
    step = math.pi / N
    z = math.pi / 2 + np.arange(N + 1) * step
    dy = np.abs(np.sin(z[1:]) - np.sin(z[:-1]))
    dy /= dy.max() if dy.max() != 0 else 1.0
    return dy


def sine_based_z_interpolation(
    folder: str | Path,
    start_frame: int,
    input_image_name: str,
    output_image_name: str,
    *,
    CR: int = 15,
    n_period: int = 1,
    batch_size: Optional[int] = None,
    interp: int = 2,
) -> None:
    """
    Stream-interpolate a 3D TIFF along z without building the full output in RAM.
    """
    if batch_size is None:
        batch_size = CR * 10

    folder = Path(folder)
    in_path = folder / input_image_name
    out_path = folder / output_image_name

    with tiff.TiffFile(str(in_path)) as tif:
        vol = tif.asarray()[start_frame:].astype(np.float32)

    pts_per_period = 10 * CR
    t = np.linspace(0, 0.02 * n_period, pts_per_period + 1)

    freq = 50
    amp = 0.5
    phase = -np.pi / 2
    points = amp * np.sin(2 * np.pi * freq * t + phase) + amp
    dz = np.abs(np.diff(points))
    dist = np.cumsum(dz)

    new_z = np.linspace(0, dist[-1], interp * len(dist))

    dist2idx = interp1d(
        dist,
        np.arange(len(dist), dtype=np.float32),
        kind="linear",
        fill_value="extrapolate",
    )
    mapped_template = dist2idx(new_z)

    with tiff.TiffWriter(str(out_path), bigtiff=True) as tw:
        num_slices = vol.shape[0]
        for b_start in range(0, num_slices, batch_size):
            b_end = b_start + batch_size
            if b_end > num_slices:
                break

            batch = vol[b_start:b_end].copy()

            for z_val in mapped_template:
                lo = int(np.floor(z_val))
                hi = min(lo + 1, batch.shape[0] - 1)
                w = z_val - lo
                slice32 = ((1 - w) * batch[lo] + w * batch[hi]).astype(np.uint16)
                tw.write(slice32, photometric="minisblack")

    print(f"[sine_based_z_interpolation] Saved -> {out_path}")


# -----------------------------------------------------------------------------
# Volume splitting
# -----------------------------------------------------------------------------

def _write_stack_with_spacing(
    output_path: Path,
    stack: np.ndarray,
    x: float,
    y: float,
    z: float,
) -> None:
    """Write a stack with ImageJ-style voxel metadata."""
    tiff.imwrite(
        str(output_path),
        stack,
        imagej=True,
        metadata={"axes": "ZYX", "spacing": z, "unit": "um"},
        resolution=(1.0 / x, 1.0 / y),
        compression="deflate",
    )


def process_image_volume(
    folder_name: str | Path,
    image_name: str,
    volume_size: int,
    *,
    x: float,
    y: float,
    z: float,
    save_odd_only: bool = False,
    save_even_only: bool = False,
) -> None:
    """
    Split a 3D stack into consecutive sub-volumes of `volume_size` slices.

    Even-indexed volumes are reversed to preserve the original acquisition logic.
    """
    raw_path = Path(folder_name) / image_name
    if not raw_path.exists():
        raise FileNotFoundError(f"[process_image_volume] File not found: {raw_path}")

    base = raw_path.stem
    out_odd = raw_path.with_name(f"{base}_{volume_size}_odd")
    out_even = raw_path.with_name(f"{base}_{volume_size}_even")
    out_comb = raw_path.with_name(f"{base}_{volume_size}")

    out_odd.mkdir(exist_ok=True)
    out_even.mkdir(exist_ok=True)
    out_comb.mkdir(exist_ok=True)

    with tiff.TiffFile(str(raw_path)) as tif:
        total_slices = len(tif.pages)
        vol_idx = 1

        for start in range(0, total_slices, volume_size):
            end = start + volume_size
            if end > total_slices:
                break

            block = np.stack(
                [tif.pages[i].asarray() for i in range(start, end)],
                axis=0,
            )

            if vol_idx % 2 == 0:
                block = block[::-1]

            block_uint16 = np.clip(block, 0, 65535).astype(np.uint16)

            targets: list[Path] = []
            if save_odd_only and save_even_only:
                if vol_idx % 2 == 1:
                    targets.append(out_odd / f"{vol_idx}.tif")
                else:
                    targets.append(out_even / f"{vol_idx}.tif")
                targets.append(out_comb / f"{vol_idx}.tif")
            elif save_odd_only and vol_idx % 2 == 1:
                targets.append(out_odd / f"{vol_idx}.tif")
            elif save_even_only and vol_idx % 2 == 0:
                targets.append(out_even / f"{vol_idx}.tif")
            elif not save_odd_only and not save_even_only:
                targets.append(out_comb / f"{vol_idx}.tif")

            for target in targets:
                _write_stack_with_spacing(target, block_uint16, x=x, y=y, z=z)

            if vol_idx % 100 == 0:
                print(f"  Saved volume {vol_idx} (slices {start}-{end - 1})")

            vol_idx += 1

    print(f"[process_image_volume] Finished splitting '{image_name}'.")
    print(f"  Odd volumes  -> {out_odd}")
    print(f"  Even volumes -> {out_even}")
    print(f"  Combined     -> {out_comb}")


# -----------------------------------------------------------------------------
# End-to-end helper
# -----------------------------------------------------------------------------

def run_pipeline(CR: int, start_frame: int, result_path: str | Path) -> None:
    """
    Run the entire post-reconstruction pipeline on a reconstructed TIFF stack.
    """
    result_path = Path(result_path)
    if not result_path.is_file():
        raise FileNotFoundError(f"Reconstruction result not found: {result_path}")

    folder_name = result_path.parent
    result_name = result_path.stem

    result_stack = tiff.imread(str(result_path))
    if result_stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {result_stack.shape}.")

    print(f"[INFO] Loaded result stack shape = {result_stack.shape}")
    numZ, realH, realW = result_stack.shape
    print(f"[INFO] Using realH={realH}, realW={realW} from the result stack.")

    print("\n=== STEP 2: Generate rolling-shutter simulation ===")
    rolling_stack = generate_rolling_shutter_stack(CR, realH, realW)
    print(f"[INFO] rolling_stack shape = {rolling_stack.shape}")

    print("\n=== STEP 3: Rolling-shutter calibration ===")
    calibrated_result = rolling_cali_result_in_memory(rolling_stack, result_stack)
    out_calibrated_path = folder_name / f"c_{result_name}.tif"
    tiff.imwrite(str(out_calibrated_path), calibrated_result)
    print(f"[SAVED] calibrated -> {out_calibrated_path}")

    print("\n=== STEP 4: Reslice camera-scanning mismatch ===")
    reslice_output_path = folder_name / f"rs_{result_name}.tif"
    reslice_camera_scanning_mismatch(out_calibrated_path, reslice_output_path)

    print("\n=== STEP 5: Sine-based z interpolation ===")
    sine_output_name = f"rsine_{result_name}.tif"
    sine_based_z_interpolation(
        folder_name,
        start_frame,
        reslice_output_path.name,
        sine_output_name,
        CR=CR,
        n_period=1,
    )

    print("\n=== STEP 6: Volume splitting ===")
    process_image_volume(
        folder_name,
        sine_output_name,
        volume_size=CR * 10,
        x=0.54,
        y=0.54,
        z=1.4,
        save_odd_only=True,
        save_even_only=True,
    )

    print("\nPipeline completed successfully.")
