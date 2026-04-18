"""
Edit the file paths and reconstruction settings in the USER SETTINGS section,
then run this script directly from Python.
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from caps_reconstruction import process_one_frame
from caps_tools import finalize_result, read_tif


# =============================================================================
# USER SETTINGS
# =============================================================================
# Replace these example paths with your own TIFF file locations.
DATA_PATH = Path("../examples/test_data.tif")
MASK_PATH = Path("../examples/test_mask.tif")
ORIG_PATH: Optional[Path] = None
OUTPUT_DIR = Path("../examples/results")

# Reconstruction settings.
DENOISER = "TV"
ITER_NUM = 100
TV_WEIGHT = 15.929
RHO = 0.011
TV_ITER_MAX = 5
INI_FRAME = 0
NUM_FRAME: Optional[int] = 20  # Use None to reconstruct all available coded frames.
IMG_INTEN = 2000.0
USE_PARALLEL = True
MAX_WORKERS: Optional[int] = None  # Use None for the default executor behavior.
SAVE_METRICS_CSV = True
OUTPUT_TAG = ""
# =============================================================================


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration for a CAPS reconstruction run."""

    denoiser: str = DENOISER
    iter_num: int = ITER_NUM
    tv_weight: float = TV_WEIGHT
    rho: float = RHO
    tv_iter_max: int = TV_ITER_MAX
    ini_frame: int = INI_FRAME
    num_frame: Optional[int] = NUM_FRAME
    img_inten: float = IMG_INTEN
    max_workers: Optional[int] = MAX_WORKERS
    use_parallel: bool = USE_PARALLEL

    @property
    def lambd(self) -> float:
        return self.tv_weight * self.rho


@dataclass
class ReconstructionResult:
    """Outputs of one reconstruction run."""

    final_result: np.ndarray
    perframe_psnr: list[Optional[float]]
    perframe_ssim: list[Optional[float]]
    perframe_runtime: list[Optional[float]]
    total_runtime: float
    num_mask: int
    results_dir: Path
    save_name: str
    used_ini_frame: int
    used_num_frame: int
    orig: Optional[np.ndarray]
    img_inten: float


def _validate_user_settings() -> tuple[Path, Path, Optional[Path], Path]:
    """Validate the path and reconstruction settings defined at the top of the file."""
    data_path = Path(DATA_PATH)
    mask_path = Path(MASK_PATH)
    orig_path = None if ORIG_PATH is None else Path(ORIG_PATH)
    output_dir = Path(OUTPUT_DIR)

    if not data_path.exists():
        raise FileNotFoundError(f"Data TIFF not found: {data_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask TIFF not found: {mask_path}")
    if orig_path is not None and not orig_path.exists():
        raise FileNotFoundError(f"Ground-truth TIFF not found: {orig_path}")

    if INI_FRAME < 0:
        raise ValueError("INI_FRAME must be non-negative.")
    if NUM_FRAME is not None and NUM_FRAME <= 0:
        raise ValueError("NUM_FRAME must be positive when provided.")
    if ITER_NUM <= 0:
        raise ValueError("ITER_NUM must be positive.")
    if TV_ITER_MAX <= 0:
        raise ValueError("TV_ITER_MAX must be positive.")
    if MAX_WORKERS is not None and MAX_WORKERS <= 0:
        raise ValueError("MAX_WORKERS must be positive when provided.")

    output_dir.mkdir(parents=True, exist_ok=True)
    return data_path, mask_path, orig_path, output_dir


def _prepare_data(
    data_path: Path,
    mask_path: Path,
    orig_path: Optional[Path],
    ini_frame: int,
    num_frame: Optional[int],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int, int]:
    """Load TIFF inputs, normalize them, and compute the frame subset to reconstruct."""
    meas = read_tif(data_path).astype(np.float32)
    mask = read_tif(mask_path).astype(np.float32)

    if meas.ndim != 3 or mask.ndim != 3:
        raise ValueError("Measurement and mask inputs must both be 3D TIFF stacks.")
    if meas.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Measurement and mask spatial dimensions must match, got {meas.shape[:2]} and {mask.shape[:2]}."
        )

    orig = None
    if orig_path is not None:
        orig = read_tif(orig_path).astype(np.float32)
        if orig.ndim != 3:
            raise ValueError("Ground-truth input must be a 3D TIFF stack.")
        orig_max = float(np.max(orig))
        if orig_max > 0:
            orig = orig / orig_max

    meas_max = float(np.max(meas))
    if meas_max <= 0:
        raise ValueError("Measurement stack has non-positive maximum intensity; cannot normalize.")
    meas = meas / meas_max

    mask_max = float(np.max(mask))
    if mask_max <= 0:
        raise ValueError("Mask stack has non-positive maximum intensity; cannot normalize.")
    mask = mask / mask_max

    total_frames = meas.shape[2]
    if ini_frame >= total_frames:
        raise ValueError(
            f"INI_FRAME={ini_frame} is out of range for a measurement stack with {total_frames} frames."
        )

    if num_frame is None:
        used_num_frame = total_frames - ini_frame
    else:
        used_num_frame = min(num_frame, total_frames - ini_frame)

    return meas, mask, orig, ini_frame, used_num_frame


def _make_output_name(data_path: Path, mask_path: Path, config: ReconstructionConfig, tag: str = "") -> str:
    """Build an informative output filename stem."""
    timestamp = datetime.now().strftime("@T%Y%m%d-%H-%M")
    parts = [
        f"ADMM{config.denoiser}",
        f"lambda_{config.lambd:g}",
        f"rho_{config.rho:g}",
        f"tvweight_{config.tv_weight:g}",
        f"iter_{config.iter_num}",
        data_path.stem,
        mask_path.stem,
    ]
    if tag:
        parts.append(tag)
    parts.append(timestamp)
    return "_".join(parts)


def run_reconstruction(
    meas: np.ndarray,
    mask: np.ndarray,
    orig: Optional[np.ndarray],
    config: ReconstructionConfig,
    results_dir: Path,
    save_name: str,
) -> ReconstructionResult:
    """Run CAPS reconstruction on the selected coded frames."""
    if config.num_frame is None:
        raise ValueError("config.num_frame must be set before calling run_reconstruction.")

    num_mask = mask.shape[2]
    height, width = mask.shape[:2]
    final_result = np.zeros((height, width, num_mask * config.num_frame), dtype=np.float32)

    perframe_psnr: list[Optional[float]] = [None] * config.num_frame
    perframe_ssim: list[Optional[float]] = [None] * config.num_frame
    perframe_runtime: list[Optional[float]] = [None] * config.num_frame

    if config.use_parallel:
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []
            for kf in range(config.num_frame):
                orig_k = None
                if orig is not None:
                    start = (config.ini_frame + kf) * num_mask
                    stop = (config.ini_frame + kf + 1) * num_mask
                    orig_k = orig[:, :, start:stop]

                futures.append(
                    executor.submit(
                        process_one_frame,
                        kf,
                        meas,
                        mask,
                        orig_k,
                        config.ini_frame,
                        config.img_inten,
                        config.denoiser,
                        config.iter_num,
                        config.tv_weight,
                        config.rho,
                        config.tv_iter_max,
                    )
                )

            for future in as_completed(futures):
                kf, x_k, runtime_k, psnr_k, ssim_k = future.result()
                final_result[:, :, kf * num_mask : (kf + 1) * num_mask] = x_k
                perframe_psnr[kf] = float(np.mean(psnr_k)) if psnr_k else None
                perframe_ssim[kf] = float(np.mean(ssim_k)) if ssim_k else None
                perframe_runtime[kf] = runtime_k
    else:
        for kf in range(config.num_frame):
            orig_k = None
            if orig is not None:
                start = (config.ini_frame + kf) * num_mask
                stop = (config.ini_frame + kf + 1) * num_mask
                orig_k = orig[:, :, start:stop]

            _, x_k, runtime_k, psnr_k, ssim_k = process_one_frame(
                kf,
                meas,
                mask,
                orig_k,
                config.ini_frame,
                config.img_inten,
                config.denoiser,
                config.iter_num,
                config.tv_weight,
                config.rho,
                config.tv_iter_max,
            )
            final_result[:, :, kf * num_mask : (kf + 1) * num_mask] = x_k
            perframe_psnr[kf] = float(np.mean(psnr_k)) if psnr_k else None
            perframe_ssim[kf] = float(np.mean(ssim_k)) if ssim_k else None
            perframe_runtime[kf] = runtime_k

    total_runtime = float(np.sum([value for value in perframe_runtime if value is not None]))
    return ReconstructionResult(
        final_result=final_result,
        perframe_psnr=perframe_psnr,
        perframe_ssim=perframe_ssim,
        perframe_runtime=perframe_runtime,
        total_runtime=total_runtime,
        num_mask=num_mask,
        results_dir=results_dir,
        save_name=save_name,
        used_ini_frame=config.ini_frame,
        used_num_frame=config.num_frame,
        orig=orig,
        img_inten=config.img_inten,
    )


def save_metrics_csv(result: ReconstructionResult) -> Path:
    """Save overall reconstruction summary metrics as CSV."""
    csv_path = result.results_dir / f"{result.save_name}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        header = ["total_runtime_s"]
        row = [result.total_runtime]

        if result.orig is not None:
            valid_psnr = [value for value in result.perframe_psnr if value is not None]
            valid_ssim = [value for value in result.perframe_ssim if value is not None]
            header.extend(["psnr_mean", "ssim_mean"])
            row.extend([
                float(np.mean(valid_psnr)) if valid_psnr else None,
                float(np.mean(valid_ssim)) if valid_ssim else None,
            ])

        writer.writerow(header)
        writer.writerow(row)
    return csv_path


def main() -> ReconstructionResult:
    data_path, mask_path, orig_path, results_dir = _validate_user_settings()

    meas, mask, orig, used_ini_frame, used_num_frame = _prepare_data(
        data_path=data_path,
        mask_path=mask_path,
        orig_path=orig_path,
        ini_frame=INI_FRAME,
        num_frame=NUM_FRAME,
    )

    config = ReconstructionConfig(
        denoiser=DENOISER,
        iter_num=ITER_NUM,
        tv_weight=TV_WEIGHT,
        rho=RHO,
        tv_iter_max=TV_ITER_MAX,
        ini_frame=used_ini_frame,
        num_frame=used_num_frame,
        img_inten=IMG_INTEN,
        max_workers=MAX_WORKERS,
        use_parallel=USE_PARALLEL,
    )
    save_name = _make_output_name(data_path, mask_path, config, tag=OUTPUT_TAG)

    result = run_reconstruction(
        meas=meas,
        mask=mask,
        orig=orig,
        config=config,
        results_dir=results_dir,
        save_name=save_name,
    )

    finalize_result(
        result_3d=result.final_result,
        runtime=result.total_runtime,
        perframe_psnr=result.perframe_psnr,
        perframe_ssim=result.perframe_ssim,
        orig=result.orig,
        Cr=result.num_mask,
        resultsdir=result.results_dir,
        save_name=result.save_name,
        iframe=result.used_ini_frame,
        nframe=result.used_num_frame,
        img_inten=result.img_inten,
        show_res_flag=True,
        save_res_flag=True,
        tv_weight=config.tv_weight,
        iter_max=config.iter_num,
    )

    if SAVE_METRICS_CSV:
        metrics_path = save_metrics_csv(result)
        print(f"Metrics CSV saved to: {metrics_path}")

    print(f"Reconstruction completed for {used_num_frame} coded frame(s).")
    print(f"Output directory: {results_dir}")
    print(f"Output stem: {save_name}")
    return result


if __name__ == "__main__":
    main()
