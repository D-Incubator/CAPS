"""
CAPS post-processing pipeline.

Workflow
--------
1. Inspect reconstructed TIFF
2. Generate rolling-shutter simulation
3. Run rolling-shutter calibration
4. Reslice for camera-scanning mismatch
5. Run sine-based z interpolation
6. Split the output into volumes
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import tifffile as tiff
from PyQt5.QtWidgets import QApplication, QFileDialog

from pipeline_functions import (
    generate_rolling_shutter_stack,
    process_image_volume,
    reslice_camera_scanning_mismatch,
    rolling_cali_result_in_memory,
    sine_based_z_interpolation,
)


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

CR = 15

# Set RESULT_PATH directly to skip the file dialog.
# Leave as None to pick a TIFF interactively.
RESULT_PATH: Optional[Path] = None

# Default test-data start frame.
DEFAULT_START_FRAME = 42

# Set to True if you want to type the start frame in the console at runtime.
PROMPT_FOR_START_FRAME = False

# Save the chosen start frame to "last_start_frame.txt".
SAVE_START_FRAME_TXT = True

# Output settings
SAVE_ROLLING_PATTERN = True
PIXEL_X_UM = 0.54
PIXEL_Y_UM = 0.54
PIXEL_Z_UM = 1.4


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def pick_tiff() -> Optional[Path]:
    """Open a file dialog and return a TIFF path, or None if cancelled."""
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    dialog = QFileDialog()
    dialog.setWindowTitle("Select a reconstruction TIFF")
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter("TIFF Files (*.tif *.tiff);;All Files (*)")
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)

    selected: Optional[Path] = None
    if dialog.exec_():
        selected = Path(dialog.selectedFiles()[0])
        print(f"[INFO] Selected file: {selected}")
    else:
        print("No file selected. Aborting.")

    if owns_app and app is not None:
        app.quit()

    return selected


def resolve_start_frame(default_value: int) -> int:
    """
    Return the start frame.

    By default this uses DEFAULT_START_FRAME. If PROMPT_FOR_START_FRAME is True,
    the user is asked for a value at runtime.
    """
    if not PROMPT_FOR_START_FRAME:
        print(f"[INFO] Using default start_frame = {default_value}")
        return default_value

    while True:
        user = input("Enter start_frame for sine correction (integer) -> ")
        try:
            return int(user)
        except ValueError:
            print("Please type a valid integer.")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:
    result_path = RESULT_PATH if RESULT_PATH is not None else pick_tiff()
    if result_path is None:
        raise SystemExit(1)

    result_path = Path(result_path)
    if not result_path.is_file():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    folder = result_path.parent
    result_name = result_path.stem

    print("\n=== STEP 1: Inspect reconstruction result ===")
    stack = tiff.imread(str(result_path))
    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF stack, got shape {stack.shape}.")

    numZ, H, W = stack.shape
    print(f"[INFO] stack shape = {stack.shape}")

    print("\n=== STEP 2: Generate rolling-shutter stack ===")
    rs_stack = generate_rolling_shutter_stack(CR, H, W)
    print(f"[INFO] rolling_stack shape = {rs_stack.shape}")

    if SAVE_ROLLING_PATTERN:
        rolling_path = folder / f"rolling_shutter_pattern_CR{CR}.tif"
        tiff.imwrite(str(rolling_path), rs_stack)
        print(f"[SAVED] rolling-shutter pattern -> {rolling_path}")

    print("\n=== STEP 3: Rolling-shutter calibration ===")
    calibrated = rolling_cali_result_in_memory(rs_stack, stack)
    cal_path = folder / f"c_{result_name}.tif"
    tiff.imwrite(str(cal_path), calibrated)
    print(f"[SAVED] calibrated -> {cal_path}")

    print("\n=== STEP 4: Reslice camera/scanning mismatch ===")
    rs_path = folder / f"rs_{result_name}.tif"
    reslice_camera_scanning_mismatch(cal_path, rs_path)

    start_frame = resolve_start_frame(DEFAULT_START_FRAME)

    if SAVE_START_FRAME_TXT:
        sf_txt = folder / "last_start_frame.txt"
        sf_txt.write_text(str(start_frame), encoding="utf-8")
        print(f"[INFO] start_frame saved to {sf_txt}")

    print("\n=== STEP 5: Sine-based Z interpolation ===")
    interp_value = 4 if "100Hz" in result_name else 2
    sine_out_name = f"rsine_{result_name}.tif"
    sine_based_z_interpolation(
        folder,
        start_frame,
        rs_path.name,
        sine_out_name,
        CR=CR,
        n_period=1,
        interp=interp_value,
    )

    print("\n=== STEP 6: Volume splitting ===")
    process_image_volume(
        folder,
        sine_out_name,
        volume_size=CR * 10,
        x=PIXEL_X_UM,
        y=PIXEL_Y_UM,
        z=PIXEL_Z_UM,
        save_odd_only=False,
        save_even_only=False,
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
