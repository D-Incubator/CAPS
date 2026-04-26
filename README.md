# CAPS

## Compressive axial-integrated planar scanning (CAPS) microscopy for high-speed volumetric imaging of cardiac dynamics

This repository contains acquisition, reconstruction, and post-processing code for **compressive axial-integrated planar scanning (CAPS) microscopy**, a compressed sensing framework for high-speed volumetric fluorescence imaging. CAPS combines detection-side optical multiplexing with model-based reconstruction to recover volumetric image sequences from compressed measurements, enabling efficient imaging of fast biological dynamics such as the beating heart.

The computational workflow is built around a **Plug-and-Play Alternating Direction Method of Multipliers (PnP-ADMM)** reconstruction framework, followed by post-processing steps for rolling-shutter calibration, reslicing, sine-based interpolation, and volume splitting. In addition to the Python reconstruction pipeline, this repository also includes a **LabVIEW acquisition program** for hardware control during CAPS data acquisition.

A preprint of the associated paper is available on bioRxiv:  
placeholder

---

## Repository Structure

```text
CAPS/
├── caps_env.yml
├── acquisition/
│   ├── image_acquisition.vi
│   ├── find_optimal_control_parameters.vi
│   ├── camera_config.vi
│   ├── control_laser.vi
│   ├── galvo_ETL_offset_config.vi
│   ├── generate_analogue_out_waveform.vi
│   ├── MATLAB_helper.vi
│   └── dcimg_to_tif.vi
├── recon_scripts/
│   ├── caps_main.py
│   ├── caps_reconstruction.py
│   ├── caps_tools.py
│   ├── pipeline_run.py
│   └── pipeline_functions.py
└── examples/
    ├── test_data.tif
    └── test_mask.tif
```

---

## File Overview

### Environment file
- `caps_env.yml`  
  Conda environment file for creating the recommended Python environment.

### Acquisition code
Located in the `acquisition/` folder:

- `image_acquisition.vi`  
  LabVIEW program for CAPS hardware control during data acquisition.

- `find_optimal_control_parameters.vi`  
  LabVIEW program for finding optimal control parameters of CAPS hardware.

*Note: The rest are internal functions.*

### Reconstruction scripts
Located in the `recon_scripts/` folder:

- `caps_main.py`  
  Main reconstruction script. Edit the input and output paths directly in the script, then run the reconstruction.

- `caps_reconstruction.py`  
  Core PnP-ADMM reconstruction functions and denoising routines.

- `caps_tools.py`  
  Utility functions for TIFF input/output and CAPS forward and adjoint operators.

### Post-processing scripts
Also located in the `recon_scripts/` folder:

- `pipeline_run.py`  
  Main post-processing pipeline script for rolling-shutter calibration, reslicing, sine-based interpolation, and volume splitting.

- `pipeline_functions.py`  
  Helper functions used by the post-processing pipeline.

### Example data
Located in the `examples/` folder:

- `test_data.tif`  
  Example compressed CAPS measurement data for testing the reconstruction workflow.

- `test_mask.tif`  
  Example coding mask corresponding to the test dataset.

---

## Setup

### 1. Download the repository
Download or clone this repository to your local machine.

### 2. Create the Python environment
We recommend using Anaconda.

Create the environment from the YAML file:

```bash
conda env create -f caps_env.yml
```

Activate the environment:

```bash
conda activate caps
```

---

## Reconstruction Workflow

### 1. Configure paths in `caps_main.py`
Open `recon_scripts/caps_main.py` and edit the user settings section to specify the paths for:

- `DATA_PATH`
- `MASK_PATH`
- `OUTPUT_DIR`

For the example dataset, these paths should point to files in the `examples/` folder.

### 2. Adjust reconstruction parameters
In `recon_scripts/caps_main.py`, modify the reconstruction settings as needed, such as:

- initial frame
- number of frames
- denoiser type
- number of iterations
- TV weight
- ADMM parameter `rho`
- parallel or sequential execution

### 3. Run the reconstruction
From the repository root directory, run:

```bash
python recon_scripts/caps_main.py
```

The reconstructed result will be saved to the output directory specified in the script.

---

## Post-processing Workflow

After reconstruction, you can run the post-processing pipeline to perform rolling-shutter calibration and downstream processing.

### 1. Configure `pipeline_run.py`
Open `recon_scripts/pipeline_run.py` and set:

- the reconstructed TIFF file path
- the compression ratio (`CR`)
- the start frame, if needed

The current script can be configured to use a predefined start frame for the test dataset.

### 2. Run the post-processing pipeline
From the repository root directory, run:

```bash
python recon_scripts/pipeline_run.py
```

This pipeline performs:

1. rolling-shutter simulation  
2. rolling-shutter calibration  
3. reslicing for camera-scanning mismatch  
4. sine-based z interpolation  
5. volume splitting

---

## Example Data

The repository includes example files in the `examples/` folder:

- `examples/test_data.tif`
- `examples/test_mask.tif`

These files can be used directly to test the reconstruction workflow after updating the paths in `recon_scripts/caps_main.py`.

---

## Notes

- The reconstruction code is designed for TIFF-based CAPS data.
- Input and output paths are configured directly inside the Python scripts.
- The example files are intended for demonstration and validation of the workflow.
- Runtime and memory usage depend on dataset size and reconstruction settings.
- The LabVIEW acquisition code is included for instrument control and is separate from the Python reconstruction and post-processing workflow.

---

## Citation

If you use this code in your research, please cite:

**Xinyuan Zhang et al.**  
*Compressive axial-integrated planar scanning (CAPS) microscopy for high-speed volumetric imaging of cardiac dynamics.*
