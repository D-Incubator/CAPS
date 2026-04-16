# CAPS

## Compressive axial-integrated planar scanning (CAPS) microscopy for high-speed volumetric imaging of cardiac dynamics

This repository contains the reconstruction and post-processing code for **compressive axial-integrated planar scanning (CAPS) microscopy**, a compressed sensing framework for high-speed volumetric fluorescence imaging. CAPS combines detection-side optical encoding with model-based reconstruction to recover volumetric dynamics from compressed measurements.

The reconstruction pipeline is based on a **Plug-and-Play Alternating Direction Method of Multipliers (PnP-ADMM)** framework, and the repository also includes post-processing tools for rolling-shutter calibration, reslicing, sine-based interpolation, and volume splitting.

A bioRxiv preprint of the associated paper is available here:  
**[Add bioRxiv link here]**

---

## Repository Contents

The main files in this repository are:

- `caps_main.py`  
  Main reconstruction script. Edit the input and output paths directly in the script, then run the CAPS reconstruction.

- `caps_reconstruction.py`  
  Core PnP-ADMM reconstruction functions and denoising routines.

- `caps_tools.py`  
  Utility functions for TIFF I/O, result saving, and CAPS forward/adjoint operators.

- `test_data.tif`  
  Example compressed CAPS measurement data for testing the reconstruction workflow.

- `test_mask.tif`  
  Example coding mask corresponding to the test data.

- `pipeline_run.py`  
  Main post-processing pipeline script for rolling-shutter calibration and downstream volume processing.

- `pipeline_functions.py`  
  Functions used by the post-processing pipeline.

- `caps_env.yml`  
  Conda environment file for reproducing the software environment used by this project.

---

## Folder Organization

Place all repository files in the **same folder**. A typical layout is:

```text
CAPS/
├── caps_main.py
├── caps_reconstruction.py
├── caps_tools.py
├── pipeline_run.py
├── pipeline_functions.py
├── caps_env.yml
├── test_data.tif
├── test_mask.tif
└── README.md
```

This repository is set up so that the example code can be run by editing file paths directly in the scripts.

---

## Setup

### 1. Download the repository files
Download or clone all repository files to your local machine and keep them in the same folder.

### 2. Create the Python environment
We recommend using Anaconda or Miniconda.

Create the environment using:

```bash
conda env create -f caps_env.yml
```

Activate the environment using:

```bash
conda activate caps
```

---

## Reconstruction Workflow

### 1. Configure input and output paths
Open `caps_main.py` and edit the user settings section near the top of the file. Set:

- `DATA_PATH`
- `MASK_PATH`
- `OUTPUT_DIR`
- optionally `ORIG_PATH` if ground truth is available

For a quick test, you can point the script to the included example files:

```python
DATA_PATH = Path("test_data.tif")
MASK_PATH = Path("test_mask.tif")
OUTPUT_DIR = Path(".")
```

If all files are placed in the same folder, these relative paths are sufficient.

### 2. Adjust reconstruction parameters
In `caps_main.py`, you can customize reconstruction settings such as:

- initial frame index
- number of frames to reconstruct
- denoiser type
- number of ADMM iterations
- TV weight
- ADMM penalty parameter `rho`
- sequential or parallel execution

### 3. Run the reconstruction
Execute:

```bash
python caps_main.py
```

The reconstructed result will be saved to the output directory specified in the script.

---

## Post-processing Workflow

After reconstruction, you can run the post-processing pipeline to perform rolling-shutter calibration and subsequent processing.

### 1. Configure `pipeline_run.py`
Open `pipeline_run.py` and set the reconstruction result path.

The current version is configured so that the **default start frame for the test data is 42**, while still allowing this value to be customized if needed.

### 2. Run the pipeline
Execute:

```bash
python pipeline_run.py
```

This pipeline performs the following steps:

1. rolling-shutter simulation  
2. rolling-shutter calibration  
3. reslicing for camera-scanning mismatch  
4. sine-based z interpolation  
5. volume splitting  

Processed outputs will be written to the same folder as the reconstruction result unless otherwise specified in the script.

---

## Example Data

This repository includes example files for testing the workflow:

- `test_data.tif`
- `test_mask.tif`

These files are intended to help verify that the reconstruction code and post-processing pipeline run correctly on a local system.

---

## Notes

- The code is designed for TIFF-based CAPS data.
- Paths are configured directly inside the Python scripts rather than passed through the command line.
- Depending on dataset size and hardware, reconstruction may require substantial computation time and memory.
- For reproducibility, we recommend using the provided `caps_env.yml` environment file.

---

## Citation

If you use this code in your research, please cite:

**Xinyuan Zhang et al.**  
*Compressive axial-integrated planar scanning (CAPS) microscopy for high-speed volumetric imaging of cardiac dynamics.*

Please also add the bioRxiv citation once the preprint link is available.

---

## Contact

For questions regarding the code, data, or CAPS microscopy workflow, please contact the repository author.
