# CS-LSM

## Project Description
This project implements a Plug-and-Play Alternating Direction Method of Multipliers (PnP-ADMM) algorithm for image reconstruction from compressed sensing measurements captured by our customized compressed sensing enhanced light-sheet microscope (CS-LSM). 

## References
- Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. (2016). Plug-and-play priors for model based reconstruction. *IEEE Global Conference on Signal and Information Processing (GlobalSIP)*. [Link to Paper](https://ieeexplore.ieee.org/document/7744574)
- Yuan, X., Liu, Y., Suo, J., & Dai, Q. (2020). *Plug-and-Play Algorithms for Large-Scale Snapshot Compressive Imaging*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1444–1454. [Link to Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9156491)

## Setup Instructions

### 1. Download Project Files and data
- Download all files in this project repository to your local system.
- Download data [here](https://cometmail-my.sharepoint.com/:f:/g/personal/xxz210006_utdallas_edu/ElB5h9w2YupKpZrT6fEYbNoBR5lAPcF0iJJ6MsnlzFzrXw?e=kJvc2l) to your local system.

### 2. Install the Python Environment
- Download and install [Anaconda](https://www.anaconda.com/products/distribution).
- Create the required environment by running:
  ```bash
  conda env create -f cs_lsm.yml
  ```
- Activate the environment with:
  ```bash
  conda activate cs_lsm
  ```

### 3. Configure Input Data
- Set up the input data paths by editing lines 15-27 in `cs_lsm_simu_092924.py`.

### 4. Adjust Reconstruction Parameters
- Configure the reconstruction parameters on lines 64-71 in `cs_lsm_simu_092924.py` as needed.

### 5. Run the Code
- Execute the script to run the reconstruction:
  ```bash
  python cs_lsm_simu_092924.py
  ```
- Check the output files in the `results` folder.
