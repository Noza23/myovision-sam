# myovision-sam

[![LMU: Munich](https://img.shields.io/badge/LMU-Munich-009440.svg)](https://www.en.statistik.uni-muenchen.de/index.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This is a sub-repository of the main project [myovision](https://github.com/Noza23/myovision).
It's purpose is to perform Training/Fine-Tuning of a prompt-based image segmentation foundation model [SAM](https://github.com/facebookresearch/segment-anything) on myotube images.

# Installation

Dependencies are structured according to the needs of your use-case and can be installed as follows:

```bash
git clone git@github.com:Noza23/myovision-sam.git
cd myovision-sam

# For Base Part containing only Training/Fine-Tuning
pip install .
# For Additional Dependencies for Inference on Myotube Images
pip install .[inference]
# For additionally performin inference on Nuclei Images using Stardist
pip install .[all]
```

# Training / Fine-Tuning

All modules assosicated with Training/Fine-Tuning are located in the `myo_sam/training` directory.
To start Distributed Training/Fine-Tuning of the model:

- Fill out the configuration file `train.yaml`.
- Adjust the `train.sh` Job submission script to perform training on multiple GPUs (was used on SLURM managed cluster) and start the job using:

  ```bash
  sbatch train.sh
  ```

  or locally using torchrun with desired flags and arguments:

  ```bash
  torchrun train.py
  ```

# Inference
