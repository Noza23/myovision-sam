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
# For Additional Dependencies for Inference on Myotube and Nuclei Images
pip install .[all]
```

# Training / Fine-Tuning

All modules assosicated with Training/Fine-Tuning are located in the `myo_sam.training` sub-module.
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

  or on a single GPU:

  ```bash
  python3 train.py
  ```

  ```diff
  - Note: The Snapshots are overwritten by default, so make a copy of the model before starting the training.
  ```

  ## Logging and Monitoring

  - `myosam.log` file which will be created in the execution directory will contain
    text logs of the training process.
  - `runs` directory which will be created in the execution directory will contain Tensorboard logs for monitoring the training process.

  ## Adjust to your Data

  To adjust training to your data just change the dataloader in `myo_sam.training.dataset` sub-module.

# Inference

All modules assosicated with Inference are located in the `myo_sam.inference` sub-module.
To perform Inference on Myotube & Nuclei Images in batch mode:

- Fill out the configuration file `inference.yaml`.
- Adjust the `inference.sh` Job submission script to perform inference on multiple GPUs (was used on SLURM managed cluster) and start the job using:

  ```bash
  sbatch inference.sh
  ```

  or locally using torchrun with desired flags and arguments:

  ```bash
  torchrun inference.py
  ```

  or on a single GPU:

  ```bash
  python3 inference.py
  ```

  ```diff
  - Note: Between Myotube and Nuclei Images Dirrectories you should have the following naming convention:
    - Myotube Images: `x_{myotube_image_suffix}.png`
    - Nuclei Images: `x_{nuclei_image_suffix}.png`
    Meaning: paris of images should have the same base name until the last underscore.
  ```
