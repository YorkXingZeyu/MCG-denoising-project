# MCG Signal Denoising with UNet1D

A PyTorch-based pipeline for magnetocardiography (MCG) signal denoising using a 1D U-Net architecture. Supports real and simulated datasets, configurable train/validation/test splitting (70/10/20), and modular utilities for metrics and visualization.

## Table of Contents

* [Project Structure](#project-structure)
* [Prerequisites](#prerequisites)
* [Requirements](#Requirements)
* [Data Preparation](#data-preparation)
  * [Guidelines](#Guidelines)
  * [Data preparation module](#Data-preparation-module)
* [Configuration](#configuration)
* [Usage](#usage)
  * [Training](#training)
  * [Evaluation](#evaluation)
* [Utilities](#utilities)
* [License](#license)

## Project Structure

```text
├── config/
│   └── base.yaml            # Default configuration parameters
├── Dataset/
│   ├── Real_Data/           # Real MCG .npy files
│       ├── label_1280.npy     # Ground-truth clean MCG signals
│       └── noise_1280.npy     # Corresponding noisy MCG signals
│   └── Simulated_Data/      # Simulated MCG .npy files
        ├── label_1280.npy     # Simulated clean MCG signals
        └── noise_1280.npy     # Simulated noisy MCG signals
├── Data_Preparation/
│   ├── data_preparation.py  # Data loading & preprocessing
│   └── data_loader.py       # Splitting & DataLoader creation
├── utils/
│   ├── helpers.py           # Seeding & reproducibility
│   ├── metrics.py           # Evaluation metrics
│   └── plot.py              # Visualization functions
├── models/
│   └── unet1d.py            # UNet1D model definition
├── trainandtest/
│   ├── train_unet.py        # Training loop
│   └── evaluate.py          # Evaluation loop
├── main.py                  # Entry point
└── README.md                # This file
```

## Prerequisites

* Python
* PyTorch
* scikit-learn
* matplotlib
* PyYAML
* numpy

## Requirements

Save the following into `requirements.txt` and install via `pip install -r requirements.txt`:

```txt
numpy
scikit-learn
matplotlib
torch
PyYAML
einops
```

## Data Preparation

⚠️ data will release soon

## Data Preparation

The dataset should follow the structure below:

```txt
Dataset/
├── Real_Data/
│   ├── label_1280.npy     # Ground-truth clean MCG signals
│   └── noise_1280.npy     # Corresponding noisy MCG signals
└── Simulated_Data/
    ├── label_1280.npy     # Simulated clean MCG signals
    └── noise_1280.npy     # Simulated noisy MCG signals
```

Each `.npy` file is expected to be a NumPy array of shape `[N, 1280]`, where:

* `N` is the number of signal samples.
* Each row represents a single MCG signal of length 1280.

> ⚠️ **Note:** The dataset files are currently not publicly available.
> To run this project with your own data, please prepare your MCG signals and save them in the same format and file names as shown above.

---

### 1. Guidelines for Using Your Own Data

* Ensure that `label_1280.npy` and `noise_1280.npy` are **aligned**
  (i.e., `label[i]` is the clean version of `noise[i]`).
* Place real-world recordings under `Real_Data/` and simulated signals under `Simulated_Data/`.
* File format requirements:

  * **Real Data** (`Real_Data/`): saved with `pickle.dump()` → loaded with `pickle.load()`.
  * **Simulated Data** (`Simulated_Data/`): saved with `np.save()` → loaded with `np.load(allow_pickle=True)`.
* All files must be NumPy arrays of shape `[N, 1280]`.
* Preprocessing, normalization, and train/val/test splitting are handled in:

  * `Data_Preparation/data_preparation.py`
  * `Data_Preparation/data_loader.py`

---

### 2. Data Preprocessing Details

This module provides preprocessing logic for MCG signal denoising:

* **Normalization**
  All signals are divided by **200**, assuming the original amplitude range is roughly `[-200, 200]`.
  ⚠️ If you use your own dataset, verify whether this normalization is appropriate.
  You may adjust or remove the `/ 200` step based on your signal’s dynamic range.

* **Real Data** (`Real_Data/`)

  1. Loaded via `pickle.load()`.
  2. Signals normalized (`/ 200`), then **zero-centered** per sample.

* **Simulated Data** (`Simulated_Data/`)

  1. Loaded via `np.load(..., allow_pickle=True)`.
  2. Signals normalized (`/ 200`), then **filter out** samples
     where `max|noisy – clean| ≥ 0.525`.

* **Output Format**

  * `X`: noisy signals, shape **(N, 1280, 1)**
  * `y`: clean signals, shape **(N, 1280, 1)**



## Configuration

All hyperparameters are set in `config/base.yaml`:

```yaml
seed: 42
device: cuda:0
output_folder: MCG_simulated_result  # MCG_real_result or MCG_simulated_result

train:
  batch_size: 32
  lr: 0.0001
  epochs: 200
  step_size: 1000
  gamma: 0
  early_stopping: 10
  weight_decay: 0
  unetloss_print_weight: 10
  train_model: True

test:
  batch_size: 10
  evaluate_model: True
```

## Usage

Run the full pipeline via:

```bash
python main.py --config config/base.yaml
```

### Training

* Controlled by `train.train_model` flag.
* Checkpoints and logs are saved under the specified `output_folder`.

### Evaluation

* Controlled by `test.evaluate_model` flag.
* Metrics computed in `utils/metrics.py`, plots generated by `utils/plot.py`.

## Utilities

* **helpers.py**: Fix random seeds for reproducibility.
* **metrics.py**: Compute MAE, MSE, SNR, and other denoising metrics.
* **plot.py**: Functions to visualize noisy vs. clean signals and training curves.

## License

This project is licensed under the MIT License.
