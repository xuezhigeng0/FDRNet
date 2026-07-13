# SFDRNet

Official PyTorch implementation of the manuscript:

**Precipitation Prediction via Spatiotemporal-Frequency Differential and Frequency-Dynamic Residual Network**

## Overview

SFDRNet is an encoder–translator–decoder framework developed for precipitation prediction under sparse and spatially inhomogeneous precipitation patterns. The model aims to balance prediction accuracy, temporal stability, and computational efficiency by jointly modeling spatiotemporal, multi-scale, and frequency-domain information.

The framework contains four main components:

- **Statistics-guided Dynamic Spectral Extractor (SDSE):** strengthens front-end spatiotemporal-frequency representation through adaptive cross-scale spectral fusion, statistics-guided frequency-band selection, and background-aware gating.
- **Statistics-guided ODE Differential Pyramid (SODP):** captures subtle cross-scale structural variations and performs coarse-to-fine multi-scale aggregation through ordinary-differential-equation-based interactions.
- **Gated Multi-order Projected Interaction (GMPI):** models latent temporal dependencies and long-range interactions through multi-order projections over tokenized latent features.
- **Frequency-Dynamic Residual Refinement (FDRR):** performs lightweight frequency-aware residual correction to improve precipitation boundaries, localized rain cells, and fine-grained intensity transitions.

The full SFDRNet model contains approximately **1.534 million parameters** under the experimental configuration reported in the manuscript.

## Dataset

Experiments are conducted on two single-variable total precipitation benchmarks derived from WeatherBench:

| Benchmark | Spatial resolution | Grid size |
|---|---:|---:|
| Total precipitation at 5.625° | 5.625° | 32 × 64 |
| Total precipitation at 2.8125° | 2.8125° | 64 × 128 |

The datasets are chronologically divided as follows:

- Training set: 2010–2015
- Validation set: 2016
- Test set: 2017–2018

Detailed download, preprocessing, normalization, and directory-organization instructions will be added after the released data-processing pipeline has been fully verified.

## Repository Structure

```text
FDRNet/
├── openstl/           # Model and framework implementation
├── tools/             # Training, testing, and visualization tools
├── work_dirs/         # Experiment outputs and pretrained checkpoints
├── requirements.txt   # Python dependencies
├── LICENSE
└── README.md
```

## Installation

### Tested Environment

The code has been tested in the following environment:

- Operating system: Linux
- Python: 3.10.20
- PyTorch: 2.1.1+cu118
- CUDA: 11.8
- GPU: NVIDIA GeForce RTX 3090

The reported experiments were conducted using an NVIDIA GeForce RTX 3090. The minimum GPU-memory requirement depends on the spatial resolution, batch size, input sequence length, and training configuration.

### Clone the Repository

```bash
git clone https://github.com/xuezhigeng0/FDRNet.git
cd FDRNet
```

### Create a Conda Environment

```bash
conda create -n sfdrnet python=3.10.20 -y
conda activate sfdrnet
```

### Install PyTorch

Install the tested PyTorch version with CUDA 11.8:

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

### Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Verify the Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

A successful installation should report:

```text
PyTorch: 2.1.1+cu118
CUDA: 11.8
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

The GPU name may differ when another compatible NVIDIA GPU is used.

## Data Preparation

The released data-preparation guide will include:

1. dataset download instructions;
2. expected directory structure;
3. preprocessing commands;
4. normalization settings;
5. chronological data splits;
6. input and target sequence organization.

The expected benchmark dimensions are:

```text
weather_tp_5_625:
  spatial resolution: 5.625°
  grid size: 32 × 64

weather_tp_2_8125:
  spatial resolution: 2.8125°
  grid size: 64 × 128
```

If the complete processed datasets cannot be redistributed directly, official download instructions and preprocessing scripts will be provided.

## Training

Verified training commands and configuration files will be added after the complete released training pipeline has been tested from a clean clone of this repository.

The final training guide will specify:

- dataset name;
- configuration-file path;
- input sequence length;
- prediction sequence length;
- batch size;
- learning rate;
- number of epochs;
- random seed;
- checkpoint output directory;
- GPU configuration.

## Evaluation

Verified evaluation commands and checkpoint-loading instructions will be added after the released evaluation pipeline has been tested from a clean environment.

The reported evaluation metrics include:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Signal-to-Noise Ratio (SNR)

## Pretrained Models

Pretrained SFDRNet checkpoints are currently stored under the experiment-output directory:

```text
work_dirs/weather_tp_5_625/SFDRNet/checkpoints/
```

The final release documentation will provide:

- checkpoint filename;
- corresponding dataset;
- spatial resolution;
- model configuration;
- loading command;
- expected evaluation results;
- file-integrity information.

## Reproducing the Main Results

The complete reproduction guide will provide:

1. the tested software environment;
2. dataset download and preprocessing instructions;
3. experiment configuration files;
4. training commands;
5. evaluation commands;
6. pretrained checkpoints;
7. expected quantitative results;
8. random seeds;
9. hardware information.

All released commands will be tested from a clean clone of this repository before the final archival release.

## User Guide

The final user guide will document:

- expected input tensor shape;
- expected output tensor shape;
- sequence-length settings;
- spatial-resolution settings;
- configuration options;
- checkpoint-loading behavior;
- output files and evaluation results;
- visualization procedures.

## Citation

Citation information will be updated after the manuscript is formally published.

Manuscript title:

```text
Precipitation Prediction via Spatiotemporal-Frequency Differential
and Frequency-Dynamic Residual Network
```

## Acknowledgements

This implementation is developed using components from the OpenSTL framework. Relevant third-party copyright, attribution, and license notices are retained in accordance with their original licenses.

## License

This project is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
