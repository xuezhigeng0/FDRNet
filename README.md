# SFDRNet

Official PyTorch implementation of the manuscript:

**Precipitation Prediction via Spatiotemporal-Frequency Differential and Frequency-Dynamic Residual Network**

## Overview

SFDRNet is an encoder–translator–decoder framework developed for precipitation prediction under sparse and spatially inhomogeneous precipitation patterns. The model aims to balance prediction accuracy, temporal stability, and computational efficiency by jointly modeling spatiotemporal, multi-scale, and frequency-domain information.

The framework contains four main components:

- **Statistics-guided Dynamic Spectral Extractor (SDSE):** strengthens front-end spatiotemporal-frequency representation through adaptive cross-scale spectral fusion, statistics-guided frequency-band selection, and background-aware gating.
- **Statistics-guided ODE Differential Pyramid (SODP):** captures subtle cross-scale structural variations and performs coarse-to-fine multi-scale aggregation using ordinary differential equation based interaction.
- **Gated Multi-order Projected Interaction (GMPI):** models latent temporal dependencies and long-range interactions through multi-order projections over tokenized latent features.
- **Frequency-Dynamic Residual Refinement (FDRR):** performs lightweight frequency-aware residual correction to improve precipitation boundaries, localized rain cells, and fine-grained intensity transitions.

The full SFDRNet model contains approximately **1.534 million parameters** in the experimental configuration reported in the manuscript.

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

Detailed download and preprocessing instructions will be provided in the Data Preparation section.

## Repository Structure

```text
FDRNet/
├── openstl/       # Model and framework implementation
├── tools/         # Training, testing, and visualization tools
├── work_dirs/     # Experiment outputs and checkpoints
├── LICENSE
└── README.md
```

## Installation

The exact Python, PyTorch, CUDA, and dependency requirements will be documented after the released environment is verified.

## Data Preparation

Dataset download, directory organization, preprocessing, and normalization instructions will be documented here.

## Training

Verified training commands and configuration files will be documented here.

## Evaluation

Verified testing commands, checkpoint loading instructions, and evaluation metrics will be documented here.

The reported evaluation metrics include:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Signal-to-Noise Ratio (SNR)

## Pretrained Models

Pretrained checkpoint names, download locations, and integrity checks will be provided here.

## Reproducing the Main Results

This section will provide:

1. the exact environment;
2. dataset preparation commands;
3. training configurations;
4. testing commands;
5. pretrained checkpoints;
6. expected evaluation results.

## Citation

Citation information will be updated after the manuscript is formally published.

## Acknowledgements

This implementation is developed using components from the OpenSTL framework. Relevant third-party copyright and license notices should be retained.

## License

This project is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
