# SFDRNet

Official PyTorch implementation of:

**Precipitation Prediction via Spatiotemporal-Frequency Differential and Frequency-Dynamic Residual Network**

SFDRNet is a precipitation prediction model built on the OpenSTL framework.

## Environment

The released code was tested with:

- Linux
- Python 3.10.20
- PyTorch 2.1.1+cu118
- CUDA 11.8
- timm 0.6.11
- NVIDIA RTX 3090

## Installation

```bash
conda create -n sfdrnet python=3.10.20 -y
conda activate sfdrnet

pip install torch==2.1.1+cu118 \
  torchvision==0.16.1+cu118 \
  torchaudio==2.1.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -e .
```

## Dataset

The model uses hourly WeatherBench total precipitation at 5.625-degree resolution.

Each NetCDF file must contain:

- Variable: `tp`
- Dimensions: `(time, lat, lon)`
- Spatial shape: `32 x 64`

Expected directory structure:

```text
DATA_ROOT/
└── weather/
    └── total_precipitation/
        ├── total_precipitation_2010_5.625deg.nc
        ├── total_precipitation_2011_5.625deg.nc
        ├── ...
        └── total_precipitation_2018_5.625deg.nc
```

Dataset split:

| Split | Years |
|---|---|
| Training | 2010-2015 |
| Validation | 2016 |
| Testing | 2017-2018 |

The model uses 12 input frames to predict the following 12 frames.

## Training

```bash
python tools/train.py \
  --dataname weather_tp_5_625 \
  --config_file configs/weather/tp_5_625/SFDRNet.py \
  --data_root /path/to/DATA_ROOT \
  --res_dir work_dirs \
  --ex_name SFDRNet
```

## Evaluation

Place the trained checkpoint at:

```text
work_dirs/weather_tp_5_625/SFDRNet/checkpoints/best.ckpt
```

Then run:

```bash
python tools/test.py \
  --dataname weather_tp_5_625 \
  --config_file configs/weather/tp_5_625/SFDRNet.py \
  --data_root /path/to/DATA_ROOT \
  --res_dir work_dirs \
  --ex_name SFDRNet
```

## Configuration

The released experiment configuration is:

```text
configs/weather/tp_5_625/SFDRNet.py
```

Main settings:

| Parameter | Value |
|---|---:|
| Input shape | `12 x 1 x 32 x 64` |
| Batch size | 16 |
| Epochs | 200 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Spatial hidden channels | 32 |
| Temporal hidden channels | 256 |
| Spatial blocks | 2 |
| Temporal blocks | 3 |

## Compatibility Notes

The public model name is **SFDRNet**.

For compatibility with OpenSTL and the released checkpoint, the implementation retains:

```python
method = "SimVP"
model_type = "vit"
```

Early experiment records may use the name **SimVP-TRF3**. SimVP-TRF3 and SFDRNet refer to the same model.

The cleaned implementation was strictly verified against the original checkpoint:

- 504 matching state-dictionary entries
- 0 missing entries
- 0 unexpected entries
- 0 shape mismatches

The model contains 1,533,606 parameters.

## Acknowledgements

This repository is based on OpenSTL. We thank the OpenSTL authors and contributors.

## License

This project is released under the Apache License 2.0.
