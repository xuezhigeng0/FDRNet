# SFDRNet

**Spatiotemporal-Frequency Differential and Frequency-Dynamic Residual Network for Sparse Precipitation Image Sequence Prediction**

This repository is the official PyTorch implementation of SFDRNet for sparse precipitation image sequence prediction on WeatherBench total precipitation (`tp`). It builds on OpenSTL and uses its SimVP experiment interface; vanilla SimVP is a separate model.

## Environment and Installation

The released code was tested on Linux with Python 3.10.20, PyTorch 2.1.1+cu118, CUDA 11.8, timm 0.6.11, and an NVIDIA RTX 3090. `requirements.txt` includes `requirements/runtime.txt`; `environment.yml` specifies compatible Python, PyTorch, and CUDA generations.

```bash
conda create -n sfdrnet python=3.10.20 -y
conda activate sfdrnet
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

## Dataset

Use hourly WeatherBench total precipitation NetCDF data from the [WeatherBench project](https://github.com/pangeo-data/WeatherBench). Each file must contain `tp` with dimensions `(time, lat, lon)`. The loader selects files by their resolution suffix.

| Benchmark | Resolution | Grid | Dataset identifier |
|---|---:|---:|---|
| `tp_5.625` | 5.625° | 32 × 64 | `weather_tp_5_625` |
| `tp_2.8125` | 2.8125° | 64 × 128 | `weather_tp_2_8125` |

Both benchmarks use 2010–2015 for training, 2016 for validation, and 2017–2018 for testing. Each sample uses 12 input frames to predict the following 12 frames. Validation and test data are normalized using training statistics.

The loader accepts `DATA_ROOT/weather/total_precipitation/`. It looks for `total_precipitation*_5.625deg.nc` or `total_precipitation*_2.8125deg.nc`, respectively. The supplied generator writes these filenames:

```text
DATA_ROOT/
└── weather/
    └── total_precipitation/
        ├── total_precipitation_2010_5.625deg.nc
        ├── ...
        ├── total_precipitation_2018_5.625deg.nc
        ├── total_precipitation_2010_2.8125deg.nc
        ├── ...
        └── total_precipitation_2018_2.8125deg.nc
```

The loader also accepts `DATA_ROOT/weather_5_625deg/` and `DATA_ROOT/weather_2_8125deg/` as resolution-specific roots, each containing `total_precipitation/`. Files from both resolutions may share `DATA_ROOT/weather/`.

### Synthetic installation checks

```bash
python tools/prepare_data/generate_dummy_tp.py --output_root ./dummy_data --resolution 5.625 --hours_per_year 48
python tools/prepare_data/generate_dummy_tp.py --output_root ./dummy_data --resolution 2.8125 --hours_per_year 48
```

These synthetic files are only for installation and smoke testing. They must not be used to reproduce or report manuscript quantitative results.

## Training and Evaluation

Run these commands from the repository root. Replace `/path/to/DATA_ROOT` with the parent directory containing `weather/` or `weather_<resolution>deg/`.

Train `tp_5.625`:

```bash
python tools/train.py \
  --dataname weather_tp_5_625 \
  --config_file configs/weather/tp_5_625/SFDRNet.py \
  --data_root /path/to/DATA_ROOT \
  --res_dir work_dirs \
  --ex_name SFDRNet_tp_5_625
```

Evaluate `tp_5.625` after training:

```bash
python tools/test.py \
  --test \
  --dataname weather_tp_5_625 \
  --config_file configs/weather/tp_5_625/SFDRNet.py \
  --data_root /path/to/DATA_ROOT \
  --res_dir work_dirs \
  --ex_name SFDRNet_tp_5_625
```

Train `tp_2.8125`:

```bash
python tools/train.py \
  --dataname weather_tp_2_8125 \
  --config_file configs/weather/tp_2_8125/SFDRNet.py \
  --data_root /path/to/DATA_ROOT \
  --res_dir work_dirs \
  --ex_name SFDRNet_tp_2_8125
```

Evaluate `tp_2.8125` after training:

```bash
python tools/test.py \
  --test \
  --dataname weather_tp_2_8125 \
  --config_file configs/weather/tp_2_8125/SFDRNet.py \
  --data_root /path/to/DATA_ROOT \
  --res_dir work_dirs \
  --ex_name SFDRNet_tp_2_8125
```

For each command, outputs are under `work_dirs/<ex_name>/`, including logs and `checkpoints/best.ckpt`. The evaluation command uses `--test` to load that checkpoint. A checkpoint must be produced by training or supplied separately; pretrained weights are not included here. The pipeline supports the manuscript metrics MSE, MAE, RMSE, PSNR, and SNR.

Both configurations specify batch size 16, validation batch size 16, 200 epochs, Adam with learning rate `1e-3` and weight decay `0`, cosine annealing without warmup, seed 42, `hid_S=32`, `hid_T=256`, `N_S=2`, `N_T=3`, spatial kernel size 3, and `drop_path=0.1`. Input and output shapes are `(B, 12, 1, 32, 64)` for 5.625° and `(B, 12, 1, 64, 128)` for 2.8125°.

### Release validation

With the project dependencies installed, run the lightweight CPU checks without training:

```bash
python scripts/validate_sfdrnet_release.py
```

The script checks both configs, dataset registrations, parameter counts, state-dict shapes, and one forward pass per resolution. Use `--skip-high-res-forward` if CPU memory is limited. These are release smoke checks, not a reproduction of manuscript metrics.

## Implementation Mapping

The public code names now match the manuscript terminology exactly. Historical implementation
class and attribute names are retained only for checkpoint and code compatibility.

| Manuscript component | Public code name | Historical implementation | Location |
|---|---|---|---|
| SFDRNet | `SFDRNet` | alias of `SimVP_Model` with `model_type="vit"` through the OpenSTL `SimVP` method | `openstl/models/simvp_model.py`; `openstl/methods/simvp.py` |
| Statistics-guided Dynamic Spectral Extractor (SDSE) | `SDSE` | alias of `SFFE`, used in the encoder and decoder | `openstl/models/simvp_model.py` |
| Statistics-guided ODE Differential Pyramid (SODP) | `SODP` | alias of `MDP` with `MultiStepODEFusion2D` | `openstl/modules/simvp_modules.py` |
| Gated Multi-order Projected Interaction (GMPI) | `GMPI` | alias of `MPA` with `MPABlock` | `openstl/modules/simvp_modules.py` |
| Frequency-Dynamic Residual Refinement (FDRR) | `FDRR` | alias of `FreqBandResidualMix`, used by `ViTSubBlock` | `openstl/modules/simvp_modules.py` |

## Compatibility Notes

For compatibility with OpenSTL and existing checkpoints, the experiment retains:

```python
method = "SimVP"
model_type = "vit"
```

These values select the SFDRNet implementation in this repository; they do not identify vanilla SimVP as SFDRNet. Early experiment records may use **SimVP-TRF3**. SimVP-TRF3 and SFDRNet refer to the same model. Existing historical class/attribute names, parameter names, and tensor shapes are retained. The manuscript-facing aliases `SFDRNet`, `SDSE`, `SODP`, `GMPI`, and `FDRR` point to the same implementation classes and do not change checkpoint keys.

Under either released configuration, the model has 1,533,606 parameters and 504 state-dictionary entries.

## Reported Manuscript Results

These are reported manuscript results, not automatically reproduced measurements from this repository checkout.

| Benchmark | Parameters | MSE | MAE | RMSE | PSNR | SNR |
|---|---:|---:|---:|---:|---:|---:|
| `tp_5.625` | 1.534M | 1.059e-7 | 9.112e-5 | 3.254e-4 | 70.389 | 2.476 |
| `tp_2.8125` | 1.534M | 9.187e-8 | 8.325e-5 | 3.031e-4 | 70.782 | 2.984 |

## Citation

If you use this repository, please cite the corresponding SFDRNet manuscript: “Spatiotemporal-Frequency Differential and Frequency-Dynamic Residual Network for Sparse Precipitation Image Sequence Prediction.” A full BibTeX entry can be added after publication.

## Acknowledgements and License

This repository builds upon [OpenSTL](https://github.com/chengtan9907/OpenSTL). Original OpenSTL components retain their CAIRI/OpenSTL copyright and license notices. SFDRNet-specific modifications are released under the repository's [Apache License 2.0](LICENSE) terms unless otherwise stated.

