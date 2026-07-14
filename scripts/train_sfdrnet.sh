#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data}"

python tools/train.py \
  --dataname weather_tp_5_625 \
  --config_file configs/weather/tp_5_625/SFDRNet.py \
  --data_root "${DATA_ROOT}" \
  --res_dir work_dirs \
  --ex_name SFDRNet
