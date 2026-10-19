#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data}"

python tools/test.py \
  --test \
  --dataname weather_tp_2_8125 \
  --config_file configs/weather/tp_2_8125/SFDRNet.py \
  --data_root "${DATA_ROOT}" \
  --res_dir work_dirs \
  --ex_name SFDRNet_tp_2_8125
