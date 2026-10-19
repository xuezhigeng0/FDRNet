"""Lightweight SFDRNet release checks; does not reproduce manuscript metrics."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openstl.datasets import dataset_parameters
from openstl.models import SimVP_Model, SFDRNet, SDSE
from openstl.modules import SODP, GMPI, FDRR
from openstl.utils import load_config


EXPERIMENTS = (
    ("weather_tp_5_625", "configs/weather/tp_5_625/SFDRNet.py", [12, 1, 32, 64]),
    ("weather_tp_2_8125", "configs/weather/tp_2_8125/SFDRNet.py", [12, 1, 64, 128]),
)
MODEL_KEYS = (
    "hid_S", "hid_T", "N_S", "N_T", "model_type", "spatio_kernel_enc",
    "spatio_kernel_dec", "drop", "drop_path",
)


def validate(skip_forward=False, skip_high_res_forward=False):
    # Public manuscript-facing aliases must resolve to the historical implementation
    # classes so checkpoint keys and tensor shapes remain unchanged.
    assert SFDRNet is SimVP_Model
    assert SDSE.__name__ == "SFFE"
    assert SODP.__name__ == "MDP"
    assert GMPI.__name__ == "MPA"
    assert FDRR.__name__ == "FreqBandResidualMix"

    states = []
    for dataname, config_path, expected_shape in EXPERIMENTS:
        parameters = dataset_parameters[dataname]
        assert parameters["in_shape"] == expected_shape, dataname
        assert parameters["data_name"] == "tp", dataname
        assert parameters["pre_seq_length"] == 12, dataname
        assert parameters["aft_seq_length"] == 12, dataname

        config = load_config(str(ROOT / config_path))
        assert config["method"] == "SimVP", config_path
        assert config["model_type"] == "vit", config_path
        model_args = {key: config[key] for key in MODEL_KEYS}
        model = SFDRNet(in_shape=expected_shape, **model_args)
        count = sum(parameter.numel() for parameter in model.parameters())
        assert count == 1_533_606, (dataname, count)
        states.append({key: value.shape for key, value in model.state_dict().items()})
        print(f"{dataname}: config and dataset registered; {count:,} parameters")

        run_forward = not skip_forward and (
            dataname != "weather_tp_2_8125" or not skip_high_res_forward)
        if run_forward:
            model.eval()
            input_shape = (1, *expected_shape)
            with torch.no_grad():
                prediction = model(torch.randn(input_shape))
            assert tuple(prediction.shape) == input_shape, dataname
            assert torch.isfinite(prediction).all(), dataname
            print(f"{dataname}: CPU forward {input_shape} -> {tuple(prediction.shape)}")
        del model

    assert states[0] == states[1], "State-dict keys or tensor shapes differ"
    print(f"State-dict keys and shapes match ({len(states[0])} entries).")
    print("Release smoke checks passed; manuscript metrics were not reproduced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-forward", action="store_true",
        help="Check configs and checkpoint schema without CPU forward passes.")
    parser.add_argument(
        "--skip-high-res-forward", action="store_true",
        help="Skip the 64x128 CPU pass when memory is limited.")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    validate(args.skip_forward, args.skip_high_res_forward)

