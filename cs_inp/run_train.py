import torch
import deepinv as dinv
import argparse
import yaml
from train import train

def replace_metrics(d):
    if isinstance(d, dict):
        return {k: replace_metrics(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_metrics(v) for v in d]
    elif isinstance(d, str):
        if d == "MSELoss":
            return torch.nn.MSELoss()
        else:
            return d
    else:
        return d

if __name__ == "__main__":
    # argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config_cs_split_equi.yaml",
                        help="Chemin vers le fichier de config YAML")
    args = parser.parse_args()

    # read the YAML file
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config = replace_metrics(config)

    torch.manual_seed(0)
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    train(device, **config)
