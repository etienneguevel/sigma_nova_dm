from torch.jit import ignore
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser

from omegaconf import OmegaConf

from sigma_nova_dm import ROOT


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--overwrite", default=False, action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.config_path):
        raise ValueError(f"Path indicated does not exist: {args.config_path}")

    save_path = ROOT / "logs" / args.experiment_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    elif args.overwrite:
        shutil.rmtree(save_path, ignore_errors=True)
        os.makedirs(save_path)

    else:
        raise ValueError(
            f"There is already a folder at {save_path}, please use --overwrite to replace it."
        )

    cfg = OmegaConf.load(args.config_path)
    cfg.group = args.experiment_name
    OmegaConf.save(cfg, save_path / "config.yaml")

    cmd = [
        sys.executable,
        "-m",
        "sigma_nova_dm.evaluation.eval_shu",
        "--config-path",
        str(save_path / "config.yaml"),
    ]

    for i in range(args.num_runs):
        print(f" {i + 1} / {args.num_runs}  Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        print(f"  Run {i + 1} completed successfully")


if __name__ == "__main__":
    main()
