#!/usr/bin/env python3
"""
Hyperparameter exploration script for running eval_shu.py.
This script generates a grid of hyperparameters (lr, classifier, bsz, freeze_backbone),
creates temporary YAML configs, and runs eval_shu.py for each combination.
"""

import argparse
import itertools
import os
import subprocess
import tempfile
import sys
from typing import List
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(description="Explore hyperparameters for eval_shu.py")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/shu_config_cbramod.yaml",
        help="Path to the base configuration YAML file (default: configs/shu_config_cbramod.yaml)"
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[3e-4, 1e-4, 3e-5, 1e-5],
        help="Learning rates to explore"
    )
    parser.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        default=["all_patch_reps", "all_patch_reps_twolayer"],
        help="Classifier architectures to explore"
    )
    parser.add_argument(
        "--bszs",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Batch sizes to explore"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override the number of epochs (useful for fast testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: print the commands and config changes without executing them"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Check if base config exists
    if not os.path.exists(args.base_config):
        print(f"Error: Base config file '{args.base_config}' not found.")
        sys.exit(1)

    # Generate grid
    combinations = list(itertools.product(args.lrs, args.classifiers, args.bszs))
    total_runs = len(combinations)

    print("=" * 80)
    print(f"HYPERPARAMETER EXPLORATION FOR EVAL_SHU")
    print(f"Base Config: {args.base_config}")
    print(f"Parameters to explore:")
    print(f"  lr: {args.lrs}")
    print(f"  classifier: {args.classifiers}")
    print(f"  bsz: {args.bszs}")
    if args.num_epochs is not None:
        print(f"  num_epochs (override): {args.num_epochs}")
    print(f"Total Runs: {total_runs}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Showing all run combinations:")
        for idx, (lr, classifier, bsz) in enumerate(combinations, start=1):
            print(f"  Run {idx}/{total_runs}: lr={lr}, classifier={classifier}, bsz={bsz}")
        print("=" * 80)
        print("[DRY RUN] Done.")
        return

    # Create temporary directory for configurations in the workspace to avoid permission or path issues
    temp_dir = os.path.join(os.getcwd(), "tmp_configs")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Created temp config directory: {temp_dir}\n")

    try:
        for idx, (lr, classifier, bsz) in enumerate(combinations, start=1):
            print(f"\n[{idx}/{total_runs}] Starting run:")
            print(f"  > lr: {lr}")
            print(f"  > classifier: {classifier}")
            print(f"  > bsz: {bsz}")
            
            # Load and update configuration
            cfg = OmegaConf.load(args.base_config)
            cfg.train.lr = lr
            cfg.model.classifier = classifier
            cfg.train.bsz = bsz
            if args.num_epochs is not None:
                cfg.train.num_epochs = args.num_epochs

            # Save modified config to a temporary yaml file
            temp_config_path = os.path.join(temp_dir, f"cfg_run_{idx}_lr{lr}_clf_{classifier}_bsz{bsz}.yaml")
            OmegaConf.save(cfg, temp_config_path)

            # Build command to execute
            # Running as module is clean and handles path resolution
            cmd = [
                sys.executable,
                "-m", "sigma_nova_dm.evaluation.eval_shu",
                "--config-path", temp_config_path
            ]
            
            print(f"  Executing command: {' '.join(cmd)}")
            
            # Execute subprocess and stream output
            try:
                # We run with stdout/stderr going directly to terminal so user can track it
                result = subprocess.run(cmd, check=True)
                print(f"  Run {idx} completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"  Error: Run {idx} failed with exit code {e.returncode}.", file=sys.stderr)
                # Continue with other configurations instead of crashing the whole loop
            except KeyboardInterrupt:
                print("\nExploration interrupted by user. Exiting.")
                break
            finally:
                # Clean up temporary config file
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)

    finally:
        # Clean up temporary directory if it's empty
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("HYPERPARAMETER EXPLORATION FINISHED")
    print("=" * 80)

if __name__ == "__main__":
    main()
