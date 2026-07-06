# Sigma-nova-dm

This repo is an implementation of two Deep Learning models for eeg signals:
- [CBraMod](https://github.com/wjq-learning/CBraMod)
- [EEGSimpleConv](https://github.com/elouayas/EEGSimpleConv/tree/main)

The package focuses on training the model on the `shu-mi` dataset, that can be
loaded from [HERE](https://figshare.com/articles/code/shu_dataset/19228725).

## Installation

This package is managed by `uv` but can be installed through `pip` or other
python env managers. To simply setup the env you can run :
```Bash
# first clone the repo
git clone https://github.com/etienneguevel/sigma_nova_dm.git
cd sigma_nova_dm

# make the venv
uv sync
source .venv/bin/activate
uv pip install -e .
```

## Data loading

To load the data run the `scripts/download_shu_data.sh`script :
```Bash
chmod +x ./scripts/download_shu_data.sh && ./scripts/download_shu_data.sh
```

## Run

To run an experiment the `sigma_nova_dm/evaluation/eval_shu.py` should be used,
with the `--config-path` pointing to the config file for the desired experiment.

### Config file

To select the hyperparameters of the model / training / data preprocessing,
config files in a `yaml` format are used. You can find an example of a config
template file at `shu_config_template.yaml`. The user should especially focus on
the following keys as they define paths and might differ :
- `data/datadir: mat` -> if you used the data loading script you should have the `mat` folder, else indicate the path to the raw `shu-mi` files.
- `data/db_dir` -> path to save the `lmdb` data objects.
- `model/foundation_dir` -> path leading the CBraMod backbone weights.

### Logging

The eval script results are logged using `wandb`, but if the user does not want
to track his runs `wandb` can be disabled with the `--no-wandb` flag.

### Scripts

Three scripts exist :
- `explore_shu_eegconv.py` that explores the hyperparameter space for the
    `EEGSimpleConv` model.
- `explore_shu_cbramod.py` that explores the hyperparameter space for the
    `CBraMod` model.
- `stability_check_shu.py` that reproduces `n` times (default 10) a setting with
    different seeds to check the randomness variability of the setup.
