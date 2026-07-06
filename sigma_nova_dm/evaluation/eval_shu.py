import copy
import os
import pickle
import random
import time
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from torch.optim import AdamW
from omegaconf import OmegaConf
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

from tqdm import tqdm

from sigma_nova_dm import ROOT
from sigma_nova_dm.data.shu_loaders import make_loaders_from_cfg
from sigma_nova_dm.models.eegsimpleconv import EEGSimpleConv
from sigma_nova_dm.models.model_for_shu import Model


warnings.filterwarnings("ignore")


CBRAMOD_SEED = 3407


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--bsz", type=int, required=False, default=None)
    parser.add_argument("--set-seed", default=False, action="store_true")
    parser.add_argument("--no-wandb", default=False, action="store_true")

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def do_eval(model: torch.nn.Module, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    for b in data_loader:
        # Forward
        b = (el.to(device) for el in b)
        eeg, labels = b

        with torch.no_grad():
            logits = model(eeg)
            scores = logits.sigmoid()
            preds = torch.gt(scores, 0.5).long()

        # Append the results
        all_preds.append(preds.to("cpu"))
        all_labels.append(labels.to("cpu"))
        all_scores.append(scores.to("cpu"))

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)
    # Compute the metrics
    acc = balanced_accuracy_score(all_labels, all_preds)

    roc_auc = roc_auc_score(all_labels, all_scores)
    precision, recall, ths = precision_recall_curve(all_labels, all_scores, pos_label=1)
    pr_auc = auc(recall, precision)

    cm = wandb.plot.confusion_matrix(
        probs=None, y_true=all_labels.tolist(), preds=all_preds.tolist()
    )

    return acc, roc_auc, pr_auc, cm


def main():
    args = get_args()
    if args.set_seed:
        print(f"Setting random seed : {CBRAMOD_SEED}")
        setup_seed(CBRAMOD_SEED)

    cfg = OmegaConf.load(args.config_path)

    if args.bsz:
        cfg.train.bsz = int(args.bsz)

    # Make the data_generator
    print(f"Using the dataset : {cfg.data.type}")
    train_generator, valid_generator, test_generator = make_loaders_from_cfg(cfg)

    # Init the model and load the backbone
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps") if torch.mps.is_available() else (torch.device("cpu"))
        )
    )

    print(f"Using device: {device}")
    print("-" * 80, "\n")

    if cfg.model.arch == "cbramod":
        model = Model(cfg.model, device)
        for name, p in model.named_parameters():
            if "backbone" in name:
                setattr(p, "requires_grad", not cfg.model.freeze_backbone)

            else:
                setattr(p, "requires_grad", True)

    elif cfg.model.arch == "eeg_conv":
        model = EEGSimpleConv(
            cfg.model.fm,
            cfg.model.n_convs,
            cfg.model.kernel_size,
            cfg.model.n_chan,
            cfg.model.n_classes,
        )

    else:
        raise ValueError(f"{cfg.model.arch} is not an accepted model type.")

    model = model.to(device)

    if cfg.model.arch == "eeg_conv":
        run_name = f"shu-model:eeg_conv-fm:{cfg.model.fm}-n_convs:{cfg.model.n_convs}-kernel:{cfg.model.kernel_size}-lr:{cfg.train.lr}-bsz:{cfg.train.bsz}"
    else:
        run_name = f"shu-model:{cfg.model.arch}-classifier:{cfg.model.get('classifier')}-lr:{cfg.train.lr}-bsz:{cfg.train.bsz}"

    # Init the wandb run
    mode = "disabled" if args.no_wandb else "online"
    wandb.init(
        project="sigma-nova-dm",
        group=cfg.group,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=mode,
    )

    print("MODEL INFORMATION", "\n")
    print("-" * 80, "\n")
    trainable_parameters = (
        sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1e6
    )
    total_parameters = sum([p.numel() for p in model.parameters()]) / 1e6

    print(
        "Number of trainable parameters:",
        f"{trainable_parameters} / {total_parameters}",
        "millions",
    )
    print("-" * 80, "\n")
    print(model)

    # Init the optimizer and pick the loss criterion
    optimizer = AdamW(
        [
            {
                "params": [
                    p for name, p in model.named_parameters() if "backbone" in name
                ],
                "lr": cfg.train.lr_backbone,
            },
            {
                "params": [
                    p for name, p in model.named_parameters() if "backbone" not in name
                ],
                "lr": cfg.train.lr * (cfg.train.bsz / 256) ** 0.5,
            },
        ],
        weight_decay=cfg.train.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    len_data_gen = 1 if cfg.data.type == "online" else len(train_generator)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.num_epochs * len_data_gen, eta_min=1e-6
    )

    # Train the model
    acc = None
    best_roc_auc = -float("inf")
    log_metrics = {"val": []}

    for e in tqdm(range(cfg.train.num_epochs), desc="Epoch"):
        model.train()
        for b in (pbar := tqdm(train_generator, desc="Batch", leave=False)):
            batch_start_time = time.time()
            optimizer.zero_grad()

            # Forward
            b = (el.to(device) for el in b)
            eeg, labels = b
            logits = model(eeg)

            # Backward
            loss = criterion(logits, labels)
            loss.backward()

            if cfg.train.grad_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clipping
                )

            optimizer.step()
            scheduler.step()

            # Log the results
            pbar.set_postfix({"loss": loss.item(), "last_eval_acc": acc})
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/batch_time": time.time() - batch_start_time,
                    "train/lr_backbone": optimizer.param_groups[0]["lr"],
                    "train/lr_other": optimizer.param_groups[1]["lr"],
                }
            )

        acc, roc_auc, pr_auc, _ = do_eval(model, valid_generator, device)

        val_data = {"eval/acc": acc, "eval/roc_auc": roc_auc, "eval/pr_auc": pr_auc}
        log_metrics["val"].append(val_data)

        wandb.log(val_data)

        if roc_auc > best_roc_auc:
            best_epoch = e
            best_roc_auc = roc_auc
            best_model_state = copy.deepcopy(model.state_dict())

    # Use the test generator on the trained model
    print(f"Using the weight from the epoch {best_epoch}")
    model.load_state_dict(best_model_state)
    acc, roc_auc, pr_auc, cm = do_eval(model, test_generator, device)

    print("***************************Test results************************")
    print(
        "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
            acc,
            pr_auc,
            roc_auc,
        )
    )
    print(cm)

    test_data = {
        "test/acc": acc,
        "test/pr_auc": pr_auc,
        "test/roc_auc": roc_auc,
    }

    wandb.log(test_data)
    wandb.log({"test/cm": cm})

    log_metrics["test"] = test_data

    if cfg.get("log_metrics"):
        folder_path = ROOT / "logs" / cfg.group / str(wandb.run.id)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Saving the metric file at {folder_path}")
        with open(folder_path / "metrics.pkl", "wb") as f:
            pickle.dump(log_metrics, f)

        OmegaConf.save(cfg, folder_path / "config.yaml")


if __name__ == "__main__":
    main()
