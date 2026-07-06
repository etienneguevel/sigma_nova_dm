import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from sigma_nova_dm.data.shu_dataset import shuDataGenerator, shuOfflineDataset


def get_patient_from_file(file: str) -> int:
    return int(file.split("_")[0][-2:])


def to_tensor(array):
    return torch.from_numpy(array).float()


def collate_fn(batch):

    x_data = np.array([x[0] for x in batch])
    y_label = np.array([x[1] for x in batch])
    return to_tensor(x_data), to_tensor(y_label)


def make_generator(
    data_dir: str,
    patients: list[int],
    bsz: int,
    yield_last: bool,
    sampling_freq: int,
    target_freq: int,
):
    """
    This function creates a data generator that will yield batches in an online fashion.
    All the files corresponding to the patients indices will be used to make the batches.

    Args:
        - data_dir: str, the dir to look for the shu files
        - patients: list[int], the indices of the patients to consider for this data generator
        - bsz: int, the number of elements to pick for each batch
        - yield_last: bool, whether to use the last batch (not of size bsz) or not
        - target_freq: int, freq to use as output. Resample happens if it is different from sample_freq
    """

    files = os.listdir(data_dir)
    patients_files = [
        os.path.join(data_dir, f) for f in files if get_patient_from_file(f) in patients
    ]

    return shuDataGenerator(patients_files, bsz, yield_last, sampling_freq, target_freq)


def make_dataloader(
    data_dir: str,
    db_dir: str,
    patients: list[int],
    bsz: int,
    shuffle: bool,
    sampling_freq: int,
    target_freq: int,
):
    """
    This function creates a torch DataLoader instance that will yield batches in an offline fashion.
    All the files corresponding to the patients indices will be used to make the batches.
    The results are stored within a lmdb database that is computed once, and will be used for later calls.

    Args:
        - data_dir: str, the dir to look for the shu files
        - db_dir: str, path of the lmdb database
        - patients: list[int], the indices of the patients to consider for this data generator
        - bsz: int, the number of elements to pick for each batch
        - shuffle: bool, add some randomness in the elements of the batch
        - target_freq: int, freq to use as output. Resample happens if it is different from sample_freq

    """

    files = os.listdir(data_dir)
    patients_files = [
        os.path.join(data_dir, f) for f in files if get_patient_from_file(f) in patients
    ]

    dataset = shuOfflineDataset(patients_files, db_dir, sampling_freq, target_freq)
    loader = DataLoader(dataset, bsz, shuffle, collate_fn=collate_fn)

    return loader


def make_loaders_from_cfg(cfg):
    # Make the data_generator
    if cfg.data.type == "lmdb":
        train_generator = make_dataloader(
            cfg.data.datadir,
            os.path.join(cfg.data.db_dir, "train"),
            list(range(*cfg.data.train)),
            cfg.train.bsz,
            True,
            cfg.data.sampling_freq,
            cfg.data.target_freq,
        )
        valid_generator = make_dataloader(
            cfg.data.datadir,
            os.path.join(cfg.data.db_dir, "valid"),
            list(range(*cfg.data.valid)),
            cfg.eval.bsz,
            True,
            cfg.data.sampling_freq,
            cfg.data.target_freq,
        )
        test_generator = make_dataloader(
            cfg.data.datadir,
            os.path.join(cfg.data.db_dir, "test"),
            list(range(*cfg.data.test)),
            cfg.eval.bsz,
            True,
            cfg.data.sampling_freq,
            cfg.data.target_freq,
        )

    elif cfg.data.type == "online":
        train_generator = make_generator(
            cfg.data.datadir,
            list(range(*cfg.data.train)),
            cfg.train.bsz,
            True,
            cfg.data.sampling_freq,
            cfg.data.target_freq,
        )
        valid_generator = make_generator(
            cfg.data.datadir,
            list(range(*cfg.data.valid)),
            cfg.eval.bsz,
            True,
            cfg.data.sampling_freq,
            cfg.data.target_freq,
        )
        test_generator = make_generator(
            cfg.data.datadir,
            list(range(*cfg.data.test)),
            cfg.eval.bsz,
            True,
            cfg.data.sampling_freq,
            cfg.data.target_freq,
        )

    else:
        raise ValueError(f"Type of dataloader to use not implemented : {cfg.data.type}")

    return train_generator, valid_generator, test_generator
