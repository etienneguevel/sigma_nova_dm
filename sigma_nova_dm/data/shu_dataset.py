import os
import pickle

import lmdb
import scipy
import torch
from torch.utils.data import Dataset


class shuData:
    def __init__(
        self, files: list[str], sampling_freq: int, target_freq: int | None = None
    ):
        super().__init__()
        self.files = files
        self.sampling_freq = sampling_freq
        self.target_freq = target_freq if target_freq else self.sampling_freq

    def open_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise ValueError(f"There is not file at indicated {file_path}")

        # Open the file
        data = scipy.io.loadmat(file_path)
        eeg = data["data"]
        labels = data["labels"][0]

        # Resize it
        bz, ch_num, points = eeg.shape
        target_points = int((points / self.sampling_freq) * self.target_freq)
        eeg = scipy.signal.resample(eeg, target_points, axis=2)

        eeg = eeg.reshape(bz, ch_num, -1, self.target_freq)

        # Normalize the values
        eeg = eeg / 100
        labels = labels - 1

        return eeg, labels


class shuDataGenerator(shuData):
    def __init__(
        self,
        files: list[str],
        bsz: int,
        yield_last: bool,
        sampling_freq: int,
        target_freq: int | None = None,
    ):
        super().__init__(files, sampling_freq, target_freq)
        self.bsz = bsz
        self.yield_last = yield_last

    def __iter__(self):

        signals = torch.tensor([])
        labs = torch.tensor([])

        for f in self.files:
            eeg, labels = self.open_file(f)
            while len(eeg) > (rest := (self.bsz - len(signals))):
                signals = torch.cat([signals, torch.from_numpy(eeg[:rest, ...])])

                labs = torch.cat([labs, torch.from_numpy(labels[:rest, ...])])

                eeg = eeg[rest:, ...]
                labels = labels[rest:, ...]

                yield signals, labs
                signals = torch.tensor([])
                labs = torch.tensor([])

            signals = torch.cat([signals, torch.from_numpy(eeg)])
            labs = torch.cat([labs, torch.from_numpy(labels)])

        if self.yield_last:
            yield signals, labs


class shuOfflineDataset(shuData, Dataset):
    def __init__(
        self,
        files: list[str],
        db_dir: str,
        sampling_freq: int,
        target_freq: int | None = None,
    ):
        super().__init__(files, sampling_freq, target_freq)

        self.metadata = {
            "sampling_freq": sampling_freq,
            "target_freq": target_freq,
        }

        if os.path.exists(db_dir):
            if "data.mdb" in os.listdir(db_dir):
                pass

            else:
                self._write_db(db_dir)

        else:
            os.makedirs(db_dir)
            self._write_db(db_dir)

        self.db = lmdb.open(
            db_dir, readonly=True, lock=False, readahead=True, meminit=False
        )

        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get("__keys__".encode()))
            metadata = pickle.loads(txn.get("__metadata__".encode()))

        for key, value in self.metadata.items():
            if (m_val := metadata.get(key)) != value:
                raise ValueError(
                    f"There is already a db at {db_dir} with different value {m_val} for {key}.",
                    "Please manually erase the previous db or indicate a different path.",
                )

    def __len__(self):
        return len((self.keys))

    def _write_db(self, db_dir: str):
        db = lmdb.open(db_dir, map_size=111111111111)
        list_keys = []

        with db.begin(write=True) as txn:
            for file in self.files:
                eeg_, labels = self.open_file(file)
                file_name = file.split("/")[-1][:-4]
                for i, (sample, label) in enumerate(zip(eeg_, labels)):
                    sample_key = f"{file_name}-{i}"
                    data = {"sample": sample, "label": label}
                    txn.put(key=sample_key.encode(), value=pickle.dumps(data))
                    list_keys.append(sample_key)

            txn.put(key="__keys__".encode(), value=pickle.dumps(list_keys))
            txn.put(key="__metadata__".encode(), value=pickle.dumps(self.metadata))

        db.close()

    def __getitem__(self, index: int):
        key = self.keys[index]

        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))

        data = pair["sample"]
        label = pair["label"]
        return data, label
