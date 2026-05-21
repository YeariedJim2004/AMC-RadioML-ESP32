"""
dataset.py
==========
RadioML 2016.10a Dataset Loader
- 8 modulation type filter
- SNR range filter (snr_min parameter)
- IQ signal normalization
- SNR-wise index mapping (Phase 4 evaluation ?? ????)
- Train / Val / Test split (70/15/15)
- PyTorch DataLoader ready
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# -- 8?? target modulation ------------------------------------------------------
SELECTED_MODS = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4', 'WBFM', 'AM-DSB']


class RadioMLDataset(Dataset):
    def __init__(self, file_path, selected_mods=None, snr_min=-20):
        if selected_mods is None:
            self.selected_mods = SELECTED_MODS
        else:
            self.selected_mods = selected_mods

        self.mod_to_idx = {mod: i for i, mod in enumerate(self.selected_mods)}
        self.idx_to_mod = {i: mod for mod, i in self.mod_to_idx.items()}

        print(f"[INFO] Loading dataset from: {file_path}")
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f, encoding='latin1')
        print(f"[INFO] Dataset loaded. Total keys: {len(raw_data)}")

        data_list  = []
        label_list = []
        snr_list   = []

        for (mod, snr), matrix in raw_data.items():
            if mod not in self.selected_mods:
                continue
            if snr < snr_min:          # SNR filter
                continue
            for i in range(matrix.shape[0]):
                data_list.append(matrix[i])
                label_list.append(self.mod_to_idx[mod])
                snr_list.append(snr)

        print(f"[INFO] Filtered samples: {len(data_list)}")
        print(f"[INFO] Modulations kept: {self.selected_mods}")

        # numpy array
        data_arr  = np.array(data_list,  dtype=np.float32)
        label_arr = np.array(label_list, dtype=np.int64)
        snr_arr   = np.array(snr_list,   dtype=np.int64)

        # Normalization (mean=0, std=1)
        mean = data_arr.mean()
        std  = data_arr.std()
        data_arr = (data_arr - mean) / (std + 1e-8)
        print(f"[INFO] Normalization done. Mean={mean:.4f}, Std={std:.4f}")

        # SNR index map
        unique_snrs = sorted(set(snr_list))
        print(f"[INFO] SNR range: {unique_snrs[0]} dB to {unique_snrs[-1]} dB")
        self.snr_index_map = {}
        for idx, s in enumerate(snr_arr):
            self.snr_index_map.setdefault(int(s), []).append(idx)
        print(f"[INFO] SNR index map ready. Unique SNRs: {unique_snrs}")

        self.data   = torch.tensor(data_arr)
        self.labels = torch.tensor(label_arr)
        self.snrs   = torch.tensor(snr_arr)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.snrs[idx]


def get_loaders(file_path, batch_size=64, seed=42, snr_min=-4):
    dataset = RadioMLDataset(file_path, snr_min=snr_min)

    total   = len(dataset)
    n_train = int(0.70 * total)
    n_val   = int(0.15 * total)
    n_test  = total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    print(f"[INFO] Split ? Train: {n_train} | Val: {n_val} | Test: {n_test}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, dataset
