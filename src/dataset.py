import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

SELECTED_MODS = [
    'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
    'PAM4', 'WBFM', 'AM-DSB', 'AM-SSB', 'GFSK', 'CPFSK'
]

MOD_TO_IDX = {mod: idx for idx, mod in enumerate(SELECTED_MODS)}


def compute_instantaneous_frequency(iq):
    I = iq[0]
    Q = iq[1]
    phase = np.arctan2(Q, I)
    phase_unwrapped = np.unwrap(phase)
    inst_freq = np.diff(phase_unwrapped, prepend=phase_unwrapped[0])
    std = inst_freq.std()
    if std > 1e-8:
        inst_freq = (inst_freq - inst_freq.mean()) / std
    return inst_freq.astype(np.float32)


class RadioMLDataset(Dataset):
    def __init__(self, file_path, selected_mods=None, snr_min=-20):
        if selected_mods is None:
            selected_mods = SELECTED_MODS

        with open(file_path, 'rb') as f:
            raw = pickle.load(f, encoding='latin1')

        self.samples = []

        for (mod, snr), data in raw.items():
            if mod not in selected_mods:
                continue
            if snr < snr_min:
                continue

            label = MOD_TO_IDX[mod]

            for i in range(data.shape[0]):
                iq = data[i]
                inst_freq = compute_instantaneous_frequency(iq)
                iq_3ch = np.vstack([iq, inst_freq[np.newaxis, :]])
                self.samples.append((iq_3ch, label, snr))

        print(f"[Dataset] Loaded {len(self.samples)} samples | "
              f"Mods: {len(selected_mods)} | SNR >= {snr_min} dB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iq_3ch, label, snr = self.samples[idx]
        iq_tensor    = torch.tensor(iq_3ch, dtype=torch.float32)
        label_tensor = torch.tensor(label,  dtype=torch.long)
        snr_tensor   = torch.tensor(snr,    dtype=torch.float32)
        return iq_tensor, label_tensor, snr_tensor


def get_loaders(file_path, batch_size=256, seed=42, snr_min=-4):
    dataset = RadioMLDataset(file_path, snr_min=snr_min)

    total      = len(dataset)
    train_size = int(0.70 * total)
    val_size   = int(0.15 * total)
    test_size  = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"[Split] Train: {train_size} | Val: {val_size} | Test: {test_size}")

    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, len(SELECTED_MODS)

