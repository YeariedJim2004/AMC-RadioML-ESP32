import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split

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

def augment_iq(iq):
    I = iq[0].copy()
    Q = iq[1].copy()
    if np.random.rand() < 0.5:
        snr_db = np.random.uniform(10, 30)
        signal_power = np.mean(I**2 + Q**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)
        I += np.random.normal(0, noise_std, I.shape).astype(np.float32)
        Q += np.random.normal(0, noise_std, Q.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        scale = np.random.rayleigh(scale=1.0)
        scale = np.clip(scale, 0.3, 2.0)
        I *= scale
        Q *= scale
    if np.random.rand() < 0.5:
        cfo = np.random.uniform(-0.05, 0.05)
        t = np.arange(len(I))
        phase_shift = 2 * np.pi * cfo * t
        I_new = I * np.cos(phase_shift) - Q * np.sin(phase_shift)
        Q_new = I * np.sin(phase_shift) + Q * np.cos(phase_shift)
        I, Q = I_new.astype(np.float32), Q_new.astype(np.float32)
    if np.random.rand() < 0.5:
        phase_noise = np.random.normal(0, 0.05, len(I)).astype(np.float32)
        I_new = I * np.cos(phase_noise) - Q * np.sin(phase_noise)
        Q_new = I * np.sin(phase_noise) + Q * np.cos(phase_noise)
        I, Q = I_new, Q_new
    return np.vstack([I[np.newaxis, :], Q[np.newaxis, :]])

class RadioMLDataset(Dataset):
    def __init__(self, file_path, selected_mods=None, snr_min=-20, augment=False):
        if selected_mods is None:
            selected_mods = SELECTED_MODS
        with open(file_path, 'rb') as f:
            raw = pickle.load(f, encoding='latin1')
        self.augment = augment
        self.samples = []
        for (mod, snr), data in raw.items():
            if mod not in selected_mods:
                continue
            if snr < snr_min:
                continue
            label = MOD_TO_IDX[mod]
            for i in range(data.shape[0]):
                self.samples.append((data[i], label, snr))
        print(f'[Dataset] Loaded {len(self.samples)} samples | Mods: {len(selected_mods)} | SNR >= {snr_min} dB | Augment: {augment}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iq, label, snr = self.samples[idx]
        iq_used = augment_iq(iq) if self.augment else iq.copy()
        inst_freq = compute_instantaneous_frequency(iq_used)
        iq_3ch = np.vstack([iq_used, inst_freq[np.newaxis, :]])
        return torch.tensor(iq_3ch, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(snr, dtype=torch.float32)

def get_loaders(file_path, batch_size=256, seed=42, snr_min=-4):
    base = RadioMLDataset(file_path, snr_min=snr_min, augment=False)
    total = len(base)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    splits = random_split(range(total), [train_size, val_size, test_size], generator=generator)
    train_idx, val_idx, test_idx = list(splits[0]), list(splits[1]), list(splits[2])
    train_ds = Subset(RadioMLDataset(file_path, snr_min=snr_min, augment=True),  train_idx)
    val_ds   = Subset(RadioMLDataset(file_path, snr_min=snr_min, augment=False), val_idx)
    test_ds  = Subset(RadioMLDataset(file_path, snr_min=snr_min, augment=False), test_idx)
    print(f'[Split] Train: {train_size} | Val: {val_size} | Test: {test_size}')
    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    return DataLoader(train_ds, shuffle=True, **kwargs), DataLoader(val_ds, shuffle=False, **kwargs), DataLoader(test_ds, shuffle=False, **kwargs), len(SELECTED_MODS)
