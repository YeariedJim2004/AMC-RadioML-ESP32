"""
dataset.py
==========
RadioML 2016.10a Dataset Loader
- 8 modulation type filter
- IQ signal normalization
- SNR-wise index mapping (Phase 4 evaluation এর জন্য)
- Train / Val / Test split (70/15/15)
- PyTorch DataLoader ready
"""
 
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
 
 
# ── 8টা target modulation ──────────────────────────────────────────────────
SELECTED_MODS = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'PAM4', 'WBFM', 'AM-DSB']
 
 
class RadioMLDataset(Dataset):
    """
    RadioML 2016.10a Dataset
 
    Dataset structure:
        key  : ('modulation_name', snr_value)  →  e.g. ('BPSK', -20)
        value: numpy array shape (1000, 2, 128)
               1000 samples, 2 channel (I & Q), 128 time steps
 
    এই class যা করে:
        1. শুধু 8টা modulation রাখে, বাকি বাদ দেয়
        2. IQ data normalize করে (mean=0, std=1)
        3. SNR অনুযায়ী index map তৈরি করে (পরে evaluation এ কাজে লাগবে)
        4. Label encode করে (BPSK=0, QPSK=1, ... AM-DSB=7)
    """
 
    def __init__(self, file_path, selected_mods=None):
        """
        Parameters
        ----------
        file_path    : str  → RML2016.10a_dict.pkl এর path
        selected_mods: list → কোন modulation গুলো রাখবে
                              None দিলে default 8টা নেবে
        """
 
        # ── Modulation list সেট করো ───────────────────────────────────────
        if selected_mods is None:
            self.selected_mods = SELECTED_MODS
        else:
            self.selected_mods = selected_mods
 
        # modulation name → integer label  (BPSK→0, QPSK→1, ...)
        self.mod_to_idx = {mod: i for i, mod in enumerate(self.selected_mods)}
        self.idx_to_mod = {i: mod for mod, i in self.mod_to_idx.items()}
 
        # ── Dataset load করো ─────────────────────────────────────────────
        print(f"[INFO] Loading dataset from: {file_path}")
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f, encoding='latin1')
        print(f"[INFO] Dataset loaded. Total keys: {len(raw_data)}")
 
        # ── Data extract করো ─────────────────────────────────────────────
        data_list   = []   # IQ samples
        label_list  = []   # modulation label (integer)
        snr_list    = []   # SNR value
 
        for (mod, snr), matrix in raw_data.items():
 
            # শুধু selected modulation নাও
            if mod not in self.selected_mods:
                continue
 
            # matrix shape: (1000, 2, 128)
            # প্রতিটা sample আলাদা করো
            for i in range(matrix.shape[0]):
                data_list.append(matrix[i])                  # shape (2, 128)
                label_list.append(self.mod_to_idx[mod])      # integer
                snr_list.append(snr)                         # e.g. -20, -18, ...
 
        # ── numpy array তে রূপান্তর করো ──────────────────────────────────
        self.data   = np.array(data_list,  dtype=np.float32)   # (N, 2, 128)
        self.labels = np.array(label_list, dtype=np.int64)      # (N,)
        self.snrs   = np.array(snr_list,   dtype=np.int32)      # (N,)
 
        print(f"[INFO] Filtered samples: {len(self.data)}")
        print(f"[INFO] Modulations kept: {self.selected_mods}")
        print(f"[INFO] SNR range: {self.snrs.min()} dB to {self.snrs.max()} dB")
 
        # ── IQ Normalization (mean=0, std=1) ─────────────────────────────
        # পুরো dataset এর উপর normalize করা হচ্ছে
        mean = self.data.mean()
        std  = self.data.std()
        self.data = (self.data - mean) / (std + 1e-8)   # 1e-8 দিয়ে division by zero ঠেকানো
        print(f"[INFO] Normalization done. Mean={mean:.4f}, Std={std:.4f}")
 
        # ── SNR-wise Index Map তৈরি করো ──────────────────────────────────
        # Phase 4 evaluation এ প্রতিটা SNR level এ আলাদা accuracy দেখার জন্য
        # snr_index_map[snr_value] = [index1, index2, ...]
        self.snr_index_map = {}
        unique_snrs = sorted(set(self.snrs.tolist()))
        for snr_val in unique_snrs:
            indices = np.where(self.snrs == snr_val)[0].tolist()
            self.snr_index_map[snr_val] = indices
 
        print(f"[INFO] SNR index map ready. Unique SNRs: {unique_snrs}")
 
    def __len__(self):
        """Dataset এ মোট কতটা sample আছে"""
        return len(self.data)
 
    def __getitem__(self, idx):
        """
        একটা sample return করে
 
        Returns
        -------
        iq_tensor   : torch.FloatTensor  shape (2, 128)
        label_tensor: torch.LongTensor   shape ()
        snr_tensor  : torch.IntTensor    shape ()
        """
        iq_tensor    = torch.tensor(self.data[idx],   dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        snr_tensor   = torch.tensor(self.snrs[idx],   dtype=torch.int32)
 
        return iq_tensor, label_tensor, snr_tensor
 
    def get_snr_index_map(self):
        """SNR-wise index map return করে (evaluation এ ব্যবহার হবে)"""
        return self.snr_index_map
 
    def get_class_names(self):
        """Modulation class names return করে"""
        return self.selected_mods
 
 
# ── DataLoader তৈরির function ─────────────────────────────────────────────
def get_loaders(file_path, batch_size=64, seed=42):
    """
    Train / Val / Test DataLoader তৈরি করে
 
    Parameters
    ----------
    file_path  : str → dataset এর path
    batch_size : int → default 64
    seed       : int → reproducibility এর জন্য
 
    Returns
    -------
    train_loader, val_loader, test_loader, full_dataset
    """
 
    # Full dataset load করো
    full_dataset = RadioMLDataset(file_path)
    total = len(full_dataset)
 
    # 70 / 15 / 15 split
    train_len = int(0.70 * total)
    val_len   = int(0.15 * total)
    test_len  = total - train_len - val_len
 
    print(f"\n[INFO] Split → Train: {train_len} | Val: {val_len} | Test: {test_len}")
 
    # random_split দিয়ে ভাগ করো
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=generator
    )
 
    # DataLoader তৈরি করো
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
 
    return train_loader, val_loader, test_loader, full_dataset
 
 
# ── Quick test (এই file সরাসরি run করলে চলবে) ────────────────────────────
if __name__ == '__main__':
 
    # তোমার dataset এর path এখানে দাও
    DATASET_PATH = r'C:\Users\User\AMC-RadioML-ESP32\data\RML2016.10a_dict.pkl'
 
    train_loader, val_loader, test_loader, dataset = get_loaders(DATASET_PATH)
 
    # একটা batch দেখো
    iq, label, snr = next(iter(train_loader))
    print(f"\n[TEST] IQ shape  : {iq.shape}")      # (64, 2, 128)
    print(f"[TEST] Label shape: {label.shape}")    # (64,)
    print(f"[TEST] SNR shape  : {snr.shape}")      # (64,)
    print(f"[TEST] Classes    : {dataset.get_class_names()}")
    print(f"[TEST] SNR map keys: {list(dataset.get_snr_index_map().keys())}")
 